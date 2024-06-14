import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import random
import logging
import argparse
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(level=logging.INFO)

from src.utils import read_qrels
from src.models import (
    CombinedNet,
    MC_Sampling_Net,
    Explanation_Text_Net,
    Conditional_Explanation_Text_Net
)
from src.dataloaders import (
    All_Purpose_Dataset,
    SameQIDBatchSampler
)

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # general arguments, not related to input data
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_epochs", default=15, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--max_score", default=3, type=int)
    parser.add_argument("--output_size", default=1, type=int)
    parser.add_argument("--weighted_sampling", action="store_true")
    parser.add_argument("--model_selection_metric", type=str, default="val/loss")
    parser.add_argument("--same_qid_training", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--criterion", type=str, default="mse")  # mse, ce, mse+ce, cce
    parser.add_argument("--dataset", type=str, default="trec", choices=["trec", "clueweb"])
    parser.add_argument("--calibration_type", type=str, choices=[
        "post-hoc",
        "mc-sampling",
        "pred-exp",
        "cond-exp"
    ])
    parser.add_argument("--sampling_for_testing", action="store_true")
    parser.add_argument("--clueweb_fold", type=int, default=3, choices=[0, 1, 2, 3, 4])

    # post-hoc calibration arguments
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--use_logit", action="store_true")
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--pretrained_model", default=None, choices=["t5", "bert"])
    
    # explanation scalar features arguments
    parser.add_argument("--use_gru", action="store_true")

    # prediction + explanation arguments
    parser.add_argument("--ranker_model_ckpt", default="distilbert-base-uncased")  # also used for cond-exp
    parser.add_argument("--use_bert_score", action="store_true")  # also used for cond-exp
    parser.add_argument("--pred_explanation_type", choices=[
        "qd_pair",
        "pred_exp_most_likely_exp",
        "pred_exp_most_likely_exp_oracle",
        "pred_exp_most_likely_exp_diverse_max_30_0.35"
    ], default=None)

    # conditional explanation arguments
    parser.add_argument("--tie_weights", action="store_true")
    parser.add_argument("--cond_explanation_type", choices=["first", "diverse"])

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    # sanity checks
    if args.weighted_sampling and args.same_qid_training:
        assert False, "Cannot have both weighted sampling and same qid training"

    if "ce" in args.criterion and not args.same_qid_training:
        assert False, "Cross entropy loss requires same qid training"

    if args.dataset == "clueweb" and args.max_score !=  4:
        # assert False, "Clueweb dataset has 5 relevance levels"
        print("Clueweb dataset has 5 relevance levels!")
        args.max_score = 4

    if args.dataset == "trec" and args.max_score !=  3:
        # assert False, "TREC dataset has 4 relevance levels"
        print("TREC dataset has 4 relevance levels!")
        args.max_score = 3

    # if args.calibration_type == "post-hoc" and args.criterion != "mse":
    #     assert False, "Post-hoc calibration only needs MSE loss"


    # query splits
    if args.dataset == "trec":
        # Load the data (train 19/20, valid 21, test 22)
        query_id_splits = {
            "train": set(read_qrels("msmarco-passage/trec-dl-2019/judged").keys())
            | set(read_qrels("msmarco-passage/trec-dl-2020/judged").keys()),
            "val": set(read_qrels("msmarco-passage-v2/trec-dl-2021/judged").keys()),
            "test": set(read_qrels("msmarco-passage-v2/trec-dl-2022/judged").keys()),
        }
    elif args.dataset == "clueweb":
        
        all_qids = sorted(list(read_qrels("clueweb12/b13/ntcir-www-2").keys()))
        fold_size = int(len(all_qids) / 5)
        folds = {}
        
        for i in range(5):
            start_index = i * fold_size
            end_index = start_index + fold_size if i < 4 else len(all_qids)
            folds[i] = all_qids[start_index:end_index]
        
        query_id_splits = {
            "train": folds[args.clueweb_fold % 5] + folds[(args.clueweb_fold + 1) % 5] + folds[(args.clueweb_fold + 2) % 5],
            "val": folds[(args.clueweb_fold + 3) % 5],
            "test": folds[(args.clueweb_fold + 4) % 5],
        }
    else:
        assert False, "Invalid dataset"

    

    # load input data
    all_data = []
    logging.info("Loading processed data...")
    with open(f"processed_data/merged_{args.dataset}.jsonl", "r") as file:
        for line in file:
            item = json.loads(line)
            all_data.append(item)
    logging.info("Done loading processed data!")

    if args.sampling_for_testing:
        all_data = random.sample(all_data, int(len(all_data) * 0.1))

    # build train, val, test datasets
    train_dataset = All_Purpose_Dataset(
        data_list=all_data,
        query_id_splits=query_id_splits,
        split="train",
        tokenzier_name=args.ranker_model_ckpt,
    )
    weights = train_dataset.calculate_class_weights(just_log=False)
    val_dataset = All_Purpose_Dataset(
        data_list=all_data,
        query_id_splits=query_id_splits,
        split="val",
        tokenzier_name=args.ranker_model_ckpt,
    )
    val_dataset.calculate_class_weights(just_log=True)
    test_dataset = All_Purpose_Dataset(
        data_list=all_data,
        query_id_splits=query_id_splits,
        split="test",
        tokenzier_name=args.ranker_model_ckpt,
    )
    test_dataset.calculate_class_weights(just_log=True)

    if args.weighted_sampling:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True),
            collate_fn=train_dataset.collate_fn,
        )
    elif args.same_qid_training:
        sampler = SameQIDBatchSampler(
            data_source=train_dataset.data, batch_size=args.batch_size
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, collate_fn=train_dataset.collate_fn)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn
    )

    wandb_logger = WandbLogger(
        project=f"SIGIR24-{args.calibration_type}-new",
        # log_model="all",
        config=vars(args),
        dir=f"wandb_storage/{args.calibration_type}",
    )

    if args.calibration_type == "post-hoc": 
        model = CombinedNet(
            input_dimension_size=train_dataset[0][f"{args.pretrained_model}_hidden_state"].shape[0],
            pretrained_model=args.pretrained_model,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            lr=args.learning_rate,
            max_score=args.max_score,
            output_size=args.output_size,
            logger=wandb_logger,
            use_cls=args.use_cls,
            use_logit=args.use_logit,
            crit=args.criterion,
        )
    elif args.calibration_type == "mc-sampling":
        model = MC_Sampling_Net(
            output_size=args.output_size,
            hidden_size=args.hidden_size,
            dropout_rate=args.dropout_rate,
            lr=args.learning_rate,
            max_score=args.max_score,
            logger=wandb_logger,
            crit=args.criterion,
            use_gru=args.use_gru,
        )
        
    elif args.calibration_type == "pred-exp":
        model = Explanation_Text_Net(
            model_ckpt=args.ranker_model_ckpt,
            explanation_type=args.pred_explanation_type,
            output_size=args.output_size,
            dropout_rate=args.dropout_rate,
            lr=args.learning_rate,
            max_score=args.max_score,
            logger=wandb_logger,
            crit=args.criterion,
        )
    elif args.calibration_type == "cond-exp":
        model = Conditional_Explanation_Text_Net(
            model_ckpt=args.ranker_model_ckpt,
            explanation_type=args.cond_explanation_type,
            output_size=args.output_size,
            lr=args.learning_rate,
            max_score=args.max_score,
            dropout_rate=args.dropout_rate,
            tie_weights=args.tie_weights,
            logger=wandb_logger,
            crit=args.criterion,
        )
    else:
        assert False, "Invalid calibration type"

    if (
        args.model_selection_metric == "val/loss"
        or args.model_selection_metric == "val/aece"
    ):
        callback_metric_mode = "min"
    elif args.model_selection_metric == "val/ndcg":
        callback_metric_mode = "max"
    else:
        assert False, "Invalid model selection metric"
    checkpoint_callback = ModelCheckpoint(
        monitor=args.model_selection_metric, mode=callback_metric_mode, save_top_k=1
    )
    early_stop_callback = EarlyStopping(
        monitor=args.model_selection_metric,
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode=callback_metric_mode,
        check_on_train_epoch_end=False,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
        # num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")
