import torch
import pickle
import numpy as np
import lightning as L
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
from .losses import CalibratedListwiseSoftmaxLoss
from .utils import ECE_Calculator, class_balanced_ece
from .AdaptiveBinning import AdaptiveBinningForRegression
from transformers import AutoModelForSequenceClassification, AutoModel


class BasicNet(L.LightningModule):
    def __init__(
        self,
        output_size=1,
        dropout_rate=0.8,
        lr=1e-3,
        max_score=3,
        crit="mse",
        logger=None,
    ):
        super(BasicNet, self).__init__()

        # define model architecture
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.max_score = max_score
        self.crit = crit

        # configure calibrated softmax loss if used
        if self.crit == "cce":
            self.cce = CalibratedListwiseSoftmaxLoss()

        self.reset_buffers(["training", "validation", "testing"])
        self.logger_obj = logger

    def reset_buffers(self, sets=["training", "validation", "testing"]):
        if "training" in sets:
            self.training_buffers = {
                "predictions": [],
                "labels": [],
                "qids": [],
                "logits": [],
            }
        if "validation" in sets:
            self.validation_buffers = {
                "predictions": [],
                "labels": [],
                "qids": [],
                "logits": [],
            }
        if "testing" in sets:
            self.testing_buffers = {
                "predictions": [],
                "labels": [],
                "qids": [],
                "logits": [],
            }

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        predictions, logits = self(batch)
        labels = batch["label"]

        loss = self.compute_loss(predictions, logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # accuracy = self.calculate_accuracy(logits, labels)
        # self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_buffers["predictions"].append(predictions)
        self.training_buffers["logits"].append(logits)
        self.training_buffers["labels"].append(labels)
        self.training_buffers["qids"].extend(batch["qid"])

        return loss

    def validation_step(self, batch, batch_idx):
        predictions, logits = self(batch)
        labels = batch["label"]

        loss = self.compute_loss(predictions, logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # accuracy = self.calculate_accuracy(logits, labels)
        # self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.validation_buffers["predictions"].append(predictions)
        self.validation_buffers["logits"].append(logits)
        self.validation_buffers["labels"].append(labels)
        self.validation_buffers["qids"].extend(batch["qid"])

        return loss

    def test_step(self, batch, batch_idx):
        predictions, logits = self(batch)
        labels = batch["label"]

        self.testing_buffers["predictions"].append(predictions)
        self.testing_buffers["logits"].append(logits)
        self.testing_buffers["labels"].append(labels)
        self.testing_buffers["qids"].extend(batch["qid"])

    def on_train_epoch_end(self):
        
        # if len(self.training_buffers["predictions"]) == 0:
        #     # TODO: for some reason in istella22, this is called before training_step
        #     return
        
        all_preds = torch.cat(self.training_buffers["predictions"], dim=0)
        all_logits = torch.cat(self.training_buffers["logits"], dim=0)
        all_labels = torch.cat(self.training_buffers["labels"], dim=0)

        ece = self.calculate_ece(all_preds, all_labels)
        cb_ece_vals = class_balanced_ece(all_preds, all_labels)
        cb_ece = np.mean(list(cb_ece_vals.values()))
        aece = self.calculate_adaptive_ece(all_preds, all_labels)
        ndcg_full_dct, ndcg_at_10_dct = self.calculate_ndcg(
            all_preds, all_labels, self.training_buffers["qids"]
        )
        ndcg_score = np.mean(list(ndcg_full_dct.values()))
        ndcg_at_10_score = np.mean(list(ndcg_at_10_dct.values()))

        self.log("train/ece", ece, on_step=False, on_epoch=True, logger=True)
        self.log("train/cb_ece", cb_ece, on_step=False, on_epoch=True, logger=True)
        self.log("train/ndcg", ndcg_score, on_step=False, on_epoch=True, logger=True)
        self.log("train/ndcg@10", ndcg_at_10_score, on_step=False, on_epoch=True, logger=True)
        for k, v in cb_ece_vals.items():
            self.log(f"train/cb_ece_{k}", v, on_step=False, on_epoch=True, logger=True)
        self.log("train/aece", aece, on_step=False, on_epoch=True, logger=True)

        label_distribution = self.calculate_label_distribution(all_preds)
        for label, percentage in label_distribution.items():
            self.log(
                f"train/label_distribution_{label}",
                percentage,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.reset_buffers(["training"])

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_buffers["predictions"], dim=0)
        all_logits = torch.cat(self.validation_buffers["logits"], dim=0)
        all_labels = torch.cat(self.validation_buffers["labels"], dim=0)

        ece = self.calculate_ece(all_preds, all_labels)
        cb_ece_vals = class_balanced_ece(all_preds, all_labels)
        cb_ece = np.mean(list(cb_ece_vals.values()))
        aece = self.calculate_adaptive_ece(all_preds, all_labels)
        ndcg_full_dct, ndcg_at_10_dct = self.calculate_ndcg(
            all_preds, all_labels, self.validation_buffers["qids"]
        )
        ndcg_score = np.mean(list(ndcg_full_dct.values()))
        ndcg_at_10_score = np.mean(list(ndcg_at_10_dct.values()))

        self.log("val/ece", ece, on_step=False, on_epoch=True, logger=True)
        self.log("val/cb_ece", cb_ece, on_step=False, on_epoch=True, logger=True)
        self.log("val/ndcg", ndcg_score, on_step=False, on_epoch=True, logger=True)
        self.log("val/ndcg@10", ndcg_at_10_score, on_step=False, on_epoch=True, logger=True)
        for k, v in cb_ece_vals.items():
            self.log(f"val/cb_ece_{k}", v, on_step=False, on_epoch=True, logger=True)
        self.log("val/aece", aece, on_step=False, on_epoch=True, logger=True)

        label_distribution = self.calculate_label_distribution(all_preds)
        for label, percentage in label_distribution.items():
            self.log(
                f"val/label_distribution_{label}",
                percentage,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        self.reset_buffers(["validation"])

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.testing_buffers["predictions"], dim=0)
        all_logits = torch.cat(self.testing_buffers["logits"], dim=0)
        all_labels = torch.cat(self.testing_buffers["labels"], dim=0)

        ece = self.calculate_ece(all_preds, all_labels)
        cb_ece_vals = class_balanced_ece(all_preds, all_labels)
        cb_ece = np.mean(list(cb_ece_vals.values()))
        aece = self.calculate_adaptive_ece(all_preds, all_labels)
        # accuracy = self.calculate_accuracy(all_logits, all_labels)
        mse_loss = self.calculate_mse(all_preds, all_labels)
        ndcg_full_dct, ndcg_at_10_dct = self.calculate_ndcg(
            all_preds, all_labels, self.testing_buffers["qids"]
        )
        ndcg_score = np.mean(list(ndcg_full_dct.values()))
        ndcg_at_10_score = np.mean(list(ndcg_at_10_dct.values()))
        print(ndcg_full_dct, ndcg_at_10_dct)

        self.log("test/ece", ece, on_step=False, on_epoch=True, logger=True)
        self.log("test/cb_ece", cb_ece, on_step=False, on_epoch=True, logger=True)
        # self.log("test/accuracy", accuracy, on_step=False, on_epoch=True, logger=True)
        self.log("test/mse_loss", mse_loss, on_step=False, on_epoch=True, logger=True)
        self.log("test/ndcg", ndcg_score, on_step=False, on_epoch=True, logger=True)
        self.log("test/ndcg@10", ndcg_at_10_score, on_step=False, on_epoch=True, logger=True)
        for k, v in cb_ece_vals.items():
            self.log(f"test/cb_ece_{k}", v, on_step=False, on_epoch=True, logger=True)
        self.log("test/aece", aece, on_step=False, on_epoch=True, logger=True)

        label_distribution = self.calculate_label_distribution(all_logits)
        for label, percentage in label_distribution.items():
            self.log(
                f"test/label_distribution_{label}",
                percentage,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        with open(
            f"{self.logger_obj.experiment.dir}/test_results_ep{self.current_epoch}.pkl",
            "wb",
        ) as fout:
            pickle.dump(
                {
                    "predictions": all_preds.detach().cpu().numpy(),
                    "labels": all_labels.detach().cpu().numpy(),
                    "qids": self.testing_buffers["qids"],
                    "logits": all_logits.detach().cpu().numpy(),
                },
                fout,
            )

        self.reset_buffers(["testing"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def calculate_accuracy(self, logits, targets):
    #     # predictions = torch.round(predictions)
    #     predictions = torch.argmax(logits, dim=-1, keepdim=True)
    #     correct = (predictions == targets).sum().item()
    #     accuracy = correct / len(targets)
    #     return accuracy

    def calculate_mse(self, predictions, targets):
        mse = torch.nn.MSELoss()(predictions, targets)
        return mse

    def calculate_cross_entropy(self, predictions, targets):
        # predictions and targets -> (batch_size, 1)
        softmax_target = F.softmax(targets.float(), dim=0)
        log_softmax_prediction = F.log_softmax(predictions, dim=0)
        listwise_cross_entropy = torch.mean(-softmax_target * log_softmax_prediction)
        return listwise_cross_entropy

    def calculate_calibrated_cross_entropy(self, predictions, targets):
        calibrated_cross_entropy = self.cce(predictions, targets)
        return calibrated_cross_entropy

    def compute_loss(self, predictions, logits, targets):
        if self.crit == "mse":
            return self.calculate_mse(predictions, targets.float())
        elif self.crit == "ce":
            return self.calculate_cross_entropy(predictions, targets)
        elif self.crit == "mse+ce":
            return self.calculate_cross_entropy(
                predictions, targets
            ) + self.calculate_mse(predictions, targets.float())
        elif self.crit == "cce":
            return self.calculate_calibrated_cross_entropy(predictions, targets)
        else:
            assert False, "Invalid loss criterion"

    def calculate_ece(self, predictions, targets):
        # min_pred, max_pred = torch.min(predictions).item(), torch.max(predictions).item()
        # ece_calculator = ECE_Calculator(10, 0, self.max_score)
        ece_calculator = ECE_Calculator(
            predictions=predictions.squeeze(),
            labels=targets.squeeze(),
            n_bins=20,
            mode="equal_spaced",
        )
        ece = ece_calculator.calculate()
        return ece

    def calculate_adaptive_ece(self, predictions, targets):
        predictions = predictions.squeeze().detach().cpu().numpy().tolist()
        targets = targets.squeeze().detach().cpu().numpy().tolist()
        ab_inputs = list(zip(predictions, targets))
        AECE, _, _, _, _, _, _ = AdaptiveBinningForRegression(
            ab_inputs, False
        )
        return AECE

    def calculate_ndcg(self, predictions, targets, qids):
        qid_map = {}
        predictions = predictions.squeeze().detach().cpu().numpy().tolist()
        targets = targets.squeeze().detach().cpu().numpy().tolist()
        for pred, target, qid in zip(predictions, targets, qids):
            if qid not in qid_map:
                qid_map[qid] = {"y_true": [], "y_pred": []}
            qid_map[qid]["y_true"].append(target)
            qid_map[qid]["y_pred"].append(pred)
        ndcg_full_scores = {
            k: ndcg_score(np.asarray([v["y_true"]]), np.asarray([v["y_pred"]]))
            for k, v in qid_map.items()
        }
        ndcg_at_10_scores = {
            k: ndcg_score(np.asarray([v["y_true"]]), np.asarray([v["y_pred"]]), k=10)
            for k, v in qid_map.items()
        }
        return ndcg_full_scores, ndcg_at_10_scores

    def calculate_label_distribution(self, predictions):
        predictions = torch.clamp(torch.round(predictions), 0, self.max_score)
        # predictions = torch.argmax(logits, dim=-1, keepdim=True)
        label_percentage = {}
        for i in range(self.max_score + 1):
            label_percentage[i] = (predictions == i).sum().item() / len(predictions)
        return label_percentage


class CombinedNet(BasicNet):
    """
    For calibration based on neural model's predicted probability, logits (optional), and hidden states (optional)
    """

    def __init__(
        self,
        input_dimension_size,
        pretrained_model,
        hidden_size=64,
        output_size=1,
        dropout_rate=0.8,
        use_cls=True,
        use_logit=True,
        lr=1e-3,
        max_score=3,
        crit="mse",
        logger=None,
    ):
        super(BasicNet, self).__init__()

        # define model architecture
        self.input_size = input_dimension_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_cls = use_cls
        self.use_logit = use_logit
        self.lr = lr
        self.max_score = max_score
        self.crit = crit
        self.model_name_prefix = pretrained_model

        self.prob_net_input_size = 3 if self.use_logit else 1

        if not self.use_cls:
            self.prob_net = torch.nn.Sequential(
                torch.nn.Linear(self.prob_net_input_size, self.output_size),
                torch.nn.Dropout(dropout_rate),
            )
        else:
            self.prob_net = torch.nn.Sequential(
                torch.nn.Linear(self.prob_net_input_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            )

        if self.use_cls:
            self.cls_net = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
            )
            self.combine_net = torch.nn.Linear(2 * self.hidden_size, self.output_size)

        # configure calibrated softmax loss if used
        if self.crit == "cce":
            self.cce = CalibratedListwiseSoftmaxLoss()

        self.reset_buffers(["training", "validation", "testing"])
        self.logger_obj = logger

    def forward(self, batch):
        hidden_states, logits, scores = (batch[f"{self.model_name_prefix}_hidden_state"], batch[f"{self.model_name_prefix}_logits"], batch[f"{self.model_name_prefix}_score"])
        if self.use_logit:
            prob_input = torch.cat((scores, logits), -1)
        else:
            prob_input = scores
        prob_output = self.prob_net(prob_input)

        if self.use_cls:
            cls_output = self.cls_net(hidden_states)
            combined_input = torch.cat((cls_output, prob_output), dim=1)
            logits = self.combine_net(combined_input)
        else:
            logits = prob_output

        if self.output_size == 1:
            predictions = 0.5 * torch.exp(logits)
        else:
            assert False
            # labels = torch.arange(
            #     0, self.output_size, device=logits.device, dtype=torch.float32
            # ).unsqueeze(-1)
            # probabilities = torch.softmax(logits, dim=-1)
            # predictions = torch.matmul(probabilities, labels)

        # # check if the ranking produced by prob_input is the same as the ranking produced by predictions
        # if not torch.equal(torch.argsort(prob_input, dim=0), torch.argsort(predictions, dim=0)):
        #     print("prob_input and predictions do not match")
        #     print(prob_input)
        #     print(predictions)
        #     assert False
            
        # check the parameter value of self.prob_net
        # print(self.prob_net[0].weight)

        return predictions, logits


class MC_Sampling_Net(BasicNet):
    """
    comment
    """

    def __init__(
        self,
        output_size=1,
        hidden_size=64,
        dropout_rate=0.8,
        lr=1e-3,
        max_score=3,
        logger=None,
        crit="mse",
        use_gru=False,
    ):
        super(BasicNet, self).__init__()

        # define model architecture
        self.use_gru = use_gru
        self.input_size = 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.max_score = max_score
        self.crit = crit

        if self.use_gru:
            self.gru = torch.nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)

    
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size if use_gru else self.input_size, self.output_size),
            torch.nn.Dropout(dropout_rate),
        )

        # configure calibrated softmax loss if used
        if self.crit == "cce":
            self.cce = CalibratedListwiseSoftmaxLoss()

        self.reset_buffers(["training", "validation", "testing"])
        self.logger_obj = logger

    def forward(self, batch):
        batch_input = batch['sampled_p'].to(torch.float32)
        m, n = batch_input.shape
        
        if not self.use_gru:
            mask = torch.arange(n).expand(m, n).to(batch['valid_predictions'].device) < batch['valid_predictions'].unsqueeze(1)
            mask = mask.to(batch_input.device)
            batch_input_masked = batch_input * mask
            valid_counts = mask.sum(dim=1, keepdim=True)
            mean_values = batch_input_masked.sum(dim=1) / valid_counts.squeeze()
            mean_values = mean_values.unsqueeze(-1)
            logits = self.ff(mean_values)
        else:
            # GRU processing
            gru_input = batch_input.unsqueeze(-1)
            output, _ = self.gru(gru_input)
            logits = self.ff(output[:, -1, :]) 
        
        if self.output_size == 1:
            predictions = torch.exp(logits) / 2
        else:
            assert False
        return predictions, logits


class Explanation_Text_Net(BasicNet):

    def __init__(
        self,
        model_ckpt,
        explanation_type,
        output_size=1,
        dropout_rate=0.2,
        lr=1e-5,
        max_score=3,
        logger=None,
        crit="mse",
        use_bert_score=False
    ):
        super(BasicNet, self).__init__()

        # define model architecture
        self.explanation_type = explanation_type
        self.lr = lr
        self.max_score = max_score
        self.dropout_rate = dropout_rate
        self.output_size = output_size
        self.text_calibrator = AutoModelForSequenceClassification.from_pretrained(
            model_ckpt, num_labels=output_size, ignore_mismatched_sizes=True
        )
        print("Before xavier initialization", torch.max(torch.abs(self.text_calibrator.classifier.weight)))
        # torch.nn.init.xavier_uniform_(self.text_calibrator.classifier.weight)
        torch.nn.init.uniform_(self.text_calibrator.classifier.weight, -0.1, 0.1)
        print("After xavier initialization", torch.max(torch.abs(self.text_calibrator.classifier.weight)))
        
        self.crit = crit
        self.use_bert_score = use_bert_score

        # configure calibrated softmax loss if used
        if self.crit == "cce":
            self.cce = CalibratedListwiseSoftmaxLoss()

        # if self.use_bert_score:
        #     self.combined_layer = torch.nn.Sequential(
        #         torch.nn.Linear(2, 1),
        #         # torch.nn.ReLU(),  # or another activation function like nn.Tanh()
        #         # torch.nn.Linear(64, self.output_size)
        #     )

        self.reset_buffers(["training", "validation", "testing"])
        self.logger_obj = logger

    def forward(self, batch):
        input_ids = batch[f"{self.explanation_type}_input_ids"]
        attention_mask = batch[f"{self.explanation_type}_attention_mask"]
        output = self.text_calibrator(
            input_ids=input_ids, attention_mask=attention_mask
        )  # logits (bs, num_labels); loss (none because labels are not provided)
        logits = output.logits

        if self.use_bert_score:
            bert_scores = batch['bert_score']
            # concatenated = torch.cat([logits, bert_scores], dim=1)
            # logits = self.combined_layer(concatenated)
            # logits = 0.2 * logits + 0.8 * bert_scores
            logits = bert_scores + logits
            # print(torch.sum(bert_scores) / torch.sum(logits))

        logits = torch.clamp(logits, max=10.0)
        print(torch.mean(self.text_calibrator.classifier.weight))
        predictions = 0.5 * torch.exp(logits)
        print(torch.max(torch.abs(self.text_calibrator.classifier.weight)), torch.max(logits), torch.max(predictions))

        # print the weight of the classifier layer of bert
        # print(self.text_calibrator.classifier.weight)  
        # print(torch.mean(logits), torch.mean(predictions))

        # print(logits, predictions)
        # print(input_ids.shape, attention_mask.shape, logits.shape, predictions.shape)
        # assert False
        

        # probabilities = torch.softmax(logits, dim=-1)
        # labels = torch.arange(
        #     0, self.output_size, device=logits.device, dtype=torch.float32
        # ).unsqueeze(-1)
        # predictions = torch.matmul(probabilities, labels)
        return predictions, logits

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )


class Conditional_Explanation_Text_Net(BasicNet):
    """
    For calibration based on LLM related scalar features, such as sampled_p, sent_sim, exp_sim, sent_iso, exp_iso, etc.
    """

    def __init__(
        self,
        model_ckpt,
        explanation_type,
        output_size=1,
        lr=1e-3,
        max_score=3,
        dropout_rate=0.2,
        tie_weights=False,
        logger=None,
        crit="mse",
    ):
        super(BasicNet, self).__init__()

        # define model architecture
        self.explanation_type = explanation_type  # first or diverse aggregation
        self.lr = lr
        self.max_score = max_score
        self.output_size = output_size
        self.crit = crit
        self.tie_weights = tie_weights

        # configure calibrated softmax loss if used
        if self.crit == "cce":
            self.cce = CalibratedListwiseSoftmaxLoss()

        if not self.tie_weights:  # separate weights for pos and neg
            self.text_calibrator_pos = AutoModel.from_pretrained(
                model_ckpt, num_labels=output_size
            )
            self.text_calibrator_neg = AutoModel.from_pretrained(
                model_ckpt, num_labels=output_size
            )
            calibrator_hidden_size = self.text_calibrator_pos.config.hidden_size
            self.pooler_pos = torch.nn.Sequential(
                torch.nn.Linear(
                    calibrator_hidden_size,
                    calibrator_hidden_size,
                ),
                torch.nn.ReLU(),
            )
            self.pooler_neg = torch.nn.Sequential(
                torch.nn.Linear(
                    calibrator_hidden_size,
                    calibrator_hidden_size,
                ),
                torch.nn.ReLU(),
            )
        else:
            self.text_calibrator = AutoModel.from_pretrained(
                model_ckpt, num_labels=output_size
            )
            calibrator_hidden_size = self.text_calibrator.config.hidden_size
            self.pooler = torch.nn.Sequential(
                torch.nn.Linear(
                    calibrator_hidden_size,
                    calibrator_hidden_size,
                ),
                torch.nn.ReLU(),
            )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.combine_linear = torch.nn.Linear(
            calibrator_hidden_size * 2, self.output_size
        )

        self.reset_buffers(["training", "validation", "testing"])
        self.logger_obj = logger

    def forward(self, batch):
        pos_input_ids = batch[f"conditional_exp_pos_{self.explanation_type}_input_ids"]
        pos_attention_mask = batch[
            f"conditional_exp_pos_{self.explanation_type}_attention_mask"
        ]
        neg_input_ids = batch[f"conditional_exp_neg_{self.explanation_type}_input_ids"]
        neg_attention_mask = batch[
            f"conditional_exp_neg_{self.explanation_type}_attention_mask"
        ]

        if self.tie_weights:
            input_ids = torch.cat((pos_input_ids, neg_input_ids), dim=0)
            attention_mask = torch.cat((pos_attention_mask, neg_attention_mask), dim=0)
            cls_hidden_states = (
                self.text_calibrator(input_ids=input_ids, attention_mask=attention_mask)
                .last_hidden_state[:, 0, :]
                .squeeze()
            )  # size: (2 * bs, hidden_size)
            pooler_output = self.dropout(self.pooler(cls_hidden_states))
            concat_output = torch.cat(
                (
                    pooler_output[: pos_input_ids.size(0)],
                    pooler_output[pos_input_ids.size(0) :],
                ),
                dim=1,
            )

        else:
            pos_cls_hidden_states = (
                self.text_calibrator_pos(
                    input_ids=pos_input_ids, attention_mask=pos_attention_mask
                )
                .last_hidden_state[:, 0, :]
                .squeeze()
            )
            neg_cls_hidden_states = (
                self.text_calibrator_neg(
                    input_ids=neg_input_ids, attention_mask=neg_attention_mask
                )
                .last_hidden_state[:, 0, :]
                .squeeze()
            )
            pos_pooler_output = self.dropout(self.pooler_pos(pos_cls_hidden_states))
            neg_pooler_output = self.dropout(self.pooler_neg(neg_cls_hidden_states))
            concat_output = torch.cat((pos_pooler_output, neg_pooler_output), dim=1)

        logits = self.combine_linear(concat_output)
        predictions = 0.5 * torch.exp(logits)
        # probabilities = torch.softmax(logits, dim=-1)
        # labels = torch.arange(
        #     0, self.output_size, device=logits.device, dtype=torch.float32
        # ).unsqueeze(-1)
        # predictions = torch.matmul(probabilities, labels)
        return predictions, logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
