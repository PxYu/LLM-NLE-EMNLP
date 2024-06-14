# CUDA_VISIBLE_DEVICES=0 python calibrate_with_cond_exp_texts.py --batch_size 16 --explanation_type diverse --max_epoch 10 --same_qid_training --criterion mse     &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_cond_exp_texts.py --batch_size 16 --explanation_type diverse --max_epoch 10 --same_qid_training --criterion ce      &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_cond_exp_texts.py --batch_size 16 --explanation_type diverse --max_epoch 10 --same_qid_training --criterion mse+ce  &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_cond_exp_texts.py --batch_size 16 --explanation_type diverse --max_epoch 10 --same_qid_training --criterion cce     

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion mse &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion mse &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion mse &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion mse

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion mse+ce &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion mse+ce &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion mse+ce &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion mse+ce

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion ce &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion ce &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion ce &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion ce

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased


# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion mse --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion mse --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion mse --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion mse --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion mse+ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion mse+ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion mse+ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion mse+ce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 10 --same_qid_training --criterion cce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 10 --same_qid_training --criterion cce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 10 --same_qid_training --criterion cce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 10 --same_qid_training --criterion cce --dataset istella22 --max_score 4 --output_size 5 --negative_document_type random

# CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type qd_pair                        --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp                --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_diverse_max_30 --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
# CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 16 --explanation_type most_likely_exp_oracle         --max_epoch 15 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt distilbert-base-uncased


CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 3e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 1e-6 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco

CUDA_VISIBLE_DEVICES=0 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type qd_pair                        --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=1 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp                --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=2 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_diverse_max_30 --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611 &
CUDA_VISIBLE_DEVICES=3 python calibrate_with_exp_texts.py --batch_size 8 --gradient_accumulation_steps 4 --explanation_type most_likely_exp_oracle         --learning_rate 1e-5 --same_qid_training --criterion cce --model_ckpt castorini/monobert-large-msmarco --seed 611