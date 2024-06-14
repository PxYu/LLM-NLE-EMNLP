# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monot5-base-msmarco-10k" --model_type t5 --topk 100
# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monot5-base-msmarco-10k" --model_type t5 --is_trec
# python get_doc_scores.py --corpus marcov2 --model_name "castorini/monot5-base-msmarco-10k" --model_type t5 --is_trec

# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monot5-base-msmarco" --model_type t5 --topk 100
# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monot5-base-msmarco" --model_type t5 --is_trec
# python get_doc_scores.py --corpus marcov2 --model_name "castorini/monot5-base-msmarco" --model_type t5 --is_trec

# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monobert-large-msmarco" --model_type bert --topk 100
# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monobert-large-msmarco" --model_type bert --is_trec
# python get_doc_scores.py --corpus marcov2 --model_name "castorini/monobert-large-msmarco" --model_type bert --is_trec

# python get_doc_scores.py --corpus marcov1 --model_name "castorini/rankllama-v1-7b-lora-passage" --model_type llama --is_trec
# python get_doc_scores.py --corpus marcov2 --model_name "castorini/rankllama-v1-7b-lora-passage" --model_type llama --is_trec
# python get_doc_scores.py --corpus marcov1 --model_name "castorini/rankllama-v1-7b-lora-passage" --model_type llama --topk 100

# python get_doc_scores.py --corpus marcov1 --model_name "TheBloke/Llama-2-13B-chat-AWQ" --model_type llama-tf-zero --is_trec
# python get_doc_scores.py --corpus marcov2 --model_name "TheBloke/Llama-2-13B-chat-AWQ" --model_type llama-tf-zero --is_trec
# python get_doc_scores.py --corpus marcov1 --model_name "TheBloke/Llama-2-13B-chat-AWQ" --model_type llama-tf-zero --topk 10

# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monot5-base-msmarco" --model_type t5 --topk 10
# python get_doc_scores.py --corpus marcov1 --model_name "castorini/monobert-large-msmarco" --model_type bert --topk 10


# # Re-install PyTorch with CUDA 11.8.
# pip uninstall torch -y
# conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# # Re-install xFormers with CUDA 11.8.
# pip uninstall xformers -y
# pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118



# python get_doc_scores.py --corpus clueweb --model_name "castorini/monobert-large-msmarco" --model_type bert
# python get_doc_scores.py --corpus clueweb --model_name "castorini/monot5-base-msmarco" --model_type t5

python get_doc_scores.py --corpus marcov1 --model_name veneres/monobert-msmarco --model_type bert --is_trec
python get_doc_scores.py --corpus marcov2 --model_name veneres/monobert-msmarco --model_type bert --is_trec
# python get_doc_scores.py --corpus clueweb --model_name veneres/monobert-msmarco --model_type bert