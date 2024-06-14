import os
import sys
import json
import spacy
import torch
import pickle
import logging
from tqdm.auto import tqdm
# from IsoScore import IsoScore
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
# from sentence_transformers import SentenceTransformer, util

from llm_output_processors import extract_label_from_llama_output
from src.utils import avg_cosine_sim, get_diverse_explanation_sents

logging.basicConfig(level=logging.INFO)

# sent_transformer_ckpt = sys.argv[1]  #"all-MiniLM-L6-v2"
dataset_id = sys.argv[1]  # "trec" or "istella22"

# encoded_explanation_path = (
#     f"processed_data/encoded_explanations_{sent_transformer_ckpt}_{dataset_id}.pkl"
# )

# source data
with open(f"../pygaggle/calibration-exp/explanation-data/raw_inputs_{dataset_id}.jsonl", "r",) as jsonl_file:
    all_data = [json.loads(line) for line in jsonl_file]

# llama2 generated explanations
explanations_path = f"../pygaggle/calibration-exp/local-data/pred_explanation-{dataset_id}/llama2-13B-outputs"

nlp = spacy.load("en_core_web_sm")  # for sentence tokenization
rougel_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # for acquiring diverse explanations

# if not os.path.exists(encoded_explanation_path):
    # logging.log(logging.INFO, "No encoded explanation data found, start encoding...")
    # sentence encoder
    # sent_transformer = SentenceTransformer(sent_transformer_ckpt)

record_all = {}

for data in tqdm(all_data):
    qid, docid = data["qid"], data["docid"]
    label = data["label"]
    key = f"{qid}+{docid}"

    with open(f"{explanations_path}/{key}.pkl", "rb") as fin:
        response = pickle.load(fin)
        generations = [" ".join(x.text.lower().split()) for x in response.outputs]

        # calculate sampled probability
        parsed_predictions = [
            extract_label_from_llama_output(g) for g in generations
        ]
        valid_predictions = [x for x in parsed_predictions if isinstance(x, bool)]

        if len(valid_predictions) == 0:
            print(key, label)
            continue

        record_all[key] = {}
        record_all[key]["label"] = label

        # pos_predictions = [x for x in valid_predictions if x is True]
        # sampled_probability = len(pos_predictions) / len(valid_predictions)

        probabilities = [1 if x is True else 0 for x in valid_predictions]  
        record_all[key]["sampled_p"] = probabilities

#             all_sentences, all_explanations = [], []

#             for g in generations:
#                 g = " ".join(g.split())
#                 sentences = [
#                     x.text for x in nlp(g).sents if len(word_tokenize(x.text)) >= 5
#                 ]
#                 all_sentences += sentences
#                 all_explanations.append(" ".join(sentences))

#             sent_embeddings = sent_transformer.encode(
#                 all_sentences, convert_to_tensor=True, batch_size=128
#             )
#             exp_embeddings = sent_transformer.encode(
#                 all_explanations, convert_to_tensor=True, batch_size=128
#             )

#             record_all[key]["sent_embs"] = sent_embeddings
#             record_all[key]["exp_embs"] = exp_embeddings

#     with open(encoded_explanation_path, "wb") as fout:
#         pickle.dump(record_all, fout)

# else:
#     logging.log(logging.INFO, "Encoded explanation data found, loading...")
#     with open(encoded_explanation_path, "rb") as fin:
#         record_all = pickle.load(fin)

# logging.log(logging.INFO, "Calculating similarity and isotonicity scores...")
# for key, dct in tqdm(record_all.items()):
#     sent_embeddings = record_all[key]["sent_embs"]
#     exp_embeddings = record_all[key]["exp_embs"]

#     record_all[key]["sent_sim"] = avg_cosine_sim(
#         util.cos_sim(sent_embeddings, sent_embeddings)
#     )
#     record_all[key]["exp_sim"] = avg_cosine_sim(
#         util.cos_sim(exp_embeddings, exp_embeddings)
#     )

#     # n * m where n is dimension size, and m is number of points
#     record_all[key]["sent_iso"] = IsoScore.IsoScore(
#         torch.t(sent_embeddings).numpy()
#     ).item()
#     record_all[key]["exp_iso"] = IsoScore.IsoScore(
#         torch.t(exp_embeddings).numpy()
#     ).item()

# logging.log(logging.INFO, "Saving results...")

with open(f"processed_data/pred_exp.{dataset_id}.jsonl", "w") as fout:
    for data in tqdm(all_data):
        qid, docid = data["qid"], data["docid"]
        key = f"{qid}+{docid}"
        if key not in record_all:
            print(key)
            continue
        data["sampled_p"] = record_all[key]["sampled_p"]
        # data["sent_sim"] = record_all[key]["sent_sim"]
        # data["exp_sim"] = record_all[key]["exp_sim"]
        # data["sent_iso"] = record_all[key]["sent_iso"]
        # data["exp_iso"] = record_all[key]["exp_iso"]

        # deal with pred_exp, including most probable explanation, most probable explanation (oracle),
        # and most probable explanation + diverse
        with open(f"{explanations_path}/{key}.pkl", "rb") as fin:
            response = pickle.load(fin)
        generations = [" ".join(x.text.lower().split()) for x in response.outputs]
        likelihoods = [x.cumulative_logprob for x in response.outputs]
        parsed_predictions = [extract_label_from_llama_output(g) for g in generations]
        for g, l, p in zip(generations, likelihoods, parsed_predictions):
            if isinstance(p, bool) and "most_likely_exp" not in data:
                data["most_likely_exp"] = g
            if isinstance(p, bool) and "most_likely_exp_oracle" not in data:
                if (data["label"] == 0 and p is False) or (
                    data["label"] >= 1 and p is True
                ):
                    data["most_likely_exp_oracle"] = g
            if "most_likely_exp" in data and "most_likely_exp_oracle" in data:
                break
        if "most_likely_exp_oracle" not in data:
            data["most_likely_exp_oracle"] = data["most_likely_exp"]

        data["most_likely_exp_diverse_max_30_0.35"] = get_diverse_explanation_sents(
            generations, nlp, 0.35, rougel_scorer, 30
        )
        # data["most_likely_exp_diverse"] = get_diverse_explanation_sents(
        #     generations, nlp, 0.35, rougel_scorer, None
        # )

        fout.write(json.dumps(data) + "\n")

logging.log(logging.INFO, "Done!")
