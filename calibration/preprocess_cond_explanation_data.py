import os
import sys
import json
import spacy
import pickle
import logging
from math import ceil
from tqdm.auto import tqdm
from multiprocessing import Pool
from rouge_score import rouge_scorer
from src.utils import get_diverse_explanation_sents

logging.basicConfig(level=logging.INFO)

pos_or_neg = str(sys.argv[1])
dataset_id = str(sys.argv[2])  # "trec", "istella22", or "clueweb"

# source data
with open(f"../pygaggle/calibration-exp/explanation-data/raw_inputs_{dataset_id}.jsonl", "r") as jsonl_file:
    all_data = [json.loads(line) for line in jsonl_file]

# for sentence tokenization
nlp = spacy.load("en_core_web_sm")
# for acquiring diverse explanations
rougel_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# llama2 generated explanations path
explanations_path = f"../pygaggle/calibration-exp/local-data/conditional_explanation_{pos_or_neg}-{dataset_id}/llama2-13B-outputs"

# Worker function for multiprocessing
def process_data_chunk(data_chunk):
    processed_chunk = []
    for data in tqdm(data_chunk):
        qid, docid = data["qid"], data["docid"]
        key = f"{qid}+{docid}"

        # deal with conditional_exp ("why relevant and why nonrelevant")
        exps = [
            " ".join(x.text.lower().split())
            for x in pickle.load(open(f"{explanations_path}/{key}.pkl", "rb")).outputs
        ]
        data[f"conditional_exp_{pos_or_neg}_first"] = exps[0]
        data[f"conditional_exp_{pos_or_neg}_diverse"] = get_diverse_explanation_sents(
            exps, nlp, 0.35, rougel_scorer, 30
        )
        processed_chunk.append(data)
    return processed_chunk

# Determine the number of processes
num_processes = 8  # Adjust based on your machine's capability

# Split your all_data into chunks
chunk_size = ceil(len(all_data) / num_processes)
data_chunks = [all_data[i:i + chunk_size] for i in range(0, len(all_data), chunk_size)]
print([len(x) for x in data_chunks])

# Create a pool of workers and distribute the work
with Pool(num_processes) as pool:
    results = pool.map(process_data_chunk, data_chunks)

# Flatten the list of results
flattened_results = [item for sublist in results for item in sublist]

# Write the results to file
processed_file_path = f"processed_data/cond_exp_{pos_or_neg}.{dataset_id}.jsonl"
with open(processed_file_path, "w") as fout:
    for data in tqdm(flattened_results, desc="Writing data"):
        fout.write(json.dumps(data) + "\n")

logging.info("Processing complete!")
