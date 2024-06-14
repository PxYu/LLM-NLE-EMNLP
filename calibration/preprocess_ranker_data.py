import json
import pickle
import argparse
import numpy as np
from tqdm.auto import tqdm


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["monot5-base-msmarco", "monobert-large-msmarco", "monobert-msmarco"])
parser.add_argument("--dataset", type=str, choices=["trec", "clueweb"])
args = parser.parse_args()

trec_all_pairs = []
with open(f"../pygaggle/calibration-exp/explanation-data/raw_inputs_{args.dataset}.jsonl", 'r') as fin:
    for line in fin:
        trec_all_pairs.append(json.loads(line))

if args.dataset == "trec":
    logits_files = [
        f"../pygaggle/calibration-exp/scores/{args.model}/marcov1-results.trec.pkl",
        f"../pygaggle/calibration-exp/scores/{args.model}/marcov2-results.trec.pkl"
    ]
elif args.dataset == "clueweb":
    logits_files = [
        f"../pygaggle/calibration-exp/scores/{args.model}/clueweb-results.pkl"
    ]

logits_data = {}
for file in logits_files:
    with open(file, 'rb') as fin:
        logits_data.update(pickle.load(fin))

unique_queries_from_raw = set([item['qid'] for item in trec_all_pairs])
unique_queries_from_logits = set(logits_data.keys())
print(len(unique_queries_from_raw), len(unique_queries_from_logits))

reformatted_logits_data = {}
for k, v in logits_data.items():
    reformatted_logits_data[k] = {}
    for item in v:
        reformatted_logits_data[k][item[0]] = {
            'score': item[1],
            'logits': item[2],
            'hidden_state': item[3]
        }

for data in tqdm(trec_all_pairs):
    qid = data['qid']
    docid = data['docid']
    additional_data = reformatted_logits_data[qid][docid]
    data["score"] = additional_data["score"]
    data["logits"] = additional_data["logits"]
    data["hidden_state"] = additional_data["hidden_state"]

with open(f"processed_data/{args.model}.{args.dataset}.jsonl", 'w') as fout:
    for data in tqdm(trec_all_pairs):
        dumped = json.dumps(data, cls=NumpyEncoder)
        fout.write(dumped + "\n")
