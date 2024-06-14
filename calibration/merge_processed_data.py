import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="trec", choices=["trec", "clueweb"])

args = parser.parse_args()

cond_exp_pos = f"processed_data/cond_exp_pos.{args.dataset}.jsonl"
cond_exp_neg = f"processed_data/cond_exp_neg.{args.dataset}.jsonl"
pred_exp = f"processed_data/pred_exp.{args.dataset}.jsonl"
# bert = f"processed_data/monobert-large-msmarco.{args.dataset}.jsonl"
bert = f"processed_data/monobert-msmarco.{args.dataset}.jsonl"
t5 = f"processed_data/monot5-base-msmarco.{args.dataset}.jsonl"

# merge the files
all_data = {}
with open(t5, "r") as file:
    for line in file:
        item = json.loads(line)
        qid, docid, label, query_text, document_text, score, logits, hidden_state = item['qid'], item['docid'], item['label'], item['query_text'], item['document_text'], item['score'], item['logits'], item['hidden_state']
        all_data[(qid, docid)] = {
            'qid': qid,
            'docid': docid,
            'label': label,
            'query_text': query_text,
            'document_text': document_text,
            't5_score': score,
            't5_logits': logits,
            't5_hidden_state': hidden_state,
        }

with open(bert, "r") as file:
    for line in file:
        item = json.loads(line)
        qid, docid, score, logits, hidden_state = item['qid'], item['docid'], item['score'], item['logits'], item['hidden_state']
        all_data[(qid, docid)]['bert_score'] = score
        all_data[(qid, docid)]['bert_logits'] = logits
        all_data[(qid, docid)]['bert_hidden_state'] = hidden_state

with open(pred_exp, "r") as file:
    for line in file:
        item = json.loads(line)
        qid, docid = item['qid'], item['docid']
        for k, v in item.items():
            if k not in all_data[(qid, docid)]:
                all_data[(qid, docid)][f"pred_exp_{k}"] = v

with open(cond_exp_pos, "r") as file:
    for line in file:
        item = json.loads(line)
        qid, docid = item['qid'], item['docid']
        for k, v in item.items():
            if k not in all_data[(qid, docid)]:
                all_data[(qid, docid)][k] = v

with open(cond_exp_neg, "r") as file:
    for line in file:
        item = json.loads(line)
        qid, docid = item['qid'], item['docid']
        for k, v in item.items():
            if k not in all_data[(qid, docid)]:
                all_data[(qid, docid)][k] = v

with open(f"processed_data/merged_{args.dataset}.jsonl", "w") as file:
    for k, v in all_data.items():
        data = v
        v['qid'] = k[0]
        v['docid'] = k[1]
        file.write(json.dumps(data) + "\n")
