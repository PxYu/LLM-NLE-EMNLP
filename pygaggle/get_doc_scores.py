import os
import json
# import random
import pickle
import argparse
from utils import (
    read_queries,
    read_qrels,
    read_passages_memory_efficient
)

from rankLlama import RankLlama, MonoLlama_Vllm

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, MonoBERT

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", choices=["marcov1", "marcov2", "clueweb"])
parser.add_argument("--is_trec", action='store_true')
parser.add_argument("--model_name", default=None)
parser.add_argument("--model_type", choices=["t5", "bert", "llama-r", 'llama-tf-zero'])
parser.add_argument("--topk", default=100, type=int)

args = parser.parse_args()

if args.corpus == "marcov1":

    if args.is_trec:  # trec-dl-2019 and trec-dl-2020; candidates are from relevance judgements

        queries_list = [
            read_queries("msmarco-passage/trec-dl-2019/judged"),
            read_queries("msmarco-passage/trec-dl-2020/judged")
        ]

        # candidate documents are from relevance judgements
        qrels_list = [
            read_qrels("msmarco-passage/trec-dl-2019/judged"),
            read_qrels("msmarco-passage/trec-dl-2020/judged")
        ]
        candidates = {}
        for qrels in qrels_list:
            for qid, dct in qrels.items():
                if len(dct) >= 1000:
                    print(f"Query {qid} has {len(dct)} relevance labels -- please double check!")
                    continue
                candidates[qid] = list(dct.keys())

    else:  # small dev set; candidate documents are from top-1k BM25 results

        queries_list = [read_queries("msmarco-passage/dev/small")]
        qrels = read_qrels("msmarco-passage/dev/small")

        # candidate documents are from top-1k BM25 results
        with open("calibration-exp/msmarco-passage-top1k/v1-res.pkl", 'rb') as fin:
            top1k_res = pickle.load(fin)
        candidates = {}
        for qid, lst in tqdm(top1k_res.items()):
            if qid in qrels:
                candidate_docs = lst[:min(len(lst), args.topk)]
                num_rel = len([doc for doc in candidate_docs if doc in qrels[qid]])
                if num_rel > 0:
                    candidates[qid] = candidate_docs
        
        print(len(candidates))

    if not os.path.exists("calibration-exp/msmarco-passage-v1.corpus"):
        passages = read_passages_memory_efficient("msmarco-passage", candidates)
        with open ("calibration-exp/msmarco-passage-v1.corpus", 'wb') as fout:
            pickle.dump(passages, fout)
    else:
        with open("calibration-exp/msmarco-passage-v1.corpus", 'rb') as fin:
            passages = pickle.load(fin)

elif args.corpus == "marcov2":
    assert args.is_trec
    queries_list = [
        read_queries("msmarco-passage-v2/trec-dl-2021/judged"),
        read_queries("msmarco-passage-v2/trec-dl-2022/judged")
    ]
    # candidate documents are from relevance judgements
    qrels_list = [
        read_qrels("msmarco-passage-v2/trec-dl-2021/judged"),
        read_qrels("msmarco-passage-v2/trec-dl-2022/judged")
    ]
    candidates = {}
    for qrels in qrels_list:
        for qid, dct in qrels.items():
            # num_pos = len([v for v in dct.values() if v > 0])
            num_docs = len(dct)
            if num_docs >= 2000:
                print(f"Query {qid} has {num_docs} relevance labels -- please double check!")
                continue
            candidates[qid] = list(dct.keys())

    # v2 passages are too large to read; we only read part of them and save them to disk
    if not os.path.exists("calibration-exp/msmarco-passage-v2.corpus"):
        passages = read_passages_memory_efficient("msmarco-passage-v2", candidates)
        with open ("calibration-exp/msmarco-passage-v2.corpus", 'wb') as fout:
            pickle.dump(passages, fout)
    else:
        with open("calibration-exp/msmarco-passage-v2.corpus", 'rb') as fin:
            passages = pickle.load(fin)

elif args.corpus == "istella22":
    pass
elif args.corpus == "clueweb":
    # read jsonl file
    # all_data = []
    queries, passages, candidates = {}, {}, {}
    with open("calibration-exp/explanation-data/raw_inputs_clueweb.jsonl", 'r') as fin:
        for line in fin:
            # all_data.append(json.loads(line))
            data = json.loads(line)
            qid, docid, query_text, document_text = data["qid"], data["docid"], data["query_text"], data["document_text"]
            if qid not in queries:
                queries[qid] = query_text
            if docid not in passages:
                passages[docid] = document_text
            if qid not in candidates:
                candidates[qid] = []
            candidates[qid].append(docid)

    
else:
    assert False, "NOT IMPLEMENTED YET"


if args.corpus in ["marcov1", "marcov2"]:
    queries = {}
    for q in queries_list:
        queries.update(q)



if args.model_type == 't5':
    reranker = MonoT5(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
elif args.model_type == 'bert':
    reranker = MonoBERT(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
elif args.model_type == 'llama-r':
    reranker = RankLlama(peft_model_name=args.model_name)
    assert False, "not implemented yet"
    tokenizer = reranker.tokenizer
elif args.model_type == 'llama-tf-zero':
    reranker = MonoLlama_Vllm(model_name=args.model_name)
    assert False, "tokenizer not set yet"
    reranker.set_prompt("tf-zero")  # set prompt for TRUE/False zero-shot inference

model_folder_name = args.model_name.split("/")[-1]
if args.model_type == 'llama-tf-zero':
    model_folder_name += "-tf-zero"

def truncate_to_max_tokens(text, tokenizer, max_tokens=512):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(truncated_tokens)

# Start inference
inference_results = {}

for qid, docids in tqdm(candidates.items()):

    if qid not in queries:
        print(f"Query {qid} not found in queries!")
        continue

    query_text = queries[qid]
    candidate_docids = docids

    if len(candidate_docids) == 0:
        continue

    if args.model_type in ['t5', 'bert']:

        query_obj = Query(query_text)
        texts_obj = [Text(truncate_to_max_tokens(passages[docid], tokenizer, 400), {'docid': docid}, 0) for docid in candidate_docids]
        reranked = reranker.rerank(query_obj, texts_obj)
        ranked_ids_scores = [(x.metadata["docid"], x.score, x.logits, x.last_hidden_states) for x in reranked]
        # for x in ranked_ids_scores:
        #     print(x[3].shape)
        # assert False
        inference_results[qid] = ranked_ids_scores

    elif args.model_type == 'llama-r':
        ranked_ids_scores = []
        for docid in candidate_docids:
            score = reranker.get_qd_score(query_text, passages[docid])
            ranked_ids_scores.append((docid, score, None))
        inference_results[qid] = ranked_ids_scores

    elif args.model_type == 'llama-tf-zero':
        ranked_ids_scores = []
        passage_texts = []
        for docid in candidate_docids:
            passage_text = passages[docid]
            terms = passage_text.split()
            if len(terms) >= 512:
                passage_text = " ".join(terms[:512])
            passage_texts.append(passage_text)
        scores, logits = reranker.get_qd_score(query_text, passage_texts)
        for docid, score, logits in zip(candidate_docids, scores, logits):
            ranked_ids_scores.append((docid, score, logits))
        inference_results[qid] = ranked_ids_scores


os.makedirs(f"calibration-exp/scores/{model_folder_name}", exist_ok=True)
file_name = f"{args.corpus}-results"
if args.is_trec:
    file_name += ".trec.pkl"
else:
    file_name += ".pkl"

with open(f"calibration-exp/scores/{model_folder_name}/{file_name}", 'wb') as fout:
    pickle.dump(inference_results, fout)
