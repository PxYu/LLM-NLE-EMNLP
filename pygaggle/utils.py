import ir_datasets
from tqdm.auto import tqdm

def read_qrels(dataset_name):
    qrels = {}
    dataset = ir_datasets.load(dataset_name)
    for qrel in dataset.qrels_iter():
        qid = qrel.query_id
        docid = qrel.doc_id
        relevance = qrel.relevance
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][docid] = relevance
    return qrels

def read_queries(dataset_name):
    queries = {}
    dataset = ir_datasets.load(dataset_name)
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text
    return queries

def read_passages(dataset_name):
    passages = {}
    dataset = ir_datasets.load(dataset_name)
    for doc in tqdm(dataset.docs_iter()):
        passages[doc.doc_id] = doc.text
    return passages

def read_passages_memory_efficient(dataset_name, candidates):

    candidate_passages = set()
    for _, doc_list in candidates.items():
        candidate_passages |= set(doc_list)

    passages = {}
    dataset = ir_datasets.load(dataset_name)
    for doc in tqdm(dataset.docs_iter()):
        if doc.doc_id in candidate_passages:
            passages[doc.doc_id] = doc.text
    return passages
