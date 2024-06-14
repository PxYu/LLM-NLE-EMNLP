import json
import ir_datasets
from tqdm.auto import tqdm

# dataset = ir_datasets.load("istella22")

# with open("istella22-experiments/monoT5/pyserini-index-source/source.jsonl", 'w') as fout:
#     for doc in tqdm(dataset.docs_iter()):
#         entry = {"id": doc.doc_id, "contents": f"{doc.title} {doc.text}"}
#         json.dump(entry, fout)
#         fout.write('\n')


# python -m pyserini.index.lucene \
#       --collection JsonCollection \
#       --input istella22-experiments/monoT5/pyserini-index-source \
#       --index istella22-experiments/monoT5/pyserini-index \
#       --generator DefaultLuceneDocumentGenerator \
#       --threads 1 \
#       --storePositions --storeDocvectors --storeRaw

# In[ ]:


from utils import *
from pyserini.search import LuceneSearcher
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.base import hits_to_texts

queries_istella22 = read_queries("istella22/test")
qrels_istella22 = read_qrels("istella22/test")

searcher = LuceneSearcher('istella22-experiments/monoT5/pyserini-index')


# In[ ]:


istella_top1k = {}

for qid, q_text in tqdm(queries_istella22.items()):
    # query = Query(q_text)
    hits = searcher.search(q_text, k=1000)
    texts = hits_to_texts(hits)
    istella_top1k[qid] = [x.metadata["docid"] for x in texts]


# In[ ]:


import os, pickle

os.makedirs('calibration-exp/istella-top1k', exist_ok=True)
with open("calibration-exp/istella-top1k/test.pkl", 'wb') as fout:
    pickle.dump(istella_top1k, fout)
