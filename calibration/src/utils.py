import torch
import numpy as np
import ir_datasets
from tqdm.auto import tqdm
from nltk.tokenize import word_tokenize

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


class ECE_Calculator:
    def __init__(self, predictions, labels, n_bins, mode):

        assert mode in ["equal_sized", "equal_spaced"]

        self.predictions = predictions
        self.labels = labels
        self.num_bins = n_bins

        if not torch.is_tensor(self.predictions):
            self.predictions = torch.tensor(self.predictions)
        if not torch.is_tensor(self.labels):
            self.labels = torch.tensor(self.labels)
        
        if mode == "equal_spaced":
            min_pred = torch.min(predictions).item()
            max_pred = torch.max(predictions).item()
            bin_boundaries = torch.linspace(min_pred, max_pred, n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            bin_boundaries = self.divide_into_equal_sized_bins(self.predictions, self.labels)
            self.bin_lowers = torch.tensor(bin_boundaries[:-1])
            self.bin_uppers = torch.tensor(bin_boundaries[1:])

    def divide_into_equal_sized_bins(self, input_list, corresponding_list):
        combined_lists = list(zip(input_list, corresponding_list))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
        bins = [x.tolist() for x in np.array_split(sorted_combined_lists, self.num_bins)]
        bin_boundaries = [x[0][0] for x in bins] + [bins[-1][-1][0]]
        return bin_boundaries

    def calculate(self):
        self.bin_truth = []
        self.bin_confidence = []
        self.bin_prob = []
        self.bin_ece = []

        # FIXME: figure out how it works when all confidences are the same (e.g., 0 or something)

        ece = torch.tensor(0.0, device=self.predictions.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = self.predictions.ge(bin_lower.item()) * self.predictions.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                self.bin_prob.append(prop_in_bin.item())
                avg_truth_in_bin = self.labels[in_bin].float().mean()
                avg_confidence_in_bin = self.predictions[in_bin].mean()
                self.bin_truth.append(avg_truth_in_bin.item())
                self.bin_confidence.append(avg_confidence_in_bin.item())
                bin_ece = (
                    torch.abs(avg_confidence_in_bin - avg_truth_in_bin) * prop_in_bin
                )
                ece += bin_ece
                self.bin_ece.append(bin_ece)
            else:
                self.bin_truth.append(-1)
                self.bin_prob.append(0)

        return ece.item()


def class_balanced_ece(confidences, labels, n_bins=20, mode="equal_spaced"):

    if not torch.is_tensor(confidences):
        confidences = torch.tensor(confidences)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    ece_vals = {}
    unique_labels = torch.unique(labels.squeeze()).tolist()
    for label in unique_labels:
        confidences_of_label = confidences[labels == label]
        reformatted_labels = torch.full_like(confidences_of_label, label)
        ece_val = ECE_Calculator(
            confidences_of_label, reformatted_labels, n_bins, mode
        ).calculate()
        ece_vals[label] = ece_val
    return ece_vals

def avg_cosine_sim(matrix):
    num_element = matrix.size(0)
    average_similarity = (matrix.sum() - torch.trace(matrix)) / (num_element * (num_element - 1))
    return average_similarity.item()

def get_diverse_explanation_sents(
        list_of_explanations,
        sentence_splitter,
        rouge_threshold,
        rouge_scorer,
        max_num_sents
        ):
    explanation_set = []
    for exp in list_of_explanations:
        for sent in [x.text for x in sentence_splitter(exp).sents if len(word_tokenize(x.text)) >= 5]:
            if not explanation_set:
                explanation_set.append(sent)
            else:
                max_rouge_score = max([rouge_scorer.score(sent, x)['rougeL'].fmeasure for x in explanation_set])
                if max_rouge_score <= rouge_threshold:
                    # this sentence is novel
                    explanation_set.append(sent)
                    if len(explanation_set) == max_num_sents:
                        return " ".join(explanation_set)
    return " ".join(explanation_set)