import os
import json
import torch
import random
import hashlib
import logging
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

logging.basicConfig(level=logging.INFO)

class Pretrained_Ranker_Dataset(Dataset):
    def __init__(self, data_list, query_id_splits, split):
        self.split = split
        self.data = [x for x in data_list if x["qid"] in query_id_splits[split]]
        self.qids = [item["qid"] for item in self.data]
        self.docids = [item["docid"] for item in self.data]
        self.query_texts = [item["query_text"] for item in self.data]
        self.document_texts = [item["document_text"] for item in self.data]
        self.logits = [torch.tensor(item["logits"]) for item in self.data]
        self.hidden_states = [torch.tensor(item["hidden_state"]) for item in self.data]
        self.scores = [torch.tensor(item["score"]).view(-1) for item in self.data]
        self.labels = [torch.tensor(item["label"]).view(-1) for item in self.data]

    def __len__(self):
        return len(self.data)
    
    def calculate_class_weights(self, just_log=False):
        logging.info(f"\tLabel distribution of {self.split} set:")
        flat_labels = torch.cat(self.labels).squeeze().cpu().numpy().tolist()
        for i in range(max(flat_labels) + 1):
            logging.info(f"\tLabel {i}: {flat_labels.count(i) / len(flat_labels) :.1%}")
        if not just_log:
            class_weights = 1.0 / np.array([flat_labels.count(cls) for cls in range(max(flat_labels) + 1)])
            weights = [class_weights[label] for label in flat_labels]
            return torch.DoubleTensor(weights)

    def __getitem__(self, idx):
        return {
            "qid": self.qids[idx],
            "docid": self.docids[idx],
            "query_text": self.query_texts[idx],
            "document_text": self.document_texts[idx],
            "logits": self.logits[idx],
            "hidden_state": self.hidden_states[idx],
            "score": self.scores[idx],
            "label": self.labels[idx],
        }


class Explanation_Dataset_With_Mixed_Features(Dataset):
    '''
    new features: (* means dependent to sentence transformer model)
    - sampled_p
    - sent_sim (*)
    - exp_sim (*)
    - sent_iso (*)
    - exp_iso (*)
    - most_likely_exp (MLE)
    - most_likely_exp_oracle (MLEO)
    things that might get added:
    - conditional_exp_pos_first (CEPF)
    - conditional_exp_pos_diverse (CEPD)
    - conditional_exp_neg_first (CENF)
    - conditional_exp_neg_diverse (CEND)
    '''
    def __init__(
        self,
        data_list,
        query_id_splits,
        split,
        max_seq_length=512,
        tokenzier_name="distilbert-base-uncased",
        keys_for_tokenization=[
            "qd_pair",
            "most_likely_exp",
            "most_likely_exp_oracle",
            "most_likely_exp_diverse_max_30",
        ],
        negative_document_type=None,
    ):
        self.split = split

        if negative_document_type is None:
            self.data = [x for x in data_list if x["qid"] in query_id_splits[split]]
        else:
            self.data = []
            for item in data_list:
                if item["qid"] in query_id_splits[split]:
                    if item['label'] > 0:
                        self.data.append(item)
                    elif item['neg_type'] == negative_document_type:
                        self.data.append(item)
                    else:
                        continue
        
        print(len(self.data))
        
        self.qids = [item["qid"] for item in self.data]
        self.docids = [item["docid"] for item in self.data]
        self.labels = [torch.tensor(item["label"]).view(-1) for item in self.data]
        self.sampled_ps = [torch.tensor(item["sampled_p"]).view(-1) for item in self.data]
        # self.sent_sims = [torch.tensor(item["sent_sim"]).view(-1) for item in self.data]
        # self.exp_sims = [torch.tensor(item["exp_sim"]).view(-1) for item in self.data]
        # self.sent_isos = [torch.tensor(item["sent_iso"]).view(-1) for item in self.data]
        # self.exp_isos = [torch.tensor(item["exp_iso"]).view(-1) for item in self.data]
        self.keys_for_tokenization = keys_for_tokenization

        self.max_seq_length = max_seq_length
        self.cache_dir = self._generate_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenzier_name)

        # Try loading tokenized data from cache
        cache_file_path = os.path.join(self._generate_cache_dir(), "tokenized_data.pt")
        try:
            logging.info("Trying loading tokenized data from cache...")
            self.tokenized_data = torch.load(cache_file_path)
            logging.info("Successfully loaded tokenized data from cache!")
        except FileNotFoundError:
            logging.info("Failed to load tokenized data from cache. Tokenizing data...")
            self.tokenized_data = self._tokenize_data()
            os.makedirs(self._generate_cache_dir(), exist_ok=True)
            torch.save(self.tokenized_data, cache_file_path)
            logging.info(f"Successfully tokenized data and saved to cache at {cache_file_path}!")

    def _generate_cache_dir(self):
        hash_input = str(hashlib.md5(str(self.data).encode('utf-8')).hexdigest())
        return os.path.join("cached_datasets", hash_input)
    
    def calculate_class_weights(self, just_log=False):
        logging.info(f"\tLabel distribution of {self.split} set:")
        flat_labels = torch.cat(self.labels).squeeze().cpu().numpy().tolist()
        for i in range(max(flat_labels) + 1):
            logging.info(f"\tLabel {i}: {flat_labels.count(i) / len(flat_labels) :.1%}")
        if not just_log:
            class_weights = 1.0 / np.array([flat_labels.count(cls) for cls in range(max(flat_labels) + 1)])
            weights = [class_weights[label] for label in flat_labels]
            return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.data)

    def truncate_to_top_k_tokens(self, text, k):
        tokens = self.tokenizer.tokenize(text)
        truncated_tokens = tokens[:k]
        truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    
    def _tokenize_data(self):
        tokenized_data = []
        for item in tqdm(self.data):
            tokenized_item = {}
            for key in self.keys_for_tokenization:
                if key == "qd_pair":
                    doc_text = self.truncate_to_top_k_tokens(item['document_text'], 256)
                    text = f"{item['query_text']} [SEP] {doc_text}"
                else:
                    text = item[key]
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized_item[f"{key}_input_ids"] = encoded["input_ids"][0]
                tokenized_item[f"{key}_attention_mask"] = encoded["attention_mask"][0]
            tokenized_data.append(tokenized_item)
        return tokenized_data

    def __getitem__(self, idx):
        features = {
            "qid": self.qids[idx],
            "docid": self.docids[idx],
            "label": self.labels[idx],
            "sampled_p": self.sampled_ps[idx],
            # "sent_sim": self.sent_sims[idx],
            # "exp_sim": self.exp_sims[idx],
            # "sent_iso": self.sent_isos[idx],
            # "exp_iso": self.exp_isos[idx],
        }
        for key in self.keys_for_tokenization:
            # fixed_features[key] = self.tokenized_data[idx][key]
            features[f"{key}_input_ids"] = self.tokenized_data[idx][f"{key}_input_ids"]
            features[f"{key}_attention_mask"] = self.tokenized_data[idx][f"{key}_attention_mask"]
        return features


class Conditional_Explanation_Dataset(Dataset):
    '''
    - conditional_exp_pos_first (CEPF)
    - conditional_exp_pos_diverse (CEPD)
    - conditional_exp_neg_first (CENF)
    - conditional_exp_neg_diverse (CEND)
    '''
    def __init__(
        self,
        data_list,
        query_id_splits,
        split,
        max_seq_length=512,
        tokenzier_name="distilbert-base-uncased",
        keys_for_tokenization=[
            "conditional_exp_pos_first",
            "conditional_exp_pos_diverse",
            "conditional_exp_neg_first",
            "conditional_exp_neg_diverse",
        ],
        negative_document_type=None,
    ):
        self.split = split
        
        if negative_document_type is None:
            self.data = [x for x in data_list if x["qid"] in query_id_splits[split]]
        else:
            self.data = []
            for item in data_list:
                if item["qid"] in query_id_splits[split]:
                    if item['label'] > 0:
                        self.data.append(item)
                    elif item['neg_type'] == negative_document_type:
                        self.data.append(item)
                    else:
                        continue

        self.qids = [item["qid"] for item in self.data]
        self.docids = [item["docid"] for item in self.data]
        self.labels = [torch.tensor(item["label"]).view(-1) for item in self.data]
        self.keys_for_tokenization = keys_for_tokenization

        self.max_seq_length = max_seq_length
        self.cache_dir = self._generate_cache_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenzier_name)

        # Try loading tokenized data from cache
        cache_file_path = os.path.join(self._generate_cache_dir(), "tokenized_data.pt")
        try:
            logging.info("Trying loading tokenized data from cache...")
            self.tokenized_data = torch.load(cache_file_path)
            logging.info("Successfully loaded tokenized data from cache!")
        except FileNotFoundError:
            logging.info("Failed to load tokenized data from cache. Tokenizing data...")
            self.tokenized_data = self._tokenize_data()
            os.makedirs(self._generate_cache_dir(), exist_ok=True)
            torch.save(self.tokenized_data, cache_file_path)
            logging.info(f"Successfully tokenized data and saved to cache at {cache_file_path}!")

    def _generate_cache_dir(self):
        hash_input = str(hashlib.md5(str(self.data).encode('utf-8')).hexdigest())
        return os.path.join("cached_datasets", hash_input)
    
    def calculate_class_weights(self, just_log=False):
        logging.info(f"\tLabel distribution of {self.split} set:")
        flat_labels = torch.cat(self.labels).squeeze().cpu().numpy().tolist()
        for i in range(max(flat_labels) + 1):
            logging.info(f"\tLabel {i}: {flat_labels.count(i) / len(flat_labels) :.1%}")
        if not just_log:
            class_weights = 1.0 / np.array([flat_labels.count(cls) for cls in range(max(flat_labels) + 1)])
            weights = [class_weights[label] for label in flat_labels]
            return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.data)

    # def truncate_to_top_k_tokens(self, text, k):
    #     tokens = self.tokenizer.tokenize(text)
    #     truncated_tokens = tokens[:k]
    #     truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
    #     return truncated_text
    
    def _tokenize_data(self):
        tokenized_data = []
        for item in tqdm(self.data):
            tokenized_item = {}
            for key in self.keys_for_tokenization:
                # if key == "qd_pair":
                #     text = f"{item['query_text']} [SEP] {item['document_text']}"
                # else:
                text = item[key]
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized_item[f"{key}_input_ids"] = encoded["input_ids"][0]
                tokenized_item[f"{key}_attention_mask"] = encoded["attention_mask"][0]
            tokenized_data.append(tokenized_item)
        return tokenized_data

    def __getitem__(self, idx):
        fixed_features = {
            "qid": self.qids[idx],
            "docid": self.docids[idx],
            "label": self.labels[idx],
        }
        for key in self.keys_for_tokenization:
            # fixed_features[key] = self.tokenized_data[idx][key]
            fixed_features[f"{key}_input_ids"] = self.tokenized_data[idx][f"{key}_input_ids"]
            fixed_features[f"{key}_attention_mask"] = self.tokenized_data[idx][f"{key}_attention_mask"]
        return fixed_features


class All_Purpose_Dataset(Dataset):
    '''
    'qid'
    'docid'
    'label'
    'query_text'
    'document_text'
    't5_score'
    't5_logits'
    't5_hidden_state'
    'bert_score'
    'bert_logits'
    'bert_hidden_state'
    'pred_exp_sampled_p'
    'pred_exp_most_likely_exp'
    'pred_exp_most_likely_exp_oracle'
    'pred_exp_most_likely_exp_diverse_max_30_0.35'
    'conditional_exp_pos_first'
    'conditional_exp_pos_diverse'
    'conditional_exp_neg_first'
    'conditional_exp_neg_diverse'
    '''
    def __init__(
        self,
        data_list,
        query_id_splits,
        split,
        max_seq_length=512,
        tokenzier_name="distilbert-base-uncased",
        keys_for_tokenization=[
            'qd_pair',
            'pred_exp_most_likely_exp',
            'pred_exp_most_likely_exp_oracle',
            'pred_exp_most_likely_exp_diverse_max_30_0.35',
            'conditional_exp_pos_first',
            'conditional_exp_pos_diverse',
            'conditional_exp_neg_first',
            'conditional_exp_neg_diverse'
        ],
    ):
        self.split = split
        self.data = [item for item in data_list if item["qid"] in query_id_splits[split]]
        self.keys_for_tokenization = keys_for_tokenization

        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenzier_name)

        self.cache_dir = self._generate_cache_dir()
        # Try loading tokenized data from cache
        cache_file_path = os.path.join(self._generate_cache_dir(), "tokenized_data.pt")
        try:
            logging.info("Trying loading tokenized data from cache...")
            self.tokenized_data = torch.load(cache_file_path)
            logging.info("Successfully loaded tokenized data from cache!")
        except FileNotFoundError:
            logging.info("Failed to load tokenized data from cache. Tokenizing data...")
            self.tokenized_data = self._tokenize_data()
            os.makedirs(self._generate_cache_dir(), exist_ok=True)
            torch.save(self.tokenized_data, cache_file_path)
            logging.info(f"Successfully tokenized data and saved to cache at {cache_file_path}!")

    def _generate_cache_dir(self):
        print()
        data_summary = str(self.data[:10]) + str(self.data[-10:]) + str(len(self.data))
        tokenizer_config = self.tokenizer.name_or_path
        hash_input = data_summary + tokenizer_config
        hash_output = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return os.path.join("cached_datasets", hash_output)
    
    def calculate_class_weights(self, just_log=False):
        logging.info(f"\tLabel distribution of {self.split} set:")
        labels = [torch.tensor(item["label"]).view(-1) for item in self.data]
        flat_labels = torch.cat(labels).squeeze().cpu().numpy().tolist()
        for i in range(max(flat_labels) + 1):
            logging.info(f"\tLabel {i}: {flat_labels.count(i) / len(flat_labels) :.1%}")
        if not just_log:
            class_weights = 1.0 / np.array([flat_labels.count(cls) for cls in range(max(flat_labels) + 1)])
            weights = [class_weights[label] for label in flat_labels]
            return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.data)

    def truncate_to_top_k_tokens(self, text, k):
        tokens = self.tokenizer.tokenize(text)
        truncated_tokens = tokens[:k]
        truncated_text = self.tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    
    def _tokenize_data(self):
        tokenized_data = []
        for item in tqdm(self.data):
            tokenized_item = {}
            for key in self.keys_for_tokenization:
                if key == "qd_pair":
                    doc_text = self.truncate_to_top_k_tokens(item['document_text'], 400)
                    text = f"{item['query_text']} [SEP] {doc_text}"
                else:
                    text = item[key]
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                tokenized_item[f"{key}_input_ids"] = encoded["input_ids"][0]
                tokenized_item[f"{key}_attention_mask"] = encoded["attention_mask"][0]
            tokenized_data.append(tokenized_item)
        return tokenized_data

    def __getitem__(self, idx):
        features = {
            "qid": self.data[idx]['qid'],
            "docid": self.data[idx]['docid'],
            "label": torch.tensor(self.data[idx]['label']).view(-1),
            "sampled_p": torch.tensor(self.data[idx]['pred_exp_sampled_p']),
            "valid_predictions": torch.tensor(len(self.data[idx]['pred_exp_sampled_p'])),
            't5_score': torch.tensor(self.data[idx]['t5_score']).view(-1),
            't5_logits': torch.tensor(self.data[idx]['t5_logits']),
            't5_hidden_state': torch.tensor(self.data[idx]['t5_hidden_state']),
            'bert_score': torch.tensor(self.data[idx]['bert_score']).view(-1),
            'bert_logits': torch.tensor(self.data[idx]['bert_logits']),
            'bert_hidden_state': torch.tensor(self.data[idx]['bert_hidden_state']),
        }
        for key in self.keys_for_tokenization:
            features[f"{key}_input_ids"] = self.tokenized_data[idx][f"{key}_input_ids"]
            features[f"{key}_attention_mask"] = self.tokenized_data[idx][f"{key}_attention_mask"]
        return features

    # special collate function for handling sampled_p as a sequence
    def collate_fn(self, batch):
        sampled_p_sequences = [item['sampled_p'] for item in batch]
        padded_sampled_p = pad_sequence(sampled_p_sequences, batch_first=True)

        # Create a new batch with padded sequences
        new_batch = {}
        for key in batch[0].keys():
            if key == 'sampled_p':
                new_batch[key] = padded_sampled_p
            else:
                if torch.is_tensor(batch[0][key]):
                    # Stack tensor values
                    new_batch[key] = torch.stack([item[key] for item in batch])
                else:
                    # Collect non-tensor values in a list
                    new_batch[key] = [item[key] for item in batch]

        return new_batch


class SameQIDBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.qid_indices = self._create_qid_indices()

    def _create_qid_indices(self):
        qid_indices = {}
        for idx, sample in enumerate(self.data_source):
            qid = sample["qid"]
            if qid not in qid_indices:
                qid_indices[qid] = []
            qid_indices[qid].append(idx)
        return qid_indices

    def __iter__(self):
        # Shuffle indices for each qid
        if self.shuffle:
            for indices in self.qid_indices.values():
                random.shuffle(indices)

        batches = []
        for indices in self.qid_indices.values():
            if self.batch_size < len(indices):
                for i in range(len(indices) // self.batch_size):
                    batches.append(indices[i * self.batch_size : (i + 1) * self.batch_size])
            else:
                batches.append(indices)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self):
        length = 0
        for indices in self.qid_indices.values():
            if self.batch_size < len(indices):
                length += len(indices) // self.batch_size
            else:
                length += 1
        return length