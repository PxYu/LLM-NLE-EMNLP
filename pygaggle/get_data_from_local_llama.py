import os
import sys
import json
import pickle
import argparse
from tqdm.auto import tqdm
from local_vllm import Llama_Vllm
from transformers import LlamaTokenizerFast

def truncate_to_top_k_tokens(text, k, tokenizer):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:k]
    truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
    return truncated_text

def send_qd_to_llama_for_relevance(llm, queries, documents, dataset="trec"):
    prompt_format = "[INST] <<SYS>> {system_prompt} <</SYS>> {user_input} [/INST]"

    if dataset == "trec":
        prompts = [
            prompt_format.format(
                system_prompt="""You are a search quality rater evaluating the relevance of passages. Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings.
3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query.""",
                user_input=f"\nQuery: {query}\nDocument: {document}\nOutput:",
            )
            for query, document in zip(queries, documents)
        ]
    elif dataset == "clueweb":
        prompts = [
            prompt_format.format(
                system_prompt="""You are a search quality rater evaluating the relevance of passages. Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings.
0: Two annotators rated as non-relevant.
1: One annotator rated as relevant, one as non-relevant.
2: Two annotators rated as relevant, OR one rates as highly relevant and one as non-relevant.
3: One annotator rated as highly relevant, one as relevant.
4: Two annotators rated as highly relevant.""",
                user_input=f"\nQuery: {query}\nDocument: {document}\nOutput:",
            )
            for query, document in zip(queries, documents)
        ]

    outputs = llm.llm.generate(prompts, sampling_params=llm.sampling_params)

    return outputs


def send_qd_to_llama_for_pred_and_exp(llm, queries, documents):
    prompt_format = "[INST] <<SYS>> {system_prompt} <</SYS>> {user_input} [/INST]"

    prompts = [
        prompt_format.format(
            system_prompt="""For the following query and document, judge whether they are relevant or nonrelevant, and provide an explanation.
Output "Relevant" or "Nonrelevant". DO NOT repeat the content of the query or the document.""",
            user_input=f"\nQuery: {query}\nDocument: {document}\nOutput:",
        )
        for query, document in zip(queries, documents)
    ]

    outputs = llm.llm.generate(prompts, sampling_params=llm.sampling_params)

    return outputs


def send_qd_to_llama_for_conditional_exp(llm, queries, documents, label):
    prompt_format = "[INST] <<SYS>> {system_prompt} <</SYS>> {user_input} [/INST]"

    if label == "relevant":
        label_string = "relevant"
    elif label == "nonrelevant":
        label_string = "not relevant"
    else:
        raise NotImplementedError

    prompts = [
        prompt_format.format(
            system_prompt=f"""For the following query and document, explain why they are {label_string}.""",
            user_input=f"\nQuery: {query}\nDocument: {document}\nOutput:",
        )
        for query, document in zip(queries, documents)
    ]

    outputs = llm.llm.generate(prompts, sampling_params=llm.sampling_params)

    return outputs


def get_nth_portion(my_list, k, n):
    list_length = len(my_list)
    start_index = (n - 1) * (list_length // k + 1)
    end_index = min(start_index + (list_length // k + 1), list_length)
    nth_portion = my_list[start_index:end_index]
    return nth_portion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--option",
        type=str,
        choices=[
            "relevance",
            "pred_explanation",
            "conditional_explanation_pos",
            "conditional_explanation_neg",
        ]
    )
    parser.add_argument(
        "--slice",
        type=str,
    )
    parser.add_argument("--dataset", choices=["trec", "istella22", "clueweb"])
    parser.add_argument("--model_size", choices=["7B", "13B", "70B"], default="13B")
    parser.add_argument("--sampling_percentage", type=int, default=None)
    args = parser.parse_args()

    option = args.option
    slice = args.slice
    dataset = args.dataset
    model_size = args.model_size
    sampling_percentage = args.sampling_percentage

    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    if dataset == "trec":
        topk_tokens = 1024
        if not sampling_percentage:
            input_raw_data_path = "calibration-exp/explanation-data/raw_inputs_trec.jsonl"
        else:
            input_raw_data_path = f"calibration-exp/explanation-data/raw_inputs_trec_{sampling_percentage}.jsonl"
    elif dataset == "istella22":
        topk_tokens = 1024
        input_raw_data_path = (
            "calibration-exp/explanation-data/raw_inputs_istella22.jsonl"
        )
    elif dataset == "clueweb":
        topk_tokens = 400
        input_raw_data_path = (
            "calibration-exp/explanation-data/raw_inputs_clueweb.jsonl"
        )
    else:
        print("Invalid dataset.")
        sys.exit()

    output_dir = f"local-data/{option}-{dataset}"
    if sampling_percentage:
        output_dir += f"-{sampling_percentage}"

    if option == "relevance":
        api_func = send_qd_to_llama_for_relevance
        number_of_samples_per_input = 20
        batch_size = 4
    elif option == "pred_explanation":
        api_func = send_qd_to_llama_for_pred_and_exp
        number_of_samples_per_input = 20
        batch_size = 4
    elif "conditional_explanation" in option:
        api_func = send_qd_to_llama_for_conditional_exp
        number_of_samples_per_input = 10
        batch_size = 4
        label = "relevant" if option == "conditional_explanation_pos" else "nonrelevant"
    else:
        raise NotImplementedError

    # 1. query phase

    print(slice)

    with open(input_raw_data_path, "r") as jsonl_file:
        all_data = [json.loads(line) for line in jsonl_file]
    if slice == "all":
        pass
    else:
        n, k = slice.split("/")
        n, k = int(n), int(k)  # k slices, take the n-th one
        all_data = get_nth_portion(all_data, k, n)

    print("Number of samples:", len(all_data))

    output_dir = f"calibration-exp/{output_dir}/llama2-{model_size}-outputs"
    os.makedirs(output_dir, exist_ok=True)

    model = Llama_Vllm(
        f"TheBloke/Llama-2-{model_size}-chat-AWQ",
        number_of_samples_per_input,
        # top_p=0.9,
        # temperature=0.6,
        max_tokens=256,
    )

    batch_qids, batch_queries, batch_docids, batch_documents = [], [], [], []
    for sample in tqdm(all_data):
        if os.path.exists(f"{output_dir}/{sample['qid']}+{sample['docid']}.pkl"):
            response = pickle.load(
                open(
                    f"{output_dir}/{sample['qid']}+{sample['docid']}.pkl",
                    "rb",
                )
            )
            if len(response.outputs) == number_of_samples_per_input:
                continue  # skip this sample because it is already complete

        batch_qids.append(sample["qid"])
        batch_queries.append(sample["query_text"])
        batch_docids.append(sample["docid"])
        batch_documents.append(
            truncate_to_top_k_tokens(sample["document_text"], topk_tokens, tokenizer)
        )

        if len(batch_qids) == batch_size:
            if "conditional_explanation" in option:
                outputs = api_func(model, batch_queries, batch_documents, label)
            elif option == "relevance":
                outputs = api_func(model, batch_queries, batch_documents, dataset)
            else:
                outputs = api_func(model, batch_queries, batch_documents)
            for qid, docid, output in zip(batch_qids, batch_docids, outputs):
                output_path = f"{output_dir}/{qid}+{docid}.pkl"
                with open(output_path, "wb") as fout:
                    pickle.dump(output, fout)
            batch_qids, batch_queries, batch_docids, batch_documents = [], [], [], []

    if len(batch_qids) > 0:
        if "conditional_explanation" in option:
            outputs = api_func(model, batch_queries, batch_documents, label)
        else:
            outputs = api_func(model, batch_queries, batch_documents)
        for qid, docid, output in zip(batch_qids, batch_docids, outputs):
            output_path = f"{output_dir}/{qid}+{docid}.pkl"
            with open(output_path, "wb") as fout:
                pickle.dump(output, fout)