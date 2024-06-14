"""Utilities for RankLlama model"""
import torch
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel, PeftConfig
from vllm import LLM, SamplingParams


class RankLlama:
    """Wrapper for RankLlama model"""

    def __init__(
        self,
        peft_model_name="castorini/rankllama-v1-7b-lora-passage",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        device="cuda",
    ):
        self.device = device
        self.model = self.get_peft_model(peft_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def get_peft_model(self, peft_model_name):
        config = PeftConfig.from_pretrained(peft_model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=1,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        # model = model.to(self.device)
        model.eval()
        return model

    def get_qd_score(self, query, doc):
        inputs = self.tokenizer(
            f"query: {query}",
            f"document: {doc}",
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=1024,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            score = logits[0][0].item()
        return score



class MonoLlama:
    """Wrapper for MonoLlama model"""

    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        device="cuda",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = self.load_model(model_name)
        self.template = None

    def set_prompt(self, prompt_type, max_scale=None):

        if prompt_type == "tf-zero":

            # self.template = (
            #     "<s>[INST] <<SYS>> You are a reliable NIST assessor. Given a query and a document, your task is to answer either 'true' or 'false' as to whether they are relevant or not. <</SYS>>"
            #     "query: \"{input_query}\" document: \"{input_document}\" \n Are they relevant (true) or not relevant (false): [/INST]"
            # )

            self.template = (
                "<s>[INST]<<SYS>> For the following query and document, judge whether they are relevant. Output “Yes” or “No”<</SYS>>.\n"
                "Query: {input_query}\nDocument: {input_document}\nOutput: [/INST]"
            )
        elif prompt_type == "scale-zero":
            if max_scale is None:
                raise ValueError("max_scale must be specified for scale-zero prompt")
            self.template = (
                f"<s>[INST]<<SYS>> For the following query and document, judge their level of relevance. Output an integer between 0 and {max_scale} where 0 indicates irrelevance.<</SYS>>.\n"
                "Query: {input_query}\nDocument: {input_document}\nOutput: [/INST]"
            )
        else:
            raise NotImplementedError

    def load_model(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.eval()

        return model

    def get_qd_score(self, query, doc, top_p=0.9):

        model_input = self.template.format(input_query=query, input_document=doc)
        tokenized_input = self.tokenizer(model_input, return_tensors="pt")
        retry = 0
        pos_id, neg_id = 3869, 1939
        logits = [None, None]
        while retry < 5:
            outputs = self.model.generate(
                tokenized_input.input_ids.to("cuda"),
                max_new_tokens=10,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                top_p=top_p
                )
            for decoding_step, scores in enumerate(outputs.scores):
                scores = scores.squeeze()
                nonzero_indices = torch.nonzero(scores != float('-inf')).squeeze(dim=1)
                nonzero_indices = nonzero_indices.detach().cpu().numpy().tolist()

                if pos_id in nonzero_indices and neg_id in nonzero_indices:
                    nonzero_scores = scores[nonzero_indices]
                    print(f"Decoding step {decoding_step}:")

                    logits[0] = scores[pos_id].item()
                    logits[1] = scores[neg_id].item()
                    score = softmax(logits, axis=0)[0]
                    return score, logits
            retry += 1
        return logits


class MonoLlama_Vllm:
    """Wrapper for MonoLlama model"""

    def __init__(
        self,
        model_name="TheBloke/Llama-2-13B-chat-AWQ",
        number_of_outputs_per_input=1,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n = number_of_outputs_per_input
        self.sampling_params = SamplingParams(
            n=self.n,
            temperature=1.0,
            top_p=1.0,
            logprobs=50,
            max_tokens=3,
        )
        self.llm = self.load_model(model_name)
        self.template = None
        self.pos_id, self.neg_id = 3869, 1939

    def set_prompt(self, prompt_type, max_scale=None):

        if prompt_type == "tf-zero":
            self.template = (
                "<s>[INST]<<SYS>> For the following query and document, judge whether they are relevant. Output “Yes” or “No”<</SYS>>.\n"
                "Query: {input_query}\nDocument: {input_document}\nOutput: [/INST]"
            )
        elif prompt_type == "scale-zero":
            if max_scale is None:
                raise ValueError("max_scale must be specified for scale-zero prompt")
            # self.template = (
            #     f"<s>[INST]<<SYS>> For the following query and document, judge their level of relevance. Output an integer between 0 and {max_scale} where 0 indicates irrelevance.<</SYS>>.\n"
            #     "Query: {input_query}\nDocument: {input_document}\nOutput: [/INST]"
            # )
        else:
            raise NotImplementedError

    def load_model(self, model_name):
        llm = LLM(model=model_name, quantization="awq")
        return llm

    def get_qd_score(self, query, docs):
        prompts = [self.template.format(input_query=query, input_document=doc) for doc in docs]
        outputs = self.llm.generate(
            prompts,
            sampling_params=self.sampling_params
            )

        list_of_logits = []
        list_of_scores = []
        
        for output in outputs:
            # generated_token_ids = output.outputs[0].token_ids
            # print(self.tokenizer.decode(generated_token_ids))
            # print([self.tokenizer.decode(x) for x in generated_token_ids])
            # print([self.tokenizer.convert_ids_to_tokens(x) for x in generated_token_ids])

            # specifically look at decoding step 2
            logprobs_at_the_step = output.outputs[0].logprobs[1]
            if self.pos_id in logprobs_at_the_step and self.neg_id in logprobs_at_the_step:
                logits = [logprobs_at_the_step[self.pos_id], logprobs_at_the_step[self.neg_id]]
                score = softmax(logits, axis=0)[0]
                list_of_logits.append(logits)
                list_of_scores.append(score)
            else:
                list_of_logits.append([None, None])
                list_of_scores.append([None, None])
        
        return list_of_scores, list_of_logits

                

            