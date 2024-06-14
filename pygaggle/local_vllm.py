import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Llama_Vllm:
    """Wrapper for MonoLlama model"""

    def __init__(
        self,
        model_name="TheBloke/Llama-2-13B-chat-AWQ",
        number_of_outputs_per_input=1,
        top_p=0.95,
        temperature=1.0,
        max_tokens=256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.n = number_of_outputs_per_input
        self.top_p = top_p
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sampling_params = SamplingParams(
            n=self.n,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.llm = self.load_model(model_name)

    def load_model(self, model_name):
        llm = LLM(
            model=model_name,
            tokenizer="hf-internal-testing/llama-tokenizer",
            quantization="awq",
            swap_space=16,
            tensor_parallel_size=torch.cuda.device_count()
            )
        return llm
