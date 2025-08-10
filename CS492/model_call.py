import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class modelCall:
    _MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(self):
        self.tokenizer = None
        self.model = None

    def get_llm_response(self, message):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                self._MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
            )
        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    llm_input_message = "What is the capital of France?"
    model = modelCall()
    print(model.get_llm_response(llm_input_message))
