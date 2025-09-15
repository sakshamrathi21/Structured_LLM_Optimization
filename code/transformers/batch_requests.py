import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
import time
from system_prompt import system_prompt, queries

class BatchedModel:
    _MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self, use_compile=False, use_flash=False):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2" if use_flash else None,
        )
        if use_compile:
            self.model = torch.compile(self.model)

    def warmup_with_system_prompt(self, system_prompt):
        inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        return outputs.past_key_values

    @staticmethod
    def clone_past_key_values(pkv):
        if isinstance(pkv, Cache):
            if isinstance(pkv, DynamicCache):
                new_cache = DynamicCache()
                for layer_idx in range(len(pkv.key_cache)):
                    key_states = pkv.key_cache[layer_idx].clone()
                    value_states = pkv.value_cache[layer_idx].clone()
                    new_cache.update(key_states, value_states, layer_idx)
                return new_cache
            else:
                new_cache = pkv.__class__()
                for layer_idx in range(len(pkv)):
                    k, v = (t.clone() for t in pkv[layer_idx])
                    new_cache.update(k, v, layer_idx)
                return new_cache
        else:
            return tuple(tuple(t.clone() for t in layer) for layer in pkv)

    def generate_batch_from_cache(self, pkv, queries, max_new_tokens=100):
        """
        Generate responses for a batch of queries, all sharing the same system prompt cache.
        """
        batch_inputs = self.tokenizer(queries, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]

        batch_size = input_ids.shape[0]
        generated = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        # Clone system prompt cache per batch element
        batched_pkv = [self.clone_past_key_values(pkv) for _ in range(batch_size)]

        for _ in range(max_new_tokens):
            # Forward each unfinished sequence
            new_input_ids = []
            new_pkv = []
            for i in range(batch_size):
                if finished[i]:
                    new_input_ids.append(torch.tensor([[self.tokenizer.pad_token_id]], device=self.model.device))
                    new_pkv.append(batched_pkv[i])
                    continue
                outputs = self.model(
                    input_ids=input_ids[i].unsqueeze(0),
                    past_key_values=batched_pkv[i],
                    use_cache=True,
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated[i].append(next_token)
                batched_pkv[i] = outputs.past_key_values
                if next_token.item() == self.tokenizer.eos_token_id:
                    finished[i] = True
                new_input_ids.append(next_token)
                new_pkv.append(batched_pkv[i])

            input_ids = torch.cat(new_input_ids, dim=0)
            batched_pkv = new_pkv
            if all(finished):
                break

        # Decode results
        decoded = []
        for seq in generated:
            if len(seq) == 0:
                decoded.append("")
            else:
                tokens = torch.cat(seq, dim=1)
                decoded.append(self.tokenizer.decode(tokens[0], skip_special_tokens=True))

        return decoded

    def generate_batch_builtin(self, system_prompt, queries, max_new_tokens=100, batch_size=1):
        results = []
        for i in range(0, len(queries), batch_size):
            chunk = queries[i:i+batch_size]
            full_prompts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q}
                    ],
                    tokenize=False
                )
                for q in chunk
            ]

            
            batch_inputs = self.tokenizer(
                full_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            generated_tokens = outputs[:, batch_inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
        
        return results



if __name__ == "__main__":

    model = BatchedModel(use_compile=False, use_flash=False)

    # print("\n=== Cached system prompt batch generation ===")
    # pkv = model.warmup_with_system_prompt(system_prompt)
    # start_time = time.time()
    # responses = model.generate_batch_from_cache(pkv, queries, max_new_tokens=80)
    # end_time = time.time()
    # print(end_time - start_time)
    # for i, r in enumerate(responses):
    #     print(f"Q{i+1}: {queries[i]}\nA{i+1}: {r}\n")

    print("\n=== Built-in batched generation === (num_queries =", len(queries), ")")
    start_time = time.time()
    responses2 = model.generate_batch_builtin(system_prompt, queries, max_new_tokens=80)
    end_time = time.time()
    print(end_time - start_time)
    # for i, r in enumerate(responses2):
    #     print(f"Q{i+1}: {queries[i]}\nA{i+1}: {r}\n")
