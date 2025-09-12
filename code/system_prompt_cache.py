import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from transformers.cache_utils import Cache, DynamicCache


class ModelCall:
    _MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def __init__(self):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self._MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def warmup_with_system_prompt(self, system_prompt):
        """Run system prompt once and get past_key_values."""
        inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        return outputs.past_key_values

    def continue_from_cache_manual(self, pkv, new_query, max_new_tokens=100):
        input_ids = self.tokenizer(new_query, return_tensors="pt").to(self.model.device)["input_ids"]

        generated = []
        # pkv = self.clone_past_key_values(past_key_values)  # safe copy

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=pkv,
                    use_cache=True
                )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            input_ids = next_token
            pkv = outputs.past_key_values

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        generated_ids = torch.cat(generated, dim=1)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    @staticmethod
    def clone_past_key_values(pkv):
        if isinstance(pkv, Cache):
            # DynamicCache / StaticCache cloning
            if isinstance(pkv, DynamicCache):
                new_cache = DynamicCache()
                for layer_idx in range(len(pkv.key_cache)):
                    key_states = pkv.key_cache[layer_idx].clone()
                    value_states = pkv.value_cache[layer_idx].clone()
                    new_cache.update(key_states, value_states, layer_idx)
                return new_cache
            else:
                # Other Cache subclasses
                new_cache = pkv.__class__()
                for layer_idx in range(len(pkv)):
                    k, v = (t.clone() for t in pkv[layer_idx])
                    new_cache.update(k, v, layer_idx)
                return new_cache
        else:
            # Old tuple-of-tuples past_key_values
            return tuple(tuple(t.clone() for t in layer) for layer in pkv)


    def run_full_prompt(self, system_prompt, new_query, max_new_tokens=100):
        """Generate by feeding system prompt + query together."""
        full_prompt = system_prompt + "\n" + new_query
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)["input_ids"]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids, use_cache=True)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                # print(self.tokenizer.decode(next_token[0], skip_special_tokens=True), end="", flush=True)

        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    def profile_pipeline(self, system_prompt, query, max_new_tokens=50):
        times = {}

        # Tokenization system prompt
        t0 = time.time()
        system_inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.model.device)
        times["tokenize_system"] = time.time() - t0

        # System prompt forward (fill KV cache)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**system_inputs, use_cache=True)
        pkv = outputs.past_key_values
        times["system_forward"] = time.time() - t0

        # Tokenize query
        t0 = time.time()
        query_inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)["input_ids"]
        times["tokenize_query"] = time.time() - t0

        # Token generation loop
        generated = []
        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                step_start = time.time()
                outputs = self.model(
                    input_ids=query_inputs,
                    past_key_values=pkv,
                    use_cache=True
                )
                step_time = time.time() - step_start
                times.setdefault("per_token", []).append(step_time)

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                query_inputs = next_token
                pkv = outputs.past_key_values
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        times["generation_total"] = time.time() - t0

        # Final decode
        t0 = time.time()
        gen_ids = torch.cat(generated, dim=1)
        decoded = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        times["decode_tokens"] = time.time() - t0

        return decoded, times


if __name__ == "__main__":
    model = ModelCall()

    system_prompt = """The lecture introduces semaphores, a synchronization primitive similar to condition variables, used to coordinate threads. A semaphore maintains a counter, initialized to some value. The down (wait) operation decrements the counter and blocks the thread if it becomes negative, while the up (post) operation increments the counter and wakes a blocked thread. The counter cannot be directly accessed, only modified via these operations.
Semaphores can be used for mutual exclusion (binary semaphores), functioning like locks. They are also useful for synchronization between threads, e.g., enforcing execution order (T1 before T2) or ensuring two threads finish one step before starting another. Incorrect usage can cause deadlocks, such as when both threads wait on each other.
The producer-consumer problem is revisited using semaphores. Two semaphores track empty and filled buffer slots, while a binary semaphore (mutex) ensures mutual exclusion on buffer access. Misuse, like holding locks during blocking down calls, can also lead to deadlocks.
Guidelines emphasize careful initialization, ensuring every down has a corresponding up, and using separate semaphores for signaling and locking. The lecture also covers batched processing with semaphores, showing how request threads and a batch processor can coordinate efficiently.
Semaphores thus provide flexible yet tricky synchronization, requiring careful design to avoid deadlocks.
    """
    query1 = """How does the down operation on a semaphore synchronize threads during execution?"""

    query2 = """Why is an extra binary semaphore (mutex) needed in producer-consumer implementation?"""

    decoded, times = model.profile_pipeline(system_prompt, query1)

    print("\n=== Profiling Results ===")
    for stage, t in times.items():
        if isinstance(t, list):
            print(f"{stage}: mean {sum(t)/len(t):.4f}s, first token {t[0]:.4f}s")
        else:
            print(f"{stage}: {t:.4f}s")

    print("\nGenerated response:", decoded)

    exit(1)

    # # # ---------------- Experiment 1: Cached system prompt ----------------
    # print("\nRunning cached prompt experiment...")
    # t0_start = time.time()
    # past = model.warmup_with_system_prompt(system_prompt)
    # past1 = model.clone_past_key_values(past)
    # t0_end = time.time()
    # print("[SYSTEM PROMPT] Time taken for filling system prompt along with cloning is: ", round(t0_end - t0_start, 3), "seconds")
    # t1_start = time.time()
    # resp1 = model.continue_from_cache_manual(past, query1)
    # resp2 = model.continue_from_cache_manual(past1, query2)
    # t1_end = time.time()

    # print("[CACHE] Time for 2 queries:", round(t1_end - t1_start, 3), "seconds")
    # print("[CACHE] Q1:", resp1)
    # print("-----------------------")
    # print("[CACHE] Q2:", resp2)

    # # # ---------------- Experiment 2: Full prompt every time ----------------
    # print("\nRunning full prompt experiment...")
    # t2_start = time.time()
    # resp3 = model.run_full_prompt(system_prompt, query1)
    # resp4 = model.run_full_prompt(system_prompt, query2)
    # t2_end = time.time()

    # print("[FULL] Time for 2 queries:", round(t2_end - t2_start, 3), "seconds")
    # # print("[FULL] Q1:", resp3)
    # # print("[FULL] Q2:", resp4)

    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #             record_shapes=True,
    #             profile_memory=True,
    #             with_stack=True,
    #             on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")) as prof:
    #     with record_function("experiment_cached_queries"):
    #         past = model.warmup_with_system_prompt(system_prompt)
    #         resp1 = model.continue_from_cache_manual(past, query1)
    #         resp2 = model.continue_from_cache_manual(past, query2)
    # with open("profile_report.txt", "w") as f:
    #     f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1000))
    # prof.export_chrome_trace("trace.json")

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     past = model.warmup_with_system_prompt(system_prompt)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        resp1 = model.run_full_prompt(system_prompt, query1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     resp2 = model.continue_from_cache_manual(past, query2)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))



    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
