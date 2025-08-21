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

    def continue_from_cache_manual(self, past_key_values, new_query, max_new_tokens=50):
        input_ids = self.tokenizer(new_query, return_tensors="pt").to(self.model.device)["input_ids"]

        generated = []
        pkv = self.clone_past_key_values(past_key_values)  # safe copy

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


    def run_full_prompt(self, system_prompt, new_query, max_new_tokens=50):
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


if __name__ == "__main__":
    model = ModelCall()

    system_prompt = """You are an expert in computer science, specializing in operating systems. You have deep knowledge of OS concepts including process scheduling, virtual memory, file systems, I/O management, concurrency, deadlocks, and kernel-level optimizations.
    """
    query1 = """Explain in detail how the Linux Completely Fair Scheduler (CFS) ensures fairness among processes, and analyze its time complexity compared to traditional round-robin scheduling."""

    query2 = """Describe the sequence of steps the Linux kernel follows when handling a page fault in a process running inside a virtual memory system, including all key data structures involved."""


    # # ---------------- Experiment 1: Cached system prompt ----------------
    # print("\nRunning cached prompt experiment...")
    # past = model.warmup_with_system_prompt(system_prompt)

    # t1_start = time.time()
    # resp1 = model.continue_from_cache_manual(past, query1)
    # resp2 = model.continue_from_cache_manual(past, query2)
    # t1_end = time.time()

    # print("[CACHE] Time for 2 queries:", round(t1_end - t1_start, 3), "seconds")
    # # print("[CACHE] Q1:", resp1)
    # # print("[CACHE] Q2:", resp2)

    # # ---------------- Experiment 2: Full prompt every time ----------------
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        past = model.warmup_with_system_prompt(system_prompt)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        resp1 = model.continue_from_cache_manual(past, query1)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        resp2 = model.continue_from_cache_manual(past, query2)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=240))



    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
