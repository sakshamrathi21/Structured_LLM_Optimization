import random
import string
import time
import sys as system

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt
from vllm import LLM, SamplingParams


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_families = int(system.argv[1]) if len(system.argv) > 1 else 10
    max_num_seqs = int(system.argv[2]) if len(system.argv) > 2 else 100
    prefixes = generate_prefixes(num_families)  # 10 families
    prompts = []
    families = []  

    for prefix in prefixes: 
        for q in queries:
            modified_system_prompt = prefix + " " + system_prompt
            chat_messages = [
                {"role": "system", "content": modified_system_prompt},
                {"role": "user", "content": q},
            ]
            try:
                full_prompt = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                full_prompt = modified_system_prompt + " Question: " + q

            prompts.append(full_prompt)
            families.append(prefix)  # <-- mark which family it belongs to

    print(f"Generated {len(prompts)} prompts in total.")

    enable_chunked_prefill = True
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.98,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        max_num_seqs=max_num_seqs,
    )
    if max_num_seqs < 100:
        multiplier = 1000 // len(prompts)
    elif max_num_seqs < 1000:
        multiplier = 10000 // len(prompts)
    else:
        multiplier = 100000 // len(prompts)
    prompts = prompts * multiplier
    families = families * multiplier
    paired = list(zip(prompts, families))
    random.shuffle(paired)
    prompts, families = zip(*paired)
    prompts = list(prompts)
    families = list(families)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    print("Total number of requests to simulate:", len(prompts), "for max_num_seqs =", max_num_seqs)
    outputs = llm.generate(list(prompts), sampling_params)
    print("Simulation completed at time:", time.time())

    # print(
    #     "\nThroughput:",
    #     len(outputs) / (end_time - start_time),
    #     "req/s (enabling chunked prefill:",
    #     enable_chunked_prefill,
    #     ")",
    # )

if __name__ == "__main__":
    main()
