from nanovllm import LLM, SamplingParams
import os

import random
import string
import time
import sys as system

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt


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
    whether_random = int(system.argv[3]) if len(system.argv) > 3 else 1
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
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/")
    llm = LLM(model_path, tensor_parallel_size=1, max_num_seqs=max_num_seqs)
    paired = list(zip(prompts, families))
    if whether_random:
        random.shuffle(paired)
    prompts, families = zip(*paired)
    prompts = list(prompts)
    families = list(families)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    print("Total number of requests to simulate:", len(prompts), "for max_num_seqs =", max_num_seqs)
    initial_time = time.time()
    print("Simulation started at time:", initial_time)
    outputs = llm.generate(list(prompts), sampling_params)
    print("Simulation completed at time:", time.time())
    print("Throughput of vLLM:", len(outputs) / (time.time() - initial_time), "req/s", "for number of families =", num_families, "and max_num_seqs =", max_num_seqs, "whether random =", whether_random)
    # print(outputs)

if __name__ == "__main__":
    main()