import random
import string
import time
import sys as system

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt
import sglang as sgl


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
    req_per_sec = int(system.argv[2]) if len(system.argv) > 2 else 20
    max_batch_size = int(system.argv[3]) if len(system.argv) > 3 else 100
    randomness = int(system.argv[4]) if len(system.argv) > 4 else 0
    prefixes = generate_prefixes(num_families)
    prompts = []
    families = []  
    current_load = (num_families * len(queries)) / req_per_sec
    multiplier = max(1, 10/current_load)
    multiplier = int(multiplier)
    system_prompts = []
    for prefix in prefixes:
        chat_messages = [
            {"role": "system", "content": prefix + " " + system_prompt},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = prefix + " " + system_prompt
        system_prompts.append(full_prompt)
    
    for prefix in prefixes: 
        for q in queries*multiplier:
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
            families.append(prefix)

    print(f"Generated {len(prompts)} prompts in total.")
    paired = list(zip(prompts, families))
    if randomness == 2:
        random.shuffle(paired)
    elif randomness == 1:
        # every 50 sized chunk should remain as it is, but shuffle the chunks
        chunked = [paired[i:i+50] for i in range(0, len(paired), 50)]
        random.shuffle(chunked)
        paired = [item for chunk in chunked for item in chunk]
    prompts, families = zip(*paired)

    enable_chunked_prefill = True
    llm = sgl.Engine(
        model_path=model_name, 
        mem_fraction_static=0.98,
        max_total_tokens=32768,
        max_running_requests=max_batch_size,
    )
    sampling_params = {"temperature": 0.7, "max_new_tokens": 30}
    # out = llm.generate(list(system_prompts), sampling_params)
    initial_start_time = time.time()
    for i in range(0, len(prompts), req_per_sec):
        start_time = time.time()
        outputs = llm.generate(list(prompts[i:i+req_per_sec]), sampling_params)
        time.sleep(max(0, 1 - time.time() + start_time))
    end_time = time.time()
    throughput = len(prompts) / (end_time - initial_start_time)
    print(
        "Throughput: ",
        throughput,
        "req/s (enabling chunked prefill:",
        enable_chunked_prefill,
        ")",
    )
    with open("vllm_multiple_families_varying_workloads_output.txt", "a") as f:
        f.write(f"Throughput Test with {num_families} families, {req_per_sec} req/sec and max_batch_size of {max_batch_size} (randomness: {randomness}) for SGLANG: {throughput}\n")

if __name__ == "__main__":
    main()
