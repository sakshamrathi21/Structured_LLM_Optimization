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


def run():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_families = int(system.argv[1]) if len(system.argv) > 1 else 10
    max_num_seqs = int(system.argv[2]) if len(system.argv) > 2 else 100
    whether_random = int(system.argv[3]) if len(system.argv) > 3 else 1
    prefixes = generate_prefixes(num_families)
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
            families.append(prefix)

    print(f"Generated {len(prompts)} prompts in total.")

    llm = sgl.Engine(
        model_path=model_name,
        mem_fraction_static=0.98,
        max_total_tokens=32768,
        max_running_requests=max_num_seqs,
    )
    paired = list(zip(prompts, families))
    if whether_random:
        random.shuffle(paired)
    prompts, families = zip(*paired)
    prompts = list(prompts)
    families = list(families)
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 30}
    print(
        "Total number of requests to simulate:",
        len(prompts),
        "for max_num_seqs =",
        max_num_seqs,
    )
    initial_time = time.time()
    print("Simulation started at time:", initial_time)
    outputs = llm.generate(list(prompts), sampling_params)
    print("Simulation completed at time:", time.time())
    print(
        "Throughput of SGLANG:",
        len(outputs) / (time.time() - initial_time),
        "req/s",
        "for number of families =",
        num_families,
        "and max_num_seqs =",
        max_num_seqs,
        "whether random =",
        whether_random,
    )


if __name__ == "__main__":
    import multiprocessing

    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    run()
