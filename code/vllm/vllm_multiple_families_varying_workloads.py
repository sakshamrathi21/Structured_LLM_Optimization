import random
import string
import time
import sys as system
import argparse

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt as sys_prompt
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
    # model_name = "qwen/qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_families", type=int, default=2)
    parser.add_argument("--req_per_sec", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=500)
    parser.add_argument("--randomness", type=int, default=1)
    parser.add_argument("--len_factor", type=float, default=1.0)
    parser.add_argument("--policy", type=str, default="fcfs", choices=["fcfs", "priority", "radix_cost", "family_load", "greedy_l2"])
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--write_output", action="store_true", help="Whether to write output to a file.")
    args = parser.parse_args()

    num_families = args.num_families
    req_per_sec = args.req_per_sec
    max_batch_size = args.max_batch_size
    randomness = args.randomness
    len_factor = args.len_factor
    prefixes = generate_prefixes(num_families)
    prompts = []
    families = []  
    current_load = (num_families * len(queries)) / req_per_sec
    multiplier = max(1, 10/current_load)
    multiplier = int(multiplier)
    multiplier = args.multiplier
    system_prompts = []
    m_sys_prompt = sys_prompt
    if len_factor > 1.0:
        m_sys_prompt *= int(len_factor)
    else:
        m_sys_prompt = m_sys_prompt[0:int(len(m_sys_prompt)*len_factor)]
    system_prompt = m_sys_prompt
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
        cnt = 0
        for q in (queries*multiplier):
            cnt += 1
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
    if randomness > 0:
        chunked = [paired[i:i+randomness] for i in range(0, len(paired), randomness)]
        random.shuffle(chunked)
        paired = [item for chunk in chunked for item in chunk]
    prompts, families = zip(*paired)

    enable_chunked_prefill = True
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        max_num_seqs=max_batch_size,
        scheduling_policy=args.policy,
    )

    initial_start_time = time.time()
    prompts = list(prompts)
    if not req_per_sec:
        llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens))
    else:
        for i in range(0, len(prompts), req_per_sec):
            start_time = time.time()
            llm.generate(prompts[i:i+req_per_sec], SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens))
            time.sleep(max(0, 1 - time.time() + start_time))
    end_time = time.time()
    throughput = len(prompts) / (end_time - initial_start_time)
    print(time.time(), "Completed all requests.")
    print(
        "Throughput: ",
        throughput,
        "req/s (enabling chunked prefill:",
        enable_chunked_prefill,
        ")",
    )
    if args.write_output:
        with open("vllm_multiple_families_varying_workloads_output.txt", "a") as f:
            f.write(f"Throughput Test with {num_families} families, {req_per_sec} req/sec and max_batch_size of {max_batch_size} (randomness: {randomness}) (len factor: {len_factor}) (model_name: {model_name}) (policy: {args.policy}) (max_new_tokens: {args.max_new_tokens}) (num_prompts: {len(prompts)}): {throughput}\n")

if __name__ == "__main__":
    main()
