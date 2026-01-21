#!/home/saksham/CS496/code/vllm/.venv/bin/python
import argparse
import random
import string
import time
import os

from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams
import sys
sys.path.append("..")
from system_prompt import queries, system_prompt as sys_prompt


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str,
                        default="/home/saksham/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/")

    parser.add_argument("--num_families", type=int, default=2)
    parser.add_argument("--max_batch_size", type=int, default=500)
    parser.add_argument("--randomness", type=int, default=1,
                        help="Chunk size for randomized scheduling")
    parser.add_argument("--len_factor", type=float, default=1.0)
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--write_output", action="store_true")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prefixes = generate_prefixes(args.num_families)
    prompts = []
    families = []

    # ----- SYSTEM PROMPT LENGTH SCALING -----
    m_sys_prompt = sys_prompt
    if args.len_factor > 1.0:
        m_sys_prompt *= int(args.len_factor)
    else:
        m_sys_prompt = m_sys_prompt[:int(len(m_sys_prompt) * args.len_factor)]

    # ----- PROMPT GENERATION -----
    for prefix in prefixes:
        for q in (queries * args.multiplier):
            modified_system_prompt = prefix + " " + m_sys_prompt
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

    # ----- RANDOMIZED CHUNK SHUFFLING -----
    paired = list(zip(prompts, families))
    if args.randomness > 0:
        chunks = [
            paired[i:i + args.randomness]
            for i in range(0, len(paired), args.randomness)
        ]
        random.shuffle(chunks)
        paired = [item for chunk in chunks for item in chunk]

    prompts, families = zip(*paired)
    prompts = list(prompts)

    # ----- MODEL INITIALIZATION -----
    llm = LLM(
        args.model_name,
        tensor_parallel_size=1,
        max_num_seqs=args.max_batch_size,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=args.max_new_tokens
    )

    print(
        "Total requests:", len(prompts),
        "| max_batch_size:", args.max_batch_size
    )

    # ----- RUN WORKLOAD -----
    start_time = time.time()
    print("Simulation started at:", start_time)

    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()
    throughput = len(outputs) / (end_time - start_time)

    print("Simulation completed at:", end_time)
    print(
        "Throughput:", throughput, "req/s |",
        "families:", args.num_families,
        "| batch:", args.max_batch_size,
        "| randomness:", args.randomness,
        "| len_factor:", args.len_factor,
        "| max_new_tokens:", args.max_new_tokens,
    )

    # ----- OPTIONAL LOGGING -----
    if args.write_output:
        with open("nanovllm_multiple_families_output.txt", "a") as f:
            f.write(
                f"Throughput Test | families={args.num_families} "
                f"batch={args.max_batch_size} "
                f"randomness={args.randomness} "
                f"len_factor={args.len_factor} "
                f"multiplier={args.multiplier} "
                f"max_new_tokens={args.max_new_tokens} "
                f"num_prompts={len(prompts)} "
                f"throughput={throughput}\n"
            )


if __name__ == "__main__":
    main()
