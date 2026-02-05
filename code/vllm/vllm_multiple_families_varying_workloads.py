#!/home/saksham/CS496/code/vllm/.venv/bin/python
import random
import string
import time
import argparse
import subprocess
import os

from typing import get_args

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt as sys_prompt
from vllm import LLM, SamplingParams
from vllm.config.scheduler import SchedulerPolicy


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def build_dcgmi_filename(args):
    name = (
        f"dcgmi_model-{args.model_name.replace('/', '_')}"
        f"_families-{args.num_families}"
        f"_rps-{args.req_per_sec}"
        f"_batch-{args.max_batch_size}"
        f"_rand-{args.randomness}"
        f"_lenf-{args.len_factor}"
        f"_policy-{args.policy}"
        f"_maxnew-{args.max_new_tokens}"
        f".log"
    )
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num_families", type=int, default=2)
    parser.add_argument("--req_per_sec", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=500)
    parser.add_argument("--randomness", type=int, default=1)
    parser.add_argument("--len_factor", type=float, default=1.0)
    parser.add_argument("--policy", type=str, default="fcfs", choices=list(get_args(SchedulerPolicy)))
    parser.add_argument("--multiplier", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--write_output", action="store_true")
    parser.add_argument("--write_dcgmi", action="store_true", help="Enable DCGMI monitoring and logging")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prefixes = generate_prefixes(args.num_families)
    prompts = []
    families = []

    m_sys_prompt = sys_prompt
    if args.len_factor > 1.0:
        m_sys_prompt *= int(args.len_factor)
    else:
        m_sys_prompt = m_sys_prompt[0:int(len(m_sys_prompt) * args.len_factor)]

    system_prompt = m_sys_prompt

    for prefix in prefixes:
        for q in (queries * args.multiplier):
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
    if args.randomness > 0:
        chunked = [paired[i:i+args.randomness] for i in range(0, len(paired), args.randomness)]
        random.shuffle(chunked)
        paired = [item for chunk in chunked for item in chunk]

    prompts, families = zip(*paired)
    prompts = list(prompts)

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.75,
        enable_chunked_prefill=True,
        # max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        max_num_seqs=args.max_batch_size,
        scheduling_policy=args.policy,
        # enforce_eager=True,
    )

    initial_start_time = time.time()

    # ----- CONDITIONAL DCGMI WRAPPING -----

    if args.write_dcgmi:
        dcgmi_file = build_dcgmi_filename(args)
        print(f"[INFO] Logging DCGMI output to: {dcgmi_file}")

        dcgmi_cmd = [
            "sudo", "dcgmi", "dmon",
            "-e", "1002,1003,1004,1005",
            "-d", "1"
        ]

        with open(dcgmi_file, "w") as f:
            dcgmi_proc = subprocess.Popen(
                dcgmi_cmd,
                stdout=f,
                stderr=subprocess.STDOUT
            )

            try:
                llm.generate(
                    prompts,
                    SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens),
                    poisson_lambda=args.req_per_sec
                )
            finally:
                print("[INFO] Stopping DCGMI monitoring...")
                dcgmi_proc.terminate()
                try:
                    dcgmi_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    dcgmi_proc.kill()

    else:
        # Normal run without DCGMI
        llm.generate(
            prompts,
            SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens),
            poisson_lambda=args.req_per_sec
        )

    end_time = time.time()

    throughput = len(prompts) / (end_time - initial_start_time)

    print(time.time(), "Completed all requests.")
    print(f"Throughput Test with {args.num_families} families, "
        f"{args.req_per_sec} req/sec and max_batch_size of {args.max_batch_size} "
        f"(randomness: {args.randomness}) (len factor: {args.len_factor}) "
        f"(model_name: {args.model_name}) (policy: {args.policy}) "
        f"(max_new_tokens: {args.max_new_tokens}) "
        f"(num_prompts: {len(prompts)}): {throughput}\n")

    if args.write_output:
        with open("vllm_multiple_families_varying_workloads_output.txt", "a") as f:
            f.write(
                f"Throughput Test with {args.num_families} families, "
                f"{args.req_per_sec} req/sec and max_batch_size of {args.max_batch_size} "
                f"(randomness: {args.randomness}) (len factor: {args.len_factor}) "
                f"(model_name: {args.model_name}) (policy: {args.policy}) "
                f"(max_new_tokens: {args.max_new_tokens}) "
                f"(num_prompts: {len(prompts)}): {throughput}\n"
            )


if __name__ == "__main__":
    main()
