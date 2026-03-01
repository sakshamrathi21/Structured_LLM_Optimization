#!/home/saksham/CS496/code/vllm/.venv/bin/python
import random
import string
import time
import argparse

from transformers import AutoTokenizer
from system_prompt import queries, system_prompt as sys_prompt
import sglang as sgl


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--num_families", type=int, default=5)
    parser.add_argument("--req_per_sec", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=500)
    parser.add_argument("--randomness", type=int, default=1)
    parser.add_argument("--len_factor", type=float, default=5.0)
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--write_output", action="store_true")
    parser.add_argument("--policy", type=str, default="dfs-weight")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    prefixes = generate_prefixes(args.num_families)
    prompts = []
    families = []

    # Adjust system prompt length
    if args.len_factor > 1.0:
        system_prompt = sys_prompt * int(args.len_factor)
    else:
        system_prompt = sys_prompt[: int(len(sys_prompt) * args.len_factor)]

    # Build prompts
    for prefix in prefixes:
        for q in queries * args.multiplier:
            chat_messages = [
                {"role": "system", "content": prefix + " " + system_prompt},
                {"role": "user", "content": q},
            ]
            try:
                full_prompt = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                full_prompt = prefix + " " + system_prompt + " Question: " + q

            prompts.append(full_prompt)
            families.append(prefix)

    print(f"Generated {len(prompts)} prompts in total.")

    # Randomization (same logic as vLLM script)
    paired = list(zip(prompts, families))
    if args.randomness > 0:
        chunked = [
            paired[i : i + args.randomness]
            for i in range(0, len(paired), args.randomness)
        ]
        random.shuffle(chunked)
        paired = [item for chunk in chunked for item in chunk]

    prompts, families = zip(*paired)
    prompts = list(prompts)

    # Initialize SGLang engine
    llm = sgl.Engine(
        model_path=args.model_name,
        mem_fraction_static=0.98,
        max_total_tokens=32768,
        max_running_requests=args.max_batch_size,
        schedule_policy=args.policy,
    )

    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": args.max_new_tokens,
    }

    # Run benchmark
    start_time = time.time()
    llm.generate(
        prompts,
        sampling_params,
        poisson_rate=args.req_per_sec,
    )
    end_time = time.time()

    throughput = len(prompts) / (end_time - start_time)
    print("Throughput:", throughput, "req/s")

    if args.write_output:
        with open("vllm_multiple_families_varying_workloads_output.txt", "a") as f:
            f.write(
                f"Throughput Test | families={args.num_families} | "
                f"req/s={args.req_per_sec} | max_batch={args.max_batch_size} | "
                f"randomness={args.randomness} | len_factor={args.len_factor} | "
                f"model={args.model_name} | max_new_tokens={args.max_new_tokens} | "
                f"num_prompts={len(prompts)} | throughput={throughput} | policy={args.policy}\n"
            )


if __name__ == "__main__":
    main()
