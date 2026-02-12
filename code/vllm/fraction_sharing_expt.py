import time
import argparse
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_exact_tokens(tokenizer, target_tokens, tag):
    base = f"{tag} "
    toks = tokenizer.encode(base, add_special_tokens=False)

    if len(toks) == 0:
        raise RuntimeError("Tokenizer produced empty tokens")

    reps = (target_tokens // len(toks)) + 2
    text = base * reps

    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]

    while len(ids) < target_tokens:
        ids.append(ids[-1])

    return tokenizer.decode(ids)


def build_prompts(tokenizer, L, N, fraction):
    shared_tokens = int(L * fraction)
    unique_tokens = L - shared_tokens

    print("\n[INFO] Prompt composition:")
    print(f"  Shared tokens : {shared_tokens}")
    print(f"  Unique tokens : {unique_tokens}")

    shared_prefix = generate_exact_tokens(tokenizer, shared_tokens, "SHARED")

    prompts = []

    for i in range(N):

        unique_suffix = generate_exact_tokens(tokenizer, unique_tokens, f"U{i}")

        system_prompt = shared_prefix + " " + unique_suffix

        # enforce exact L
        ids = tokenizer.encode(system_prompt, add_special_tokens=False)[:L]
        system_prompt = tokenizer.decode(ids)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Request {i}: explain AI briefly."},
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + "\n\nQuestion: explain AI."

        prompts.append(full_prompt)

    random.shuffle(prompts)
    return prompts


def run_experiment(llm, prompts, max_new_tokens, iterations):

    params = SamplingParams(temperature=0.7, max_tokens=max_new_tokens)

    print("\n==============================")
    print("Warm pass: building KV cache")
    print("==============================")

    llm.generate(prompts, params)

    print("\nWarm pass complete.")
    print("Starting timed runs...\n")

    times = []

    for i in range(iterations):

        start = time.time()

        llm.generate(prompts, params)

        elapsed = time.time() - start
        throughput = len(prompts) / elapsed

        print(
            f"[ITER {i+1}] "
            f"time = {elapsed:.3f}s | "
            f"throughput = {throughput:.2f} req/s"
        )

        times.append(elapsed)

    avg_time = sum(times) / len(times)
    avg_throughput = len(prompts) / avg_time

    print("\n====================================")
    print("AVERAGED RESULTS")
    print("====================================")
    print(f"Average iteration time : {avg_time:.3f}s")
    print(f"Average throughput     : {avg_throughput:.2f} req/s")
    print("====================================\n")


def main():

    parser = argparse.ArgumentParser(
        description="Fractional prefix sharing throughput experiment"
    )

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    parser.add_argument(
        "--L", type=int, default=1200, help="system prompt token length"
    )

    parser.add_argument(
        "--fraction", type=float, default=0.5, help="shared prefix fraction (0–1)"
    )

    parser.add_argument("--requests", type=int, default=500)

    parser.add_argument("--iterations", type=int, default=3)

    parser.add_argument("--max_new_tokens", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Building prompts...")
    prompts = build_prompts(tokenizer, args.L, args.requests, args.fraction)

    print(f"\nGenerated {len(prompts)} prompts")

    print("\nInitializing vLLM...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=args.batch_size,
    )

    run_experiment(llm, prompts, args.max_new_tokens, args.iterations)


if __name__ == "__main__":
    main()
