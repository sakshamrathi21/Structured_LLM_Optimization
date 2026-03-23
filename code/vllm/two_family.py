import time
import argparse
import random
import subprocess

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

    A_count = int(N * fraction)
    B_count = N - A_count

    print("\n[INFO] Batch composition:")
    print(f"  Family A requests : {A_count}")
    print(f"  Family B requests : {B_count}")
    print(f"  Tokens per prompt : {L}")

    prompts = []

    def build_family_prompt(tag, idx):

        text = generate_exact_tokens(tokenizer, L, f"{tag}")

        ids = tokenizer.encode(text, add_special_tokens=False)[:L]
        text = tokenizer.decode(ids)

        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": f"Request {tag}-{idx}: explain AI briefly."},
        ]

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return text + "\n\nExplain AI briefly."

    for i in range(A_count):
        prompts.append(build_family_prompt("A", i))

    for i in range(B_count):
        prompts.append(build_family_prompt("B", i))

    random.shuffle(prompts)

    return prompts


def build_dcgmi_filename(args):
    name = (
        f"dcgmi_twofamily_model-{args.model.replace('/', '_')}"
        f"_L-{args.L}"
        f"_frac-{args.fraction}"
        f"_reqs-{args.requests}"
        f"_batch-{args.batch_size}"
        f"_maxnew-{args.max_new_tokens}"
        f".log"
    )
    return name


def run_experiment(llm, prompts, max_new_tokens, iterations, write_dcgmi=False, args=None):

    params = SamplingParams(temperature=0.7, max_tokens=max_new_tokens)

    print("\n==============================")
    print("Warm pass: building KV cache")
    print("==============================")

    llm.generate(prompts, params)

    print("\nWarm pass complete.")
    print("Starting timed runs...\n")

    times = []

    # ----- CONDITIONAL DCGMI WRAPPING -----
    dcgmi_proc = None
    dcgmi_file = None

    if write_dcgmi and args:
        dcgmi_file = build_dcgmi_filename(args)
        print(f"[INFO] Logging DCGMI output to: {dcgmi_file}")

        dcgmi_cmd = ["sudo", "dcgmi", "dmon", "-e", "1002,1003,1004,1005", "-d", "1"]

        f_dcgmi = open(dcgmi_file, "w")
        dcgmi_proc = subprocess.Popen(dcgmi_cmd, stdout=f_dcgmi, stderr=subprocess.STDOUT)

    try:
        for i in range(iterations):

            start = time.time()

            llm.generate(prompts, params)
            # print("scheduler end time:", time.time())
            elapsed = time.time() - start
            throughput = len(prompts) / elapsed

            print(
                f"[ITER {i+1}] "
                f"time = {elapsed:.3f}s | "
                f"throughput = {throughput:.2f} req/s"
            )

            times.append(elapsed)

    finally:
        if dcgmi_proc:
            print("[INFO] Stopping DCGMI monitoring...")
            dcgmi_proc.terminate()
            try:
                dcgmi_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dcgmi_proc.kill()
            f_dcgmi.close()

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
        description="Two-family batch mixture throughput experiment"
    )

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

    parser.add_argument(
        "--L", type=int, default=10000, help="exact system prompt token length"
    )

    parser.add_argument(
        "--fraction", type=float, default=0.5, help="fraction of family A (0-1)"
    )

    parser.add_argument("--requests", type=int, default=500)

    parser.add_argument("--iterations", type=int, default=3)

    parser.add_argument("--max_new_tokens", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--write_dcgmi", action="store_true", help="Enable DCGMI monitoring and logging"
    )

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
        gpu_memory_utilization=0.98,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=32768,
    )

    run_experiment(llm, prompts, args.max_new_tokens, args.iterations, args.write_dcgmi, args)


if __name__ == "__main__":
    main()