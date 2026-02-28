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


def build_prompts(tokenizer, L, N, f1, f2, num_families):
    """
    L            : total system prompt token length (fixed for all requests)
    f1           : fraction of L shared by ALL requests (common prefix)
    f2           : fraction of N requests assigned to family 1 (dominant family)
    num_families : total number of families (each has a unique suffix)

    Layout per prompt:
        [ common_prefix (f1 * L tokens) | family_suffix ((1-f1) * L tokens) ]

    Request distribution:
        family 1  -> floor(f2 * N) requests
        families 2..num_families -> remaining N requests split equally
    """
    common_tokens = int(f1 * L)
    suffix_tokens = L - common_tokens

    # --- request counts per family ---
    f1_count = int(f2 * N)
    remaining = N - f1_count
    other_count = num_families - 1
    if other_count < 1:
        raise ValueError("num_families must be >= 2")
    # distribute remaining equally; last family absorbs rounding difference
    per_other = remaining // other_count
    counts = [f1_count] + [per_other] * (other_count - 1) + [remaining - per_other * (other_count - 1)]

    print("\n[INFO] Prompt composition:")
    print(f"  Total length (L)          : {L} tokens")
    print(f"  Common prefix (f1={f1:.2f}) : {common_tokens} tokens")
    print(f"  Family suffix             : {suffix_tokens} tokens")
    print(f"  Number of families        : {num_families}")
    print(f"  Requests per family       : {counts}")
    print(f"  Total requests            : {sum(counts)}")

    # --- build shared assets ---
    common_prefix = generate_exact_tokens(tokenizer, common_tokens, "COMMON")
    family_suffixes = [
        generate_exact_tokens(tokenizer, suffix_tokens, f"FAM{k+1}")
        for k in range(num_families)
    ]

    # --- build prompts ---
    prompts = []
    req_idx = 0
    for fam_idx, count in enumerate(counts):
        suffix = family_suffixes[fam_idx]
        system_prompt_raw = common_prefix + " " + suffix
        # trim to exactly L tokens
        ids = tokenizer.encode(system_prompt_raw, add_special_tokens=False)[:L]
        system_prompt = tokenizer.decode(ids)

        for i in range(count):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Request {req_idx}: explain AI briefly."},
            ]
            try:
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                full_prompt = system_prompt + "\n\nExplain AI briefly."
            prompts.append(full_prompt)
            req_idx += 1

    random.shuffle(prompts)
    return prompts


def build_dcgmi_filename(args):
    name = (
        f"dcgmi_families_model-{args.model.replace('/', '_')}"
        f"_L-{args.L}"
        f"_f1-{args.f1}"
        f"_f2-{args.f2}"
        f"_fams-{args.num_families}"
        f"_reqs-{args.requests}"
        f"_batch-{args.batch_size}"
        f"_maxnew-{args.max_new_tokens}"
        f".log"
    )
    return name


def run_experiment(llm, prompts, max_new_tokens, iterations, write_dcgmi=False, args=None):
    params = SamplingParams(
        temperature=0.7,
        max_tokens=max_new_tokens,
    )

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
        description="Tiered prefix sharing throughput experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=2000,
        help="System prompt token length (fixed for all requests)",
    )
    parser.add_argument(
        "--f1",
        type=float,
        default=0.667,
        help="Fraction of L shared by ALL requests (common prefix length)",
    )
    parser.add_argument(
        "--f2",
        type=float,
        default=0.5,
        help="Fraction of N requests assigned to family 1 (dominant family)",
    )
    parser.add_argument(
        "--num_families",
        type=int,
        default=6,
        help="Total number of suffix families",
    )
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default="fcfs")
    parser.add_argument(
        "--write_dcgmi", action="store_true", help="Enable DCGMI monitoring and logging"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Building prompts...")
    prompts = build_prompts(
        tokenizer,
        args.L,
        args.requests,
        args.f1,
        args.f2,
        args.num_families,
    )
    print(f"\nGenerated {len(prompts)} prompts")

    print("\nInitializing vLLM...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.75,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=args.batch_size,
        scheduling_policy=args.policy,
    )

    run_experiment(
        llm,
        prompts,
        args.max_new_tokens,
        args.iterations,
        args.write_dcgmi,
        args,
    )


if __name__ == "__main__":
    main()