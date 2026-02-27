import time
import argparse
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_exact_tokens(tokenizer, target_tokens, tag):
    if target_tokens == 0:
        return ""
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


def build_prompts_by_family(tokenizer, L, num_families, requests_per_family):
    print("\n[INFO] Prompt composition:")
    print(f"  Total length (L)          : {L} tokens")
    print(f"  Common prefix (f1=0.00)   : 0 tokens")
    print(f"  Family suffix             : {L} tokens")
    print(f"  Number of families        : {num_families}")
    print(f"  Requests per family       : {requests_per_family}")
    print(f"  Total requests            : {num_families * requests_per_family}")

    family_prompts = {}
    for fam_idx in range(num_families):
        tag = f"FAM{fam_idx+1}"
        family_prefix = generate_exact_tokens(tokenizer, L, tag)
        ids = tokenizer.encode(family_prefix, add_special_tokens=False)[:L]
        system_prompt = tokenizer.decode(ids)

        prompts = []
        for i in range(requests_per_family):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Request {fam_idx*requests_per_family+i}: explain AI briefly."},
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
        family_prompts[fam_idx] = prompts

    return family_prompts


def make_single_family_batches(family_prompts, batch_size, num_families, requests_per_family):
    """
    Split each family's prompts into chunks of `batch_size`.
    Returns a list of batches, each containing only one family's prompts.
    Each family contributes requests_per_family / batch_size batches.
    Batches are then shuffled in order (but contents within each batch shuffled).
    """
    batches = []
    for fam_idx in range(num_families):
        fam_p = family_prompts[fam_idx].copy()
        random.shuffle(fam_p)
        for start in range(0, len(fam_p), batch_size):
            chunk = fam_p[start:start + batch_size]
            if chunk:
                batches.append(chunk)
    # shuffle batch order so families are interleaved across time
    random.shuffle(batches)
    return batches


def run_experiment_setting(llm, batches, params, setting_name, iterations):
    total_requests = sum(len(b) for b in batches)
    print(f"\n{'='*60}")
    print(f"Setting: {setting_name}")
    print(f"  Num batches     : {len(batches)}")
    print(f"  Batch sizes     : {list(set(len(b) for b in batches))}")
    print(f"  Total requests  : {total_requests}")
    print(f"{'='*60}")

    all_times = []
    for it in range(iterations):
        iter_start = time.time()
        for batch in batches:
            llm.generate(batch, params)
        iter_elapsed = time.time() - iter_start
        throughput = total_requests / iter_elapsed
        print(f"  [ITER {it+1}] time = {iter_elapsed:.3f}s | throughput = {throughput:.2f} req/s")
        all_times.append(iter_elapsed)

    avg_time = sum(all_times) / len(all_times)
    avg_throughput = total_requests / avg_time
    print(f"  --> AVG time = {avg_time:.3f}s | AVG throughput = {avg_throughput:.2f} req/s")
    return avg_time, avg_throughput


def main():
    parser = argparse.ArgumentParser(
        description="Batch size vs family grouping throughput experiment"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--L", type=int, default=10000, help="System prompt token length")
    parser.add_argument("--num_families", type=int, default=8)
    parser.add_argument("--requests_per_family", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", type=str, default="fcfs")
    args = parser.parse_args()

    random.seed(args.seed)

    total_requests = args.num_families * args.requests_per_family  # 800
    max_batch_size = total_requests  # 800

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Building prompts by family...")
    family_prompts = build_prompts_by_family(
        tokenizer, args.L, args.num_families, args.requests_per_family
    )

    print("\nInitializing vLLM...")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_seqs=max_batch_size,
        scheduling_policy=args.policy,
    )

    params = SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens)

    # --- Pre-warm KV cache ---
    print("\n[Warming up KV cache...]")
    all_prompts_flat = [p for fam in family_prompts.values() for p in fam]
    llm.generate(all_prompts_flat, params)
    print("[Warm-up complete.]\n")

    results = []

    # -------------------------------------------------------
    # Setting 1: 1 batch of 800 (all 8 families mixed)
    # -------------------------------------------------------
    batch_all = all_prompts_flat.copy()
    random.shuffle(batch_all)
    avg_t, avg_tp = run_experiment_setting(
        llm, [batch_all], params,
        "1 batch x 800 req (all 8 families mixed)",
        args.iterations
    )
    results.append(("1x800  (8 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Setting 2: 2 batches of 400, each with 4 families
    # -------------------------------------------------------
    batch1 = [p for fam_idx in range(0, 4) for p in family_prompts[fam_idx]]
    batch2 = [p for fam_idx in range(4, 8) for p in family_prompts[fam_idx]]
    random.shuffle(batch1); random.shuffle(batch2)
    avg_t, avg_tp = run_experiment_setting(
        llm, [batch1, batch2], params,
        "2 batches x 400 req (4 families each)",
        args.iterations
    )
    results.append(("2x400  (4 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Setting 3: 4 batches of 200, each with 2 families
    # -------------------------------------------------------
    batches = []
    for start in range(0, args.num_families, 2):
        b = [p for fam_idx in range(start, start + 2) for p in family_prompts[fam_idx]]
        random.shuffle(b)
        batches.append(b)
    avg_t, avg_tp = run_experiment_setting(
        llm, batches, params,
        "4 batches x 200 req (2 families each)",
        args.iterations
    )
    results.append(("4x200  (2 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Setting 4: 8 batches of 100, each with 1 family
    # -------------------------------------------------------
    batches = []
    for fam_idx in range(args.num_families):
        b = family_prompts[fam_idx].copy()
        random.shuffle(b)
        batches.append(b)
    random.shuffle(batches)
    avg_t, avg_tp = run_experiment_setting(
        llm, batches, params,
        "8 batches x 100 req (1 family each)",
        args.iterations
    )
    results.append(("8x100  (1 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Setting 5: 16 batches of 50, each with 1 family
    # Each family (100 req) -> 2 batches of 50
    # Total: 8 families x 2 = 16 batches
    # -------------------------------------------------------
    batches = make_single_family_batches(
        family_prompts, batch_size=50,
        num_families=args.num_families,
        requests_per_family=args.requests_per_family
    )
    avg_t, avg_tp = run_experiment_setting(
        llm, batches, params,
        "16 batches x 50 req (1 family each)",
        args.iterations
    )
    results.append(("16x50  (1 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Setting 6: 32 batches of 25, each with 1 family
    # Each family (100 req) -> 4 batches of 25
    # Total: 8 families x 4 = 32 batches
    # -------------------------------------------------------
    batches = make_single_family_batches(
        family_prompts, batch_size=25,
        num_families=args.num_families,
        requests_per_family=args.requests_per_family
    )
    avg_t, avg_tp = run_experiment_setting(
        llm, batches, params,
        "32 batches x 25 req (1 family each)",
        args.iterations
    )
    results.append(("32x25  (1 fam/batch)", avg_t, avg_tp))

    # -------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"{'Setting':<25} {'Avg Time (s)':>14} {'Avg Throughput (req/s)':>22}")
    print("-"*60)
    for name, t, tp in results:
        print(f"{name:<25} {t:>14.3f} {tp:>22.2f}")
    print("="*60)


if __name__ == "__main__":
    main()