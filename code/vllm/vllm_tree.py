#!/usr/bin/env python3
import random
import string
import time
import argparse
import subprocess
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def generate_text_of_length(tokenizer, target_tokens, base_text="A"):
    """Generate text that tokenizes to approximately target_tokens length."""
    # Start with repeated base text
    text = base_text * (target_tokens * 4)  # Rough estimate

    # Iteratively adjust to hit target
    current_tokens = len(tokenizer.encode(text))

    if current_tokens < target_tokens:
        while len(tokenizer.encode(text)) < target_tokens:
            text += base_text
    else:
        # Binary search to find right length
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if len(tokenizer.encode(text[:mid])) <= target_tokens:
                left = mid
            else:
                right = mid - 1
        text = text[:left]

    return text


def build_mode1_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 1: All requests share a system prompt of length 3*L
    """
    shared_prefix = generate_text_of_length(tokenizer, 4 * L, "SHARED_4L ")

    prompts = []
    for i in range(num_requests):
        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": shared_prefix},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = shared_prefix + " Question: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode2_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 2: All share 2*L, then split into two groups each with additional L tokens
    """
    level1_shared = generate_text_of_length(tokenizer, 2 * L, "LEVEL1_2L ")
    level2_group1 = generate_text_of_length(tokenizer, 2*L, "GROUP1_L ")
    level2_group2 = generate_text_of_length(tokenizer, 2*L, "GROUP2_L ")

    prompts = []
    half = num_requests // 2

    for i in range(num_requests):
        if i < half:
            system_prompt = level1_shared + " " + level2_group1
        else:
            system_prompt = level1_shared + " " + level2_group2

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode3_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 3: All share L, then split into two groups each with additional 2*L tokens
    """
    level1_shared = generate_text_of_length(tokenizer, L, "LEVEL1_L ")
    level2_group1 = generate_text_of_length(tokenizer, 2 * L, "GROUP1_2L ")
    level2_group2 = generate_text_of_length(tokenizer, 2 * L, "GROUP2_2L ")

    prompts = []
    half = num_requests // 2

    for i in range(num_requests):
        if i < half:
            system_prompt = level1_shared + " " + level2_group1
        else:
            system_prompt = level1_shared + " " + level2_group2

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode4_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 4: Three-level tree
    - Level 1: All share L
    - Level 2: Two groups, each shares additional L (total 2L)
    - Level 3: Four groups, each shares additional L (total 3L)
    """
    level1_shared = generate_text_of_length(tokenizer, L, "LEVEL1_L ")

    # Level 2: Two branches
    level2_branch1 = generate_text_of_length(tokenizer, L, "LEVEL2_BRANCH1_L ")
    level2_branch2 = generate_text_of_length(tokenizer, L, "LEVEL2_BRANCH2_L ")

    # Level 3: Four leaves
    level3_leaf1 = generate_text_of_length(tokenizer, L, "LEVEL3_LEAF1_L ")
    level3_leaf2 = generate_text_of_length(tokenizer, L, "LEVEL3_LEAF2_L ")
    level3_leaf3 = generate_text_of_length(tokenizer, L, "LEVEL3_LEAF3_L ")
    level3_leaf4 = generate_text_of_length(tokenizer, L, "LEVEL3_LEAF4_L ")

    prompts = []
    quarter = num_requests // 4

    for i in range(num_requests):
        if i < quarter:
            # Branch 1, Leaf 1
            system_prompt = level1_shared + " " + level2_branch1 + " " + level3_leaf1
        elif i < 2 * quarter:
            # Branch 1, Leaf 2
            system_prompt = level1_shared + " " + level2_branch1 + " " + level3_leaf2
        elif i < 3 * quarter:
            # Branch 2, Leaf 3
            system_prompt = level1_shared + " " + level2_branch2 + " " + level3_leaf3
        else:
            # Branch 2, Leaf 4
            system_prompt = level1_shared + " " + level2_branch2 + " " + level3_leaf4

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode5_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 5: Two separate groups, each with their own 3*L shared context
    """
    group1_shared = generate_text_of_length(tokenizer, 4 * L, "GROUP1_3L ")
    group2_shared = generate_text_of_length(tokenizer, 4 * L, "GROUP2_3L ")

    prompts = []
    half = num_requests // 2

    for i in range(num_requests):
        if i < half:
            system_prompt = group1_shared
        else:
            system_prompt = group2_shared

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts


#!/usr/bin/env python3
"""
Extended modes for radix tree experiments
Modes 6-22 for testing prefetcher and locality hypotheses
"""
import random
from transformers import AutoTokenizer


def generate_token_sequence(tokenizer, target_tokens, prefix_text="PREFIX"):
    """Generate a sequence of exactly target_tokens."""
    pattern = f"{prefix_text} "
    pattern_tokens = len(tokenizer.encode(pattern, add_special_tokens=False))

    repetitions = (target_tokens // pattern_tokens) + 1
    text = pattern * repetitions

    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    exact_text = tokenizer.decode(token_ids)

    return exact_text


def build_mode6_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 6: Gradual Divergence - 2.5*L shared, then 10 groups with 0.5*L each
    Tests: Does high initial sharing + many small branches hurt less?
    """
    shared_prefix = generate_token_sequence(tokenizer, int(2.5 * L), "SHARED_2.5L")

    groups_per_request = num_requests // 10
    group_suffixes = []
    for i in range(10):
        suffix = generate_token_sequence(tokenizer, int(0.5 * L), f"G{i}")
        group_suffixes.append(suffix)

    prompts = []
    for i in range(num_requests):
        group_idx = min(i // groups_per_request, 9)
        system_prompt = shared_prefix + " " + group_suffixes[group_idx]

        # Adjust to exactly 3*L
        tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
        tokens = tokens[: 3 * L] if len(tokens) > 3 * L else tokens
        while len(tokens) < 3 * L:
            tokens.append(tokenizer.encode(" ", add_special_tokens=False)[0])
        system_prompt = tokenizer.decode(tokens)

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = system_prompt + "\n\nQuestion: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode7_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 7: Front-Heavy - 2.8*L shared, then 50 groups with 0.2*L each
    Tests: Extreme front-loading of shared content
    """
    shared_prefix = generate_token_sequence(tokenizer, int(2.8 * L), "SHARED_2.8L")

    groups_per_request = num_requests // 50
    group_suffixes = []
    for i in range(50):
        suffix = generate_token_sequence(tokenizer, int(0.2 * L), f"G{i:02d}")
        group_suffixes.append(suffix)

    prompts = []
    for i in range(num_requests):
        group_idx = min(i // groups_per_request, 49)
        system_prompt = shared_prefix + " " + group_suffixes[group_idx]

        tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
        tokens = tokens[: 3 * L] if len(tokens) > 3 * L else tokens
        while len(tokens) < 3 * L:
            tokens.append(tokenizer.encode(" ", add_special_tokens=False)[0])
        system_prompt = tokenizer.decode(tokens)

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = system_prompt + "\n\nQuestion: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode8_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 8: Back-Heavy - 0.2*L shared, then 50 groups with 2.8*L each
    Tests: Minimal prefetch warmup - worst case
    """
    shared_prefix = generate_token_sequence(tokenizer, int(0.2 * L), "SHARED_0.2L")

    groups_per_request = num_requests // 50
    group_suffixes = []
    for i in range(50):
        suffix = generate_token_sequence(tokenizer, int(2.8 * L), f"LARGE_G{i:02d}")
        group_suffixes.append(suffix)

    prompts = []
    for i in range(num_requests):
        group_idx = min(i // groups_per_request, 49)
        system_prompt = shared_prefix + " " + group_suffixes[group_idx]

        tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
        tokens = tokens[: 3 * L] if len(tokens) > 3 * L else tokens
        while len(tokens) < 3 * L:
            tokens.append(tokenizer.encode(" ", add_special_tokens=False)[0])
        system_prompt = tokenizer.decode(tokens)

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = system_prompt + "\n\nQuestion: " + query

        prompts.append(full_prompt)

    return prompts

def build_mode9_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 3 (Revised):
    - Level 1: All share L
    - Level 2: Two branches, each with 3L
    """

    # Level 1 shared by all
    level1_shared = generate_text_of_length(tokenizer, 1 * L, "LEVEL1_L ")

    # Two 3L branches
    branch1 = generate_text_of_length(tokenizer, 3 * L, "BRANCH1_3L ")
    branch2 = generate_text_of_length(tokenizer, 3 * L, "BRANCH2_3L ")

    prompts = []
    half = num_requests // 2

    for i in range(num_requests):
        if i < half:
            system_prompt = level1_shared + " " + branch1
        else:
            system_prompt = level1_shared + " " + branch2

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts

def build_mode10_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 4 (Revised):
    - Level 1: All share L
    - Level 2: Two branches, each L
    - Level 3: Four leaves (2 per branch), each 2L
    """

    # Level 1
    level1_shared = generate_text_of_length(tokenizer, L, "LEVEL1_L ")

    # Level 2 (2 branches of L each)
    branch1_L2 = generate_text_of_length(tokenizer, L, "L2_BRANCH1_L ")
    branch2_L2 = generate_text_of_length(tokenizer, L, "L2_BRANCH2_L ")

    # Level 3 (4 leaves of 2L each)
    leaf1 = generate_text_of_length(tokenizer, 2 * L, "L3_LEAF1_2L ")
    leaf2 = generate_text_of_length(tokenizer, 2 * L, "L3_LEAF2_2L ")
    leaf3 = generate_text_of_length(tokenizer, 2 * L, "L3_LEAF3_2L ")
    leaf4 = generate_text_of_length(tokenizer, 2 * L, "L3_LEAF4_2L ")

    prompts = []
    quarter = num_requests // 4

    for i in range(num_requests):
        if i < quarter:
            system_prompt = level1_shared + " " + branch1_L2 + " " + leaf1
        elif i < 2 * quarter:
            system_prompt = level1_shared + " " + branch1_L2 + " " + leaf2
        elif i < 3 * quarter:
            system_prompt = level1_shared + " " + branch2_L2 + " " + leaf3
        else:
            system_prompt = level1_shared + " " + branch2_L2 + " " + leaf4

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts


def build_mode11_prompts(tokenizer, L, num_requests, query_base="What is AI?"):
    """
    Mode 5 (Revised):
    - Level 1: All share L
    - Level 2: 2 branches (L each)
    - Level 3: 4 branches (L each)
    - Level 4: 8 branches (L each)
    """

    # Level 1
    level1 = generate_text_of_length(tokenizer, L, "LEVEL1_L ")

    # Level 2 (2 branches)
    level2 = [
        generate_text_of_length(tokenizer, L, "L2_BRANCH0_L "),
        generate_text_of_length(tokenizer, L, "L2_BRANCH1_L "),
    ]

    # Level 3 (4 branches)
    level3 = [
        generate_text_of_length(tokenizer, L, f"L3_BRANCH{i}_L ")
        for i in range(4)
    ]

    # Level 4 (8 branches)
    level4 = [
        generate_text_of_length(tokenizer, L, f"L4_BRANCH{i}_L ")
        for i in range(8)
    ]

    prompts = []
    eighth = num_requests // 8

    for i in range(num_requests):
        leaf_idx = i // eighth  # 0–7

        l2_idx = leaf_idx // 4       # 0–1
        l3_idx = leaf_idx // 2       # 0–3
        l4_idx = leaf_idx            # 0–7

        system_prompt = (
            level1 + " " +
            level2[l2_idx] + " " +
            level3[l3_idx] + " " +
            level4[l4_idx]
        )

        query = f"{query_base} Request {i}."
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            full_prompt = system_prompt + " Question: " + query

        prompts.append(full_prompt)

    return prompts

def build_dcgmi_filename(args):
    name = (
        f"dcgmi_radix_model-{args.model_name.replace('/', '_')}"
        f"_mode-{args.mode}"
        f"_L-{args.L}"
        f"_total-{args.total_requests}"
        f"_batch-{args.max_batch_size}"
        f"_rps-{args.req_per_sec}"
        f"_maxnew-{args.max_new_tokens}"
        f".log"
    )
    return name


def main():
    parser = argparse.ArgumentParser(description="Radix Tree Sharing Experiment")
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--mode", type=int, required=True, help="Sharing mode (1-5)")
    parser.add_argument(
        "--L", type=int, default=1300, help="Base system prompt length in tokens"
    )
    parser.add_argument(
        "--total_requests", type=int, default=10000, help="Total number of requests"
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=500, help="Maximum batch size"
    )
    parser.add_argument(
        "--req_per_sec",
        type=float,
        default=None,
        help="Requests per second (Poisson arrival), None for batch mode",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--write_dcgmi", action="store_true", help="Enable DCGMI monitoring and logging"
    )
    parser.add_argument(
        "--write_output", action="store_true", help="Write results to output file"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="vllm:fcfs",
        help="Engine and scheduling policy in format 'engine:policy' (e.g., 'vllm:fcfs', 'sglang:lpm')",
    )
    parser.add_argument(
        "--query_base", type=str, default="What is AI?", help="Base query text"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle requests (keep deterministic order)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )

    args = parser.parse_args()

    # Parse engine and policy from --policy argument
    policy_parts = args.policy.split(":", 1)
    if len(policy_parts) == 2:
        engine_name = policy_parts[0].lower()
        scheduling_policy = policy_parts[1]
    else:
        # Backward compatibility: no engine prefix means vllm
        engine_name = "vllm"
        scheduling_policy = args.policy

    print(f"[INFO] Engine: {engine_name}, Scheduling Policy: {scheduling_policy}")

    # Set random seed for reproducibility
    random.seed(args.seed)

    print(f"[INFO] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build prompts based on mode
    print(f"[INFO] Building prompts for Mode {args.mode}...")
    if args.mode == 1:
        prompts = build_mode1_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 2:
        prompts = build_mode2_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 3:
        prompts = build_mode3_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 4:
        prompts = build_mode4_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 5:
        prompts = build_mode5_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 6:
        prompts = build_mode6_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 7:
        prompts = build_mode7_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 8:
        prompts = build_mode5_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 9:
        prompts = build_mode9_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 10:
        prompts = build_mode10_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )
    elif args.mode == 11:
        prompts = build_mode11_prompts(
            tokenizer, args.L, args.total_requests, args.query_base
        )

    print(f"[INFO] Generated {len(prompts)} prompts")

    # Shuffle requests to simulate random arrival order
    if not args.no_shuffle:
        print(f"[INFO] Shuffling requests with seed={args.seed}...")
        random.shuffle(prompts)
        print(f"[INFO] Requests shuffled")
    else:
        print(f"[INFO] Keeping deterministic order (no shuffle)")

    # Sample a few prompts to verify token lengths
    print(f"\n[INFO] Verifying prompt lengths (sampling first 3):")
    for i in range(min(3, len(prompts))):
        tokens = tokenizer.encode(prompts[i])
        print(f"  Prompt {i}: {len(tokens)} tokens")

    # -------------------------------------------------------------------------
    # Initialize engine
    # -------------------------------------------------------------------------
    if engine_name == "sglang":
        import sglang as sgl

        print(f"\n[INFO] Initializing SGLang engine with policy={scheduling_policy}...")
        llm = sgl.Engine(
            model_path=args.model_name,
            mem_fraction_static=0.98,
            max_running_requests=args.max_batch_size,
            schedule_policy=scheduling_policy,
            enable_mixed_chunk=True,
            max_total_tokens=65536
        )
        sampling_params = {
            "temperature": 0.7,
            "max_new_tokens": args.max_new_tokens,
        }

    elif engine_name == "vllm":
        from vllm import LLM, SamplingParams

        print(f"\n[INFO] Initializing vLLM engine with policy={scheduling_policy}...")
        llm = LLM(
            model=args.model_name,
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            max_num_seqs=args.max_batch_size,
            scheduling_policy=scheduling_policy,
            max_num_batched_tokens=32768,
        )
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=args.max_new_tokens
        )

    else:
        raise ValueError(
            f"Unknown engine '{engine_name}'. Expected 'vllm' or 'sglang'. "
            f"Use --policy engine:policy (e.g., 'vllm:fcfs' or 'sglang:lpm')."
        )

    # -------------------------------------------------------------------------
    # Define a unified generate function
    # -------------------------------------------------------------------------
    def run_generate():
        if engine_name == "sglang":
            llm.generate(
                prompts,
                sampling_params,
                poisson_rate=args.req_per_sec,
            )
        else:  # vllm
            if args.req_per_sec is not None:
                llm.generate(prompts, sampling_params, poisson_lambda=args.req_per_sec)
            else:
                llm.generate(prompts, sampling_params)

    # -------------------------------------------------------------------------
    # Run with or without DCGMI monitoring
    # -------------------------------------------------------------------------
    initial_start_time = time.time()

    if args.write_dcgmi:
        dcgmi_file = build_dcgmi_filename(args)
        print(f"[INFO] Logging DCGMI output to: {dcgmi_file}")

        dcgmi_cmd = ["sudo", "dcgmi", "dmon", "-e", "1002,1003,1004,1005", "-d", "1"]

        with open(dcgmi_file, "w") as f:
            dcgmi_proc = subprocess.Popen(dcgmi_cmd, stdout=f, stderr=subprocess.STDOUT)
            try:
                run_generate()
            finally:
                print("[INFO] Stopping DCGMI monitoring...")
                dcgmi_proc.terminate()
                try:
                    dcgmi_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    dcgmi_proc.kill()
    else:
        run_generate()

    end_time = time.time()

    throughput = len(prompts) / (end_time - initial_start_time)
    total_time = end_time - initial_start_time

    result_str = (
        f"Radix Tree Experiment - Engine={engine_name}, Mode {args.mode}, "
        f"L={args.L}, Total Requests={args.total_requests}, "
        f"Batch Size={args.max_batch_size}, "
        f"RPS={args.req_per_sec}, Max New Tokens={args.max_new_tokens}, "
        f"Model={args.model_name}, Policy={args.policy} | "
        f"Throughput: {throughput:.2f} req/sec, Total Time: {total_time:.2f}s\n"
    )

    print(f"\n{'='*80}")
    print(f"[RESULTS] {result_str}")
    print(f"{'='*80}\n")

    if args.write_output:
        output_file = "radix_tree_experiment_results.txt"
        with open(output_file, "a") as f:
            f.write(result_str)
        print(f"[INFO] Results appended to {output_file}")

if __name__ == "__main__":
    main()
