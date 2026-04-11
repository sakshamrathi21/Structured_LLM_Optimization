#!/home/saksham/CS496/code/vllm/.venv/bin/python
import random
import string
import time
import argparse
import subprocess
from transformers import AutoTokenizer
from system_prompt import queries, system_prompt as sys_prompt
import sglang as sgl

TOTAL_QUERIES_IN_DB = 100


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def build_dcgmi_filename(args):
    name = (
        f"dcgmi_sglang_model-{args.model_name.replace('/', '_')}"
        f"_families-{args.num_families}"
        f"_rps-{args.req_per_sec}"
        f"_batch-{args.max_batch_size}"
        f"_rand-{args.randomness}"
        f"_sptokens-{args.sys_prompt_tokens}"
        f"_policy-{args.policy}"
        f"_maxnew-{args.max_new_tokens}"
        f".log"
    )
    return name


def truncate_or_repeat_to_token_length(text, target_tokens, tokenizer):
    """Truncate or repeat `text` so that it encodes to approximately `target_tokens` tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) >= target_tokens:
        tokens = tokens[:target_tokens]
    else:
        repeats = (target_tokens // len(tokens)) + 1
        tokens = (tokens * repeats)[:target_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--num_families", type=int, default=5)
    parser.add_argument("--req_per_sec", type=int, default=None)
    parser.add_argument("--max_batch_size", type=int, default=500)
    parser.add_argument("--randomness", type=int, default=1)
    parser.add_argument(
        "--sys_prompt_tokens",
        type=int,
        default=500,
        help="Target system prompt length in tokens (prefix excluded).",
    )
    parser.add_argument(
        "--num_requests",
        type=int,
        default=500,
        help=(
            "Total number of requests to generate for the experiment. "
            f"Assumes {TOTAL_QUERIES_IN_DB} queries available in the DB. "
            "Requests are distributed evenly across families; each family cycles "
            "through the query pool as many times as needed."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--write_output", action="store_true")
    parser.add_argument("--policy", type=str, default="dfs-weight")
    parser.add_argument(
        "--write_dcgmi", action="store_true", help="Enable DCGMI monitoring and logging"
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build system prompt truncated/repeated to the requested token length
    system_prompt = truncate_or_repeat_to_token_length(
        sys_prompt, args.sys_prompt_tokens, tokenizer
    )
    actual_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    print(f"System prompt length: {actual_tokens} tokens (target: {args.sys_prompt_tokens})")

    prefixes = generate_prefixes(args.num_families)

    # Distribute num_requests evenly across families, cycling through the query pool
    requests_per_family = args.num_requests // args.num_families
    remainder = args.num_requests % args.num_families  # first `remainder` families get +1

    extended_queries = (queries * ((TOTAL_QUERIES_IN_DB // len(queries)) + 1))[:TOTAL_QUERIES_IN_DB]

    prompts = []
    families = []

    for idx, prefix in enumerate(prefixes):
        n = requests_per_family + (1 if idx < remainder else 0)
        family_queries = (extended_queries * ((n // TOTAL_QUERIES_IN_DB) + 1))[:n]

        for q in family_queries:
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

    # Shuffle in chunks of size `randomness`
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
        max_prefill_tokens=1048576,
        chunked_prefill_size=131072,
        enable_mixed_chunk=True
    )

    sampling_params = {
        "temperature": 0.7,
        "max_new_tokens": args.max_new_tokens,
    }

    # Run benchmark
    start_time = time.time()

    # ----- CONDITIONAL DCGMI WRAPPING -----

    if args.write_dcgmi:
        dcgmi_file = build_dcgmi_filename(args)
        print(f"[INFO] Logging DCGMI output to: {dcgmi_file}")

        dcgmi_cmd = ["sudo", "dcgmi", "dmon", "-e", "1002,1003,1004,1005", "-d", "1"]

        with open(dcgmi_file, "w") as f:
            dcgmi_proc = subprocess.Popen(dcgmi_cmd, stdout=f, stderr=subprocess.STDOUT)

            try:
                llm.generate(
                    prompts,
                    sampling_params,
                    poisson_rate=args.req_per_sec,
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
            sampling_params,
            poisson_rate=args.req_per_sec,
        )

    end_time = time.time()

    throughput = len(prompts) / (end_time - start_time)
    print("sglang: Throughput:", throughput*args.max_new_tokens, "toks/s")

    if args.write_output:
        with open("paper_eval_results.csv", "a") as f:
            f.write(
                f"sglang,{args.num_families},{args.req_per_sec},{args.max_batch_size},"
                f"{args.randomness},{args.sys_prompt_tokens},{args.model_name},"
                f"{args.max_new_tokens},{args.num_requests},"
                f"{len(prompts)},{throughput*args.max_new_tokens},{args.policy}\n"
            )


if __name__ == "__main__":
    main()