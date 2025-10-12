import random
import string
import time

from transformers import AutoTokenizer

from gpu_mem import start_logging, stop_logging
from system_prompt import queries, system_prompt
from vllm import LLM, SamplingParams


def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


def main():
    model_name = "qwen/qwen2.5-0.5b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prefixes = generate_prefixes(100)

    prompts = []
    for prefix in prefixes:
        modified_system_prompt = prefix + " " + system_prompt
        # modified_system_prompt = system_prompt
        for q in queries:
            chat_messages = [
                {"role": "system", "content": modified_system_prompt},
                {"role": "user", "content": q},
            ]
            full_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(full_prompt)

    print(f"Generated {len(prompts)} prompts in total.")

    random.shuffle(prompts)

    enable_chunked_prefill = True
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.75,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        max_num_seqs=1000,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    print(
        "Throughput:",
        len(outputs) / (end_time - start_time),
        "req/s (enabling chunked prefill:",
        enable_chunked_prefill,
        ")",
    )


if __name__ == "__main__":
    main()
