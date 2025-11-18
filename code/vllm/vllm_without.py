import time

import torch
from transformers import AutoTokenizer

from gpu_mem import start_logging, stop_logging
from system_prompt import queries, system_prompt
from vllm import LLM, SamplingParams


def main():
    model_name = "qwen/qwen2.5-0.5b-instruct"
    # model_name = "lmsys/longchat-13b-16k"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    system_prompt_halved = system_prompt[0 : len(system_prompt)]
    prompts = []
    for q in queries:
        chat_messages = [
            {"role": "system", "content": system_prompt_halved},
            {"role": "user", "content": q},
        ]
        full_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(full_prompt)
    enable_chunked_prefill = True
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=32768,
        enable_prefix_caching=True,
        max_num_seqs=3000,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)

    prompts *= 300
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print(
        "Throughput: ",
        len(outputs) / (end_time - start_time),
        "req/s (enabling chunked prefill:",
        enable_chunked_prefill,
        ")",
    )

    # for interval in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    #     initial_start_time = time.time()
    #     multiplier = int(interval)
    #     if multiplier == 0:
    #         multiplier += 1
    #     all_prompts = prompts * multiplier
    #     for i in range(0, len(all_prompts), interval):
    #         start_time = time.time()
    #         outputs = llm.generate(all_prompts[i:i+interval], sampling_params)
    #         time.sleep(max(0, 1 - time.time() + start_time))
    #     print(f"Throughput for interval ({interval}): ", len(all_prompts) / (time.time() - initial_start_time), "req/s (enabling chunked prefill:", enable_chunked_prefill, ")")


if __name__ == "__main__":
    main()
