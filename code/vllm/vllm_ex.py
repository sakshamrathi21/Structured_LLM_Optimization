import time
import asyncio
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
from system_prompt import system_prompt, queries


async def workload(llm, prompts, sampling_params, qps):
    """
    Send requests at ~target QPS. If queries pile up,
    batch them in one llm.generate() call.
    """
    latencies = []
    interval = 1 / qps
    start_time = time.time()
    i = 0
    backlog = []
    next_time = time.time()

    while i < len(prompts):
        now = time.time()
        intervals_passed = int((now - next_time) / interval) + 1
        for _ in range(intervals_passed):
            if i < len(prompts):
                backlog.append(prompts[i])
                i += 1

        if backlog:
            batch_start = time.time()
            _ = llm.generate(list(backlog), sampling_params)
            batch_latency = time.time() - batch_start
            for _ in backlog:
                latencies.append(batch_latency)
            backlog.clear()

        next_time += intervals_passed * interval
        sleep_time = max(0, next_time - time.time())
        await asyncio.sleep(sleep_time)

    return latencies, time.time() - start_time


def summarize(latencies):
    latencies, total_time = latencies
    arr = np.array(latencies)
    print(f"Total requests: {len(arr)}")
    print(f"Avg latency: {arr.mean():.3f}s")
    print(f"P95 latency: {np.percentile(arr, 95):.3f}s")
    print(f"Throughput: {len(arr) / total_time:.2f} req/s", flush=True)


async def main(qpss):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.75,
        max_num_batched_tokens=2048,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # prefix_messages = [{"role": "system", "content": system_prompt}]
    # prefix_prompt = tokenizer.apply_chat_template(
    #     prefix_messages,
    #     tokenize=False,
    #     add_generation_prompt=False,
    # )
    # print("hello", type(prefix_prompt))
    # llm.generate([{"prompt": prefix_prompt, "stream": False, "is_prompt_prefix": True}])
    prompts = []
    for q in queries:
        user_only = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(user_only)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=80)

    for qps in qpss:
        print(f"\n=== Running with QPS={qps} ===")
        latencies = await workload(llm, prompts, sampling_params, qps)
        summarize(latencies)


if __name__ == "__main__":
    # take the command line arg
    import sys

    if len(sys.argv) > 1:
        qps = [int(sys.argv[1])]
    else:
        qps = 5
    import asyncio

    asyncio.run(main(qps))
