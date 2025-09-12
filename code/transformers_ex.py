import time
import asyncio
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from system_prompt import system_prompt, queries


async def workload(model, tokenizer, prompts, qps, device):
    """
    Send requests at ~target QPS, one by one (batch size = 1).
    """
    latencies = []
    interval = 1 / qps
    start_time = time.time()
    i = 0
    next_time = time.time()

    while i < len(prompts):
        now = time.time()
        intervals_passed = int((now - next_time) / interval) + 1

        for _ in range(intervals_passed):
            if i < len(prompts):
                prompt = prompts[i]
                i += 1

                batch_start = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    _ = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.7,
                        do_sample=True,
                    )
                batch_latency = time.time() - batch_start
                latencies.append(batch_latency)

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Build prompts with system + user messages
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

    device = next(model.parameters()).device

    for qps in qpss:
        print(f"\n=== Running with QPS={qps} ===")
        latencies = await workload(model, tokenizer, prompts, qps, device)
        summarize(latencies)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        qps = [int(sys.argv[1])]
    else:
        qps = [5]
    asyncio.run(main(qps))
