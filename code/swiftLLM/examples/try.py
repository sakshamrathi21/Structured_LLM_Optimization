import time
import asyncio
import argparse
import numpy as np
from transformers import AutoTokenizer
import swiftllm
import sys
sys.path.append("../")
from system_prompt import system_prompt, queries

async def workload(engine : swiftllm.Engine, prompts, qps, output_len, streaming=False):
    """
    Send requests at ~target QPS. If queries pile up,
    send the entire backlog in a single batch.
    """
    latencies = []
    interval = 1 / qps
    start_time = time.perf_counter()
    i = 0
    backlog = []
    next_time = time.perf_counter()

    while i < len(prompts):
        now = time.perf_counter()
        intervals_passed = int((now - next_time) / interval) + 1
        for _ in range(intervals_passed):
            if i < len(prompts):
                backlog.append(prompts[i])
                i += 1

        if backlog:
            batch_start = time.perf_counter()

            raw_requests = [
                swiftllm.RawRequest(p, output_len) for p in backlog
            ]

            # Single batched call instead of multiple gather
            results = await engine.add_requests_and_wait(raw_requests)
            # results = await engine.add_request_and_wait(raw_requests[0])
            batch_latency = time.perf_counter() - batch_start
            for _ in backlog:
                latencies.append(batch_latency)
            # print(f"Batch of {len(backlog)} requests done in {batch_latency:.3f}s")
            backlog.clear()

        next_time += intervals_passed * interval
        sleep_time = max(0, next_time - time.perf_counter())
        await asyncio.sleep(sleep_time)

    return latencies, time.perf_counter() - start_time


def summarize(latencies):
    latencies, total_time = latencies
    arr = np.array(latencies)
    print(f"Total requests: {len(arr)}")
    print(f"Avg latency: {arr.mean():.3f}s")
    print(f"P95 latency: {np.percentile(arr, 95):.3f}s")
    print(f"Throughput: {len(arr) / total_time:.2f} req/s", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to HF model weights")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--qps", type=int, nargs="+", default=[5], help="QPS values to test")
    parser.add_argument("--output-len", type=int, default=80, help="Output length per request")
    args = parser.parse_args()

    engine_config = swiftllm.EngineConfig(
        model_path=args.model_path,
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.75,
        num_cpu_blocks=8192,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=1024,
        max_batch_size=32,
        max_tokens_in_batch=2048,
    )

    engine = swiftllm.Engine(engine_config)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    await engine.initialize()
    asyncio.create_task(engine.start_all_event_loops())
    prompts = [
        system_prompt + q for q in queries
    ]

    for qps in args.qps:
        print(f"\n=== Running with QPS={qps} ===")
        latencies = await workload(engine, prompts, qps, args.output_len, streaming=args.streaming)
        summarize(latencies)


if __name__ == "__main__":
    asyncio.run(main())
