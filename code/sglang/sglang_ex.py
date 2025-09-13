import time
import numpy as np

from sglang import assistant, function, gen, system, user
from sglang import RuntimeEndpoint, set_default_backend
from sglang.utils import print_highlight

from system_prompt import system_prompt, queries

port = 30000
set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))


@function
def text_qa(s, question):
    s += system(system_prompt)
    s += user(question)
    s += assistant(gen("answer", stop="\n"))


def run_workload(qps, queries):
    """
    Run queries at ~target QPS.
    If queries pile up, send them all at once in one run_batch().
    """
    latencies = []
    states = []
    interval = 1 / qps
    start_time = time.time()
    next_time = start_time
    i = 0
    backlog = []

    while i < len(queries) or backlog:
        now = time.time()
        # how many slots have passed since last scheduling tick
        intervals_passed = int((now - next_time) / interval) + 1
        for _ in range(intervals_passed):
            if i < len(queries):
                backlog.append({"question": queries[i]})
                i += 1

        if backlog:
            batch_start = time.time()
            batch_states = text_qa.run_batch(backlog, progress_bar=False)
            batch_latency = time.time() - batch_start

            for _ in backlog:
                latencies.append(batch_latency)
            states.extend(batch_states)
            backlog.clear()

        next_time += intervals_passed * interval
        sleep_time = max(0, next_time - time.time())
        if sleep_time > 0:
            time.sleep(sleep_time)

    total_time = time.time() - start_time
    return states, latencies, total_time


def summarize(latencies, total_time):
    arr = np.array(latencies)
    print(f"Total requests: {len(arr)}")
    print(f"Avg latency: {arr.mean():.3f}s")
    print(f"P95 latency: {np.percentile(arr, 95):.3f}s")
    print(f"Throughput: {len(arr) / total_time:.2f} req/s", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        qpss = [int(arg) for arg in sys.argv[1:]]
    else:
        qpss = [5]

    for qps in qpss:
        print(f"\n=== Running with QPS={qps} ===")
        states, latencies, total_time = run_workload(qps, queries)
        summarize(latencies, total_time)

        # uncomment if you want to see actual answers
        # for i, state in enumerate(states):
        #     print_highlight(f"Answer {i+1}: {state['answer']}")
