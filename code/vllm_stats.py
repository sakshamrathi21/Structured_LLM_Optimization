from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

import cProfile
import pstats
import io

def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system_prompt = """The lecture introduces semaphores, a synchronization primitive similar to condition variables, used to coordinate threads. A semaphore maintains a counter, initialized to some value. The down (wait) operation decrements the counter and blocks the thread if it becomes negative, while the up (post) operation increments the counter and wakes a blocked thread. The counter cannot be directly accessed, only modified via these operations.
    Semaphores can be used for mutual exclusion (binary semaphores), functioning like locks. They are also useful for synchronization between threads, e.g., enforcing execution order (T1 before T2) or ensuring two threads finish one step before starting another. Incorrect usage can cause deadlocks, such as when both threads wait on each other.
    The producer-consumer problem is revisited using semaphores. Two semaphores track empty and filled buffer slots, while a binary semaphore (mutex) ensures mutual exclusion on buffer access. Misuse, like holding locks during blocking down calls, can also lead to deadlocks.
    Guidelines emphasize careful initialization, ensuring every down has a corresponding up, and using separate semaphores for signaling and locking. The lecture also covers batched processing with semaphores, showing how request threads and a batch processor can coordinate efficiently.
    Semaphores thus provide flexible yet tricky synchronization, requiring careful design to avoid deadlocks."""
    queries = [
        "How does the down operation on a semaphore synchronize threads?",
        "Why is an extra binary semaphore needed in producer-consumer?",
        "Explain deadlocks caused by semaphores.",
        "What is the difference between counting and binary semaphores?",
        "How do semaphores prevent race conditions in multithreaded programs?",
        "Why do semaphores sometimes lead to busy waiting?",
        "Compare the use of semaphores and mutex locks.",
        "Can semaphores be used for resource allocation? Explain.",
        "What happens if a semaphore is initialized with a negative value?",
        "How do semaphores help solve the reader-writer problem?",
        "What role do semaphores play in barrier synchronization?",
        "Why can debugging semaphore-based solutions be difficult?",
        "How do semaphores help implement bounded-buffer problems?",
        "Why is ordering important when using multiple semaphores?",
        "How do semaphores differ from condition variables?",
        "Can semaphores guarantee fairness among waiting threads?",
        "What is a typical real-world scenario for using semaphores?",
        "How does a semaphore queue waiting processes internally?",
        "Why might semaphores cause priority inversion?",
        "Explain how semaphores can implement critical sections.",
        "What are the limitations of semaphore-based synchronization?",
        "How can misuse of semaphores cause deadlock or starvation?",
        "When should you use semaphores instead of monitors?",
        "How do semaphores interact with interrupt handling in OS kernels?",
    ]


    # Build full prompts with system + user, no prefix caching
    prompts = []
    for q in queries:
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        full_prompt = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(full_prompt)

    llm = LLM(
        model=model_name,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(temperature=0.7, max_tokens=80)
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
    profiler.disable()
    print("Time:", end - start, "s, Outputs:", len(outputs))
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())
    # for q, out in zip(queries, outputs):
    #     print(f"\nUser: {q}\nAssistant: {out.outputs[0].text.strip()}")

if __name__ == "__main__":
    main()
