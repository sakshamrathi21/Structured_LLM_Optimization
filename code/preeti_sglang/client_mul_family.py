import random
import string
import time
import threading
import requests

from system_prompt import queries, system_prompt


# ======================================================
# CONFIGURATION
# ======================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
URL = "http://127.0.0.1:30000/v1/chat/completions"
NUM_FAMILIES = 10
TEMPERATURE = 0.7
MAX_TOKENS = 50


# ======================================================
# PREFIX GENERATION
# ======================================================
def generate_prefixes(n):
    """Generate AA, AB, AC… style prefixes for prompt families."""
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]


# ======================================================
# THREAD TARGET
# ======================================================
def send_request(session, family, prompt, request_id):
    """Send a single prompt request to the SGLang server."""
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"{family} {system_prompt}"},
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    try:
        start = time.time()
        response = session.post(URL, json=data, timeout=30)
        latency = time.time() - start

        if response.status_code == 200:
            if request_id % 25 == 0:
                reply = response.json()["choices"][0]["message"]["content"]
                print(f"[{family}] Req {request_id} ({latency:.2f}s)\n{reply[:150]}...\n")
        else:
            print(f"[{family}] Req {request_id} HTTP {response.status_code}: {response.text[:80]}")
    except Exception as e:
        print(f"[{family}] Req {request_id} Exception: {e}")


# ======================================================
# MAIN EXECUTION
# ======================================================
def main():
    prefixes = generate_prefixes(NUM_FAMILIES)
    prompts = []
    families = []

    for family in prefixes:
        for q in queries:
            prompt = q.strip()
            prompts.append(prompt)
            families.append(family)

    paired = list(zip(prompts, families))
    random.shuffle(paired)
    prompts, families = zip(*paired)

    print(f"Generated {len(prompts)} total prompts across {NUM_FAMILIES} families.")
    print("Starting multithreaded benchmark...\n")

    session = requests.Session()

    start_time = time.time()
    threads = []

    for i, (prompt, family) in enumerate(zip(prompts, families)):
        t = threading.Thread(target=send_request, args=(session, family, prompt, i + 1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    elapsed = end_time - start_time
    throughput = len(prompts) / elapsed

    print("\n===============================================")
    print(f"Completed {len(prompts)} requests in {elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/sec")
    print("===============================================")


if __name__ == "__main__":
    main()
