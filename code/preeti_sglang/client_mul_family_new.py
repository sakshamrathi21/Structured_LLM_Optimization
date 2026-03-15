import random
import string
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from system_prompt import queries, system_prompt

# ======================================
# --- Configuration ---
# ======================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
URL = "http://127.0.0.1:30000/v1/chat/completions"
NUM_FAMILIES = 20
NUM_PROMPTS_PER_FAMILY = 100
TEMPERATURE = 0.7
MAX_TOKENS = 50
MAX_CONNECTIONS = 500  # max TCP connections to reuse
MAX_WORKERS = 500       # max threads in ThreadPoolExecutor

# ======================================
# --- Prepare Session Pool ---
# ======================================
session_pool = [requests.Session() for _ in range(MAX_CONNECTIONS)]
session_lock = None
next_session_idx = 0

def get_session():
    """Round-robin session selection to reuse TCP connections."""
    global next_session_idx
    global session_lock
    if session_lock is None:
        import threading
        session_lock = threading.Lock()
    with session_lock:
        session = session_pool[next_session_idx]
        next_session_idx = (next_session_idx + 1) % MAX_CONNECTIONS
    return session

# ======================================
# --- Utility: Generate Family Prefixes ---
# ======================================
def generate_prefixes(n):
    letters = string.ascii_uppercase
    prefixes = []
    for a in letters:
        for b in letters:
            prefixes.append(a + b)
    return prefixes[:n]

# ======================================
# --- Warmup Phase (System Prompt Only) ---
# ======================================
def warmup_prefix(family):
    """Send system prompt to warm up KV cache for this family."""
    print(f"[Warmup] Sending system prompt for family {family}...")
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": f"{family} {system_prompt}"}],
        "temperature": 0.0,
        "max_tokens": 10
    }
    session = get_session()
    try:
        response = session.post(URL, json=data)
        if response.status_code == 200:
            print(f"[Warmup] Family {family} cached successfully.")
        else:
            print(f"[Warmup] Family {family} failed ({response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"[Warmup] Connection error for {family}: {e}")

# ======================================
# --- Send Request Function (User Prompts Only) ---
# ======================================
def send_request(family, prompt, request_id):
    """Send only the user prompt — system prompt already cached."""
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    session = get_session()
    try:
        response = session.post(URL, json=data)
        # optional: print occasionally
        # if request_id % 50 == 0:
        #     print(f"[Req {request_id}] Sent for {family}")
    except requests.exceptions.RequestException as e:
        print(f"[ConnError {request_id}] {e}")

# ======================================
# --- Prompt Generation ---
# ======================================
def generate_prompts_from_families(families, num_per_family):
    all_prompts = []
    for family in families:
        for _ in range(num_per_family):
            prompt = random.choice(queries)
            all_prompts.append((family, prompt))
    # random.shuffle(all_prompts) # Uncomment if you want to shuffle the prompts *******************************  
    # for p in all_prompts:
    #     print(p)
    return all_prompts

# ======================================
# --- Main Benchmark Execution ---
# ======================================
if __name__ == "__main__":
    print("==============================")
    print("Benchmarking SGLang Server with Multi-Family Warmup (ThreadPoolExecutor)")
    print("==============================")

    families = generate_prefixes(NUM_FAMILIES)

    # Step 1: Warmup all families
    print("\n=== Step 1: Warming up all families ===")
    for family in families:
        warmup_prefix(family)
    time.sleep(2)

    # # Step 2: Generate Prompts
    all_prompts = generate_prompts_from_families(families, NUM_PROMPTS_PER_FAMILY)
    total_reqs = len(all_prompts)
    print(f"\n=== Step 2: Sending {total_reqs} prompts concurrently ===\n")

    # # Step 3: Send Concurrent Prompts using ThreadPoolExecutor
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, (family, prompt) in enumerate(all_prompts):
            futures.append(executor.submit(send_request, family, prompt, i + 1))

        # Optional: wait for all to finish
        for future in as_completed(futures):
            pass

    # Step 4: Report
    end_time = time.time()
    elapsed = end_time - start_time
    rps = total_reqs / elapsed
    print("\n==============================")
    print(f"Completed {total_reqs} requests in {elapsed:.2f} seconds")
    print(f"Throughput: {rps:.2f} requests/second")
    print("==============================")
