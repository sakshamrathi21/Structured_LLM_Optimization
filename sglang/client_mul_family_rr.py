import random
import string
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from system_prompt import queries, system_prompt
import threading
import itertools  # for cycling queries

# ======================================
# --- Configuration ---
# ======================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
URL = "http://127.0.0.1:30000/v1/chat/completions"

TOTAL_REQUESTS = 10000        # total number of requests across all families
NUM_FAMILIES = 100            # number of prefix families
TEMPERATURE = 0.7
MAX_TOKENS = 50
MAX_CONNECTIONS = 500         # max TCP connections to reuse
MAX_WORKERS = 800             # max threads in ThreadPoolExecutor
REQUEST_RATE = 200            # requests per second (fixed)

# ======================================
# --- Prepare Session Pool ---
# ======================================
session_pool = [requests.Session() for _ in range(MAX_CONNECTIONS)]
session_lock = threading.Lock()
next_session_idx = 0

def get_session():
    """Round-robin session selection to reuse TCP connections."""
    global next_session_idx
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
    """Send user prompt (system prompt already cached)."""
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    session = get_session()
    try:
        session.post(URL, json=data, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"[ConnError {request_id}] {e}")

# ======================================
# --- Prompt Generation (add family prefix once) ---
# ======================================
def generate_prompts_for_total(families, total_requests):
    num_families = len(families)
    base_count = total_requests // num_families
    remainder = total_requests % num_families

    all_prompts = []
    for i, family in enumerate(families):
        num_requests_for_family = base_count + (1 if i < remainder else 0)
        cycled_queries = itertools.islice(itertools.cycle(queries), num_requests_for_family)
        for q in cycled_queries:
            # Add prefix here once (no runtime cost later)
            prefixed_query = f"{family} {q}"
            all_prompts.append((family, prefixed_query))
    random.shuffle(all_prompts)
    # for family, prompt in all_prompts:
    #     print(f"{prompt}")
    return all_prompts

# ======================================
# --- Main Benchmark Execution ---
# ======================================
if __name__ == "__main__":
    print("==============================")
    print("Benchmarking SGLang Server with Multi-Family Warmup + Fixed Request Rate")
    print("==============================")

    families = generate_prefixes(NUM_FAMILIES)

    # Step 1: Warmup all families
    print(f"\n=== Step 1: Warming up {NUM_FAMILIES} families ===")
    for family in families:
        warmup_prefix(family)
    time.sleep(2)

    # Step 2: Generate Prompts (Total distributed among families)
    all_prompts = generate_prompts_for_total(families, TOTAL_REQUESTS)
    total_reqs = len(all_prompts)
    print(f"\n=== Step 2: Sending {total_reqs} prompts at {REQUEST_RATE} req/sec ===\n")

    # Step 3: Controlled Throughput Sending
    batch_size = REQUEST_RATE   # number of requests per second
    initial_start_time = time.time()
    completed_requests = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, total_reqs, batch_size):
            start_time = time.time()
            batch = all_prompts[i:i + batch_size]

            # Submit all requests for this 1-second window
            futures = [executor.submit(send_request, fam, prompt, i + j)
                       for j, (fam, prompt) in enumerate(batch)]

            # Wait for this batch to complete
            for future in as_completed(futures):
                completed_requests += 1

            # Maintain 1-second pacing
            elapsed = time.time() - start_time
            sleep_time = max(0, 1 - elapsed)
            time.sleep(sleep_time)

    end_time = time.time()
    total_elapsed = end_time - initial_start_time
    throughput = completed_requests / total_elapsed

    # Step 4: Report
    # print("\n==============================")
    # print(f"{completed_requests} prompts sent across {NUM_FAMILIES} families")
    # print(f"Configured rate: {REQUEST_RATE} req/sec")
    # print(f"Achieved throughput: {throughput:.2f} req/sec")
    # print(f"Total elapsed: {total_elapsed:.2f} seconds")
    # print("==============================")

    print("\n==============================")
    print(f"{completed_requests} prompts at {REQUEST_RATE} req/sec with {NUM_FAMILIES} families, max tokens: {MAX_TOKENS}, max connections: {MAX_CONNECTIONS}, max workers: {MAX_WORKERS}, mul_fam_dcgmi_{NUM_FAMILIES}_{REQUEST_RATE}rr.csv")
    print(f"Completed {completed_requests} requests in {total_elapsed:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print("==============================")

    for session in session_pool:
        session.close()


# import random
# import string
# import time
# import requests
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from system_prompt import queries, system_prompt
# import threading
# import itertools  # for cycling queries

# # ======================================
# # --- Configuration ---
# # ======================================
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# URL = "http://127.0.0.1:30000/v1/chat/completions"

# TOTAL_REQUESTS = 10000        # total number of requests across all families
# NUM_FAMILIES = 100             # number of prefix families
# TEMPERATURE = 0.7
# MAX_TOKENS = 30
# MAX_CONNECTIONS = 500         # max TCP connections to reuse
# MAX_WORKERS = 800            # max threads in ThreadPoolExecutor
# REQUEST_RATE = 200           # requests per second (fixed)

# # ======================================
# # --- Prepare Session Pool ---
# # ======================================
# session_pool = [requests.Session() for _ in range(MAX_CONNECTIONS)]
# session_lock = threading.Lock()
# next_session_idx = 0

# def get_session():
#     """Round-robin session selection to reuse TCP connections."""
#     global next_session_idx
#     with session_lock:
#         session = session_pool[next_session_idx]
#         next_session_idx = (next_session_idx + 1) % MAX_CONNECTIONS
#     return session

# # ======================================
# # --- Utility: Generate Family Prefixes ---
# # ======================================
# def generate_prefixes(n):
#     letters = string.ascii_uppercase
#     prefixes = []
#     for a in letters:
#         for b in letters:
#             prefixes.append(a + b)
#     return prefixes[:n]

# # ======================================
# # --- Warmup Phase (System Prompt Only) ---
# # ======================================
# def warmup_prefix(family):
#     """Send system prompt to warm up KV cache for this family."""
#     print(f"[Warmup] Sending system prompt for family {family}...")
#     data = {
#         "model": MODEL_NAME,
#         "messages": [{"role": "system", "content": f"{family} {system_prompt}"}],
#         "temperature": 0.0,
#         "max_tokens": 10
#     }
#     session = get_session()
#     try:
#         response = session.post(URL, json=data)
#         if response.status_code == 200:
#             print(f"[Warmup] Family {family} cached successfully.")
#         else:
#             print(f"[Warmup] Family {family} failed ({response.status_code})")
#     except requests.exceptions.RequestException as e:
#         print(f"[Warmup] Connection error for {family}: {e}")

# # ======================================
# # --- Send Request Function (User Prompts Only) ---
# # ======================================
# def send_request(family, prompt, request_id):
#     """Send only the user prompt — system prompt already cached."""
#     data = {
#         "model": MODEL_NAME,
#         "messages": [{"role": "user", "content": prompt}],
#         "temperature": TEMPERATURE,
#         "max_tokens": MAX_TOKENS
#     }
#     session = get_session()
#     try:
#         session.post(URL, json=data)
#     except requests.exceptions.RequestException as e:
#         print(f"[ConnError {request_id}] {e}")

# # ======================================
# # --- Prompt Generation (Total = 10,000 distributed evenly) ---
# # ======================================
# def generate_prompts_for_total(families, total_requests):
#     num_families = len(families)
#     base_count = total_requests // num_families
#     remainder = total_requests % num_families

#     all_prompts = []
#     for i, family in enumerate(families):
#         # Families with remainder get one extra
#         num_requests_for_family = base_count + (1 if i < remainder else 0)

#         # Cycle through queries as needed
#         cycled_queries = itertools.islice(itertools.cycle(queries), num_requests_for_family)
#         for prompt in cycled_queries:
#             all_prompts.append((family, prompt))

#     # random.shuffle(all_prompts)
#     return all_prompts

# # def generate_prompts_for_total(families, total_requests):
# #     num_families = len(families)
# #     base_count = total_requests // num_families
# #     remainder = total_requests % num_families

# #     all_prompts = []
# #     for i, family in enumerate(families):
# #         # Families with remainder get one extra
# #         num_requests_for_family = base_count + (1 if i < remainder else 0)

# #         # Cycle through queries as needed
# #         cycled_queries = itertools.islice(itertools.cycle(queries), num_requests_for_family)
# #         for q in cycled_queries:
# #             # Add prefix here once (no runtime cost during sending)
# #             prefixed_query = f"{family} {q}"
# #             all_prompts.append((family, prefixed_query))

# #     random.shuffle(all_prompts)
# #     # for family, prompt in all_prompts:
# #     #     print(f"{prompt}")
# #     return all_prompts


# # ======================================
# # --- Main Benchmark Execution ---
# # ======================================
# if __name__ == "__main__":
#     print("==============================")
#     print("Benchmarking SGLang Server with Multi-Family Warmup + Fixed Request Rate")
#     print("==============================")

#     families = generate_prefixes(NUM_FAMILIES)

#     # Step 1: Warmup all families
#     print("\n=== Step 1: Warming up all families ===")
#     for family in families:
#         warmup_prefix(family)
#     time.sleep(2)

#     # Step 2: Generate Prompts (Total distributed among families)
#     all_prompts = generate_prompts_for_total(families, TOTAL_REQUESTS)
#     total_reqs = len(all_prompts)
#     print(f"\n=== Step 2: Sending {total_reqs} prompts at {REQUEST_RATE} req/sec ===\n")

#     # Step 3: Send Concurrent Prompts with fixed request rate
#     start_time = time.time()
#     interval = 1.0 / REQUEST_RATE  # delay between submissions

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = []
#         for i, (family, prompt) in enumerate(all_prompts):
#             futures.append(executor.submit(send_request, family, prompt, i + 1))
#             time.sleep(interval)

#         for future in as_completed(futures):
#             pass

#     # Step 4: Report
#     end_time = time.time()
#     elapsed = end_time - start_time
#     rps = total_reqs / elapsed
#     print("\n==============================")
#     print(f"{total_reqs} prompts at {REQUEST_RATE} req/sec with {NUM_FAMILIES} families, max tokens: {MAX_TOKENS}, max connections: {MAX_CONNECTIONS}, max workers: {MAX_WORKERS}, mul_fam_dcgmi_{NUM_FAMILIES}_{REQUEST_RATE}rr.csv")
#     print(f"Completed {total_reqs} requests in {elapsed:.2f} seconds")
#     print(f"Throughput: {rps:.2f} requests/second")
#     print("==============================")

#     for session in session_pool:
#         session.close()