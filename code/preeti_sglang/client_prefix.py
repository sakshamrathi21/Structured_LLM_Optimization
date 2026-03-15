import random
import string
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from system_prompt import queries, system_prompt
import threading
import itertools

# ======================================
# --- Configuration ---
# ======================================
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
URL = "http://127.0.0.1:30000/v1/chat/completions"

TEMPERATURE = 0.7
MAX_CONNECTIONS = 500  # max TCP connections to reuse


# ======================================
# --- User Input ---
# ======================================
# try:
#     NUM_FAMILIES = int(input("Enter number of families to warm up (e.g., 1, 5, 10, 50, 100): "))
# except ValueError:
#     print("Invalid input. Using default NUM_FAMILIES = 10.")
#     NUM_FAMILIES = 10
import sys
NUM_FAMILIES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

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
        response = session.post(URL, json=data, timeout=20)
        if response.status_code == 200:
            print(f"[Warmup] Family {family} cached successfully.")
        else:
            print(f"[Warmup] Family {family} failed ({response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"[Warmup] Connection error for {family}: {e}")


# ======================================
# --- Main Benchmark Execution ---
# ======================================
if __name__ == "__main__":
    print("==============================")
    print("Benchmarking SGLang Server with Multi-Family Warmup")
    print("==============================")

    families = generate_prefixes(NUM_FAMILIES)

    print(f"\n=== Warming up {NUM_FAMILIES} families ===")

    # Run warmup sequentially (or parallel if desired)
    for family in families:
        warmup_prefix(family)

    for session in session_pool:
        session.close()

    print("\n✅ Warmup completed for all families.")

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

# NUM_FAMILIES = 50            # number of prefix families
# TEMPERATURE = 0.7
# MAX_CONNECTIONS = 500         # max TCP connections to reuse


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
# # --- Main Benchmark Execution ---
# # ======================================
# if __name__ == "__main__":
#     print("==============================")
#     print("Benchmarking SGLang Server with Multi-Family Warmup")
#     print("==============================")

#     families = generate_prefixes(NUM_FAMILIES)

#     # Step 1: Warmup all families
#     print(f"\n=== Warming up {NUM_FAMILIES} families ===")
#     for family in families:
#         warmup_prefix(family)

#     for session in session_pool:
#         session.close()