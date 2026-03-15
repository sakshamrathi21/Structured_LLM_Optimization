import requests
import threading
import time
import random
import os

# =============================
# --- Configuration Section ---
# =============================

PORT = 30000
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

CHAPTER_DIR = "families"   # Folder containing chapter text files
TARGET_CHAPTER = "chapter_4" # e.g. "Chapter 4", "Chapter 7", "Chapter 39"
NUM_PROMPTS = 900             # Number of concurrent requests to send

# =============================
# --- Chapter Topics ---
# =============================

CHAPTER_TOPICS = {
    "chapter_4": [
        "process creation", "context switching", "process control block (PCB)",
        "system calls", "inter-process communication", "fork and exec", "process states"
    ],
    "chapter_7": [
        "CPU scheduling algorithms", "First-Come, First-Served scheduling", "Shortest Job First",
        "Round-Robin scheduling", "priority scheduling", "multilevel feedback queue",
        "context switching overhead", "throughput optimization"
    ],
    "chapter_39": [
        "inode management", "file descriptors", "directory structure",
        "hard and soft links", "file access permissions", "journaling file systems",
        "mounting and unmounting", "metadata updates"
    ],
}

# =============================
# --- Prompt Templates ---
# =============================

PROMPT_TEMPLATES = [
    "Explain the concept of {} in detail.",
    "Provide an example of {} and explain its importance.",
    "How does {} affect overall system performance?",
    "Describe how {} works in an operating system.",
    "What are the design challenges involved in {}?",
    "Why is {} critical for OS efficiency and stability?"
]

# =============================
# --- Utility: Load Chapter ---
# =============================

def load_chapter_text(chapter_name):
    """
    Reads the given chapter text from ./families/<chapter_name>.txt
    Example: ./families/chapter_4.txt
    """
    filename = os.path.join(CHAPTER_DIR, f"{chapter_name}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Chapter file not found: {filename}\n"
            "Make sure the file exists, e.g. './families/chapter_4.txt'"
        )
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# =============================
# --- Prompt Generation ---
# =============================

def generate_prompts_for_chapter(chapter, num_prompts=100):
    """Generates diverse question prompts for a given chapter."""
    prompts = []
    topics = CHAPTER_TOPICS[chapter]
    for _ in range(num_prompts):
        topic = random.choice(topics)
        template = random.choice(PROMPT_TEMPLATES)
        prompt = template.format(topic)
        prompts.append((chapter, prompt))
    return prompts

# =============================
# --- Warmup Phase (Prefix) ---
# =============================

def warmup_prefix(chapter):
    """Sends the full chapter text once to prime KV cache."""
    prefix_text = load_chapter_text(chapter)
    print(f"\nSending prefix knowledge for {chapter} to warm up KV cache...")

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an OS expert."},
            {"role": "user", "content": prefix_text}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }

    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            print(f"Prefix for {chapter} cached successfully.\n")
        else:
            print(f"Prefix warmup failed: {response.status_code} — {response.text}\n")
    except requests.exceptions.RequestException as e:
        print(f"Connection error during prefix warmup: {e}")

# =============================
# --- Send Request Function ---
# =============================

def send_request(chapter, prompt, request_id):
    """Send a single request to the LLM server."""
    print(f"[Req {request_id}] Sending prompt: '{prompt[:50]}...'")

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"You are an OS expert discussing {chapter}."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"\n--- [Reply {request_id}] ---\n{reply}\n---------------------------\n")
        else:
            print(f"\n--- [Error {request_id}] --- {response.status_code} | {response.text}\n")
    except requests.exceptions.RequestException as e:
        print(f"\n--- [Connection Error {request_id}] ---\n{e}\n")

# =============================
# --- Main Benchmark Execution ---
# =============================

if __name__ == "__main__":
    print(f"==============================")
    print(f"Benchmarking LLM Server on {TARGET_CHAPTER}")
    print(f"==============================")

    # Step 1: Warmup Phase
    warmup_prefix(TARGET_CHAPTER)

    # Step 2: Generate Prompts
    all_prompts = generate_prompts_for_chapter(TARGET_CHAPTER, NUM_PROMPTS)
    total_reqs = len(all_prompts)
    print(f"Starting benchmark with {total_reqs} concurrent prompts...\n")

    # Step 3: Concurrent Requests
    start_time = time.time()
    threads = []

    for i, (chapter, prompt) in enumerate(all_prompts):
        t = threading.Thread(target=send_request, args=(chapter, prompt, i + 1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Step 4: Report
    end_time = time.time()
    elapsed = end_time - start_time
    rps = total_reqs / elapsed

    print("\n==============================")
    print(f"Completed {total_reqs} requests in {elapsed:.2f}s")
    print(f"Throughput: {rps:.2f} requests/sec")
    print("==============================")
