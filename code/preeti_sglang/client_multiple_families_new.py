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

CHAPTER_DIR = "./families"    # Folder containing chapter text files
CHAPTERS = ["chapter_4", "chapter_7", "chapter_39"]  # Families to benchmark
NUM_PROMPTS_PER_FAMILY = 300  # Number of concurrent prompts per chapter

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
    """Reads the chapter text from ./chapters/<chapter_name>.txt"""
    filename = os.path.join(CHAPTER_DIR, f"{chapter_name}.txt")
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Chapter file not found: {filename}\n"
            "Make sure the file exists in the chapters/ directory."
        )
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

# =============================
# --- Prompt Generation ---
# =============================

def generate_prompts_from_chapters(num_per_chapter=100):
    """Generates a combined list of (chapter, prompt) tuples."""
    all_prompts = []
    for chapter in CHAPTERS:
        topics = CHAPTER_TOPICS[chapter]
        for _ in range(num_per_chapter):
            topic = random.choice(topics)
            template = random.choice(PROMPT_TEMPLATES)
            prompt = template.format(topic)
            all_prompts.append((chapter, prompt))
    return all_prompts

# =============================
# --- Warmup Phase (Prefix) ---
# =============================

def warmup_prefix(chapter):
    """Sends the full chapter text once to warm up the KV cache."""
    prefix_text = load_chapter_text(chapter)
    print(f"\nSending prefix knowledge for {chapter} to warm up KV cache...")

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an operating systems expert."},
            {"role": "user", "content": prefix_text}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }

    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            print(f"Prefix for {chapter} cached successfully.\n")
        else:
            print(f"Prefix warmup failed for {chapter}: {response.status_code}")
            print(f"Error details: {response.text}\n")
    except requests.exceptions.RequestException as e:
        print(f"Connection error during prefix warmup for {chapter}: {e}\n")

# =============================
# --- Send Request Function ---
# =============================

def send_request(chapter, prompt, request_id):
    """Sends a single request to the LLM server."""
    print(f"[Request {request_id}] Sending prompt: '{prompt[:60]}...'")

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
    print("==============================")
    print("Benchmarking LLM Server with Multiple Chapter Families")
    print("==============================")

    # Step 1: Warm up KV cache for all chapters
    for chapter in CHAPTERS:
        warmup_prefix(chapter)

    # Step 2: Generate Prompts
    all_prompts = generate_prompts_from_chapters(NUM_PROMPTS_PER_FAMILY)
    total_reqs = len(all_prompts)
    print(f"Starting benchmark with {total_reqs} concurrent prompts across {len(CHAPTERS)} chapters...\n")

    # Step 3: Concurrent Requests
    start_time = time.time()
    threads = []

    for i, (chapter, prompt) in enumerate(all_prompts):
        t = threading.Thread(target=send_request, args=(chapter, prompt, i + 1))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Step 4: Report Results
    end_time = time.time()
    elapsed = end_time - start_time
    rps = total_reqs / elapsed

    print("\n==============================")
    print(f"Completed {total_reqs} requests in {elapsed:.2f} seconds")
    print(f"Throughput: {rps:.2f} requests/second")
    print("==============================")
