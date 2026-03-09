import requests
import threading
import time
import subprocess

# --- Configuration ---
PORT = 30000
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-0.5b-instruct"

prompts = [
    "Hi, how is life going?",
    "Write a short, futuristic story about Mumbai in the year 2077.",
    "What is the best way to cook paneer tikka?",
    "How to cook paneer tikka?",
    "What are the ingredients in paneer tikka?",
    "Explain the concept of monsoons in simple terms.",
    "Write a Python function that finds the factorial of a number.",
]

def send_request(prompt, request_id):
    print(f"[Request {request_id}] Sending prompt: '{prompt[:30]}...'")

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    try:
        response = requests.post(URL, json=data)

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"\n--- [Reply for Request {request_id}] ---\n{reply}\n---------------------------\n")
        else:
            print(f"\n--- [Error for Request {request_id}] ---")
            print(f"Status code: {response.status_code}")
            print(f"Error details: {response.text}")
            print("---------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n--- [Connection Error for Request {request_id}] ---")
        print(f"Could not connect to the server at {URL}.")
        print(f"Error: {e}")
        print("---------------------------\n")

def run_with_profiling():
    """
    Launch ncu profiler to measure L2 cache hit rate
    while running this script itself.
    """
    metrics = "l2_tex_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate"
    cmd = [
        "ncu",
        "--metrics", metrics,
        "--target-processes", "all",
        "python3", __file__, "--no-ncu"   # Avoid recursive profiling
    ]
    print(f"Running under profiler: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    import sys
    if "--no-ncu" not in sys.argv:
        # If profiling not disabled, restart under ncu
        run_with_profiling()
        sys.exit(0)

    print(f"Starting {len(prompts)} concurrent requests to the sglang server...")
    start_time = time.time()

    threads = []
    for i, prompt in enumerate(prompts):
        thread = threading.Thread(target=send_request, args=(prompt, i + 1))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"\nAll {len(prompts)} requests completed in {end_time - start_time:.2f} seconds.")
