import requests
import threading
import time
# from transformers import AutoTokenizer

# --- Configuration ---
# Make sure this port matches your running sglang server
PORT = 30000
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# "qwen/qwen2.5-0.5b-instruct"

# A list of prompts we want to send to the model concurrently
prompts = [
    "Please keep the answer under 100 words.",
    "What is the capitol of India?",
    "What is the capitol of pakistan?",
    "What is the capitol of USA?",
    "Hi, how is life going?",
    "Write a short, futuristic story about Mumbai in the year 2077.",
    "What is the best way to cook paneer tikka?",
    "How to cook paneer tikka?",
    "What are the ingredients in paneer tikka?",
    "Explain the concept of monsoons in simple terms.",
    "Write a Python function that finds the factorial of a number.",
]

# tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# # text = 'Hello, how are you?'

# for i in range(len(prompts)):
#     text = prompts[i]
#     print(tok(text)) #Display input_ids
#     print(tok.tokenize(text)) #Display tokens
#     print(tok.convert_ids_to_tokens(tok(text)["input_ids"]))

# prompts = [
#     "Please keep the answer under 1000 words.",
#     "What is an operating system?",
#     "Explain the difference between process and thread.",
#     "What is a system call in operating systems?",
#     "Explain the concept of kernel in an OS.",
#     "What is the difference between monolithic kernel and microkernel?",
#     "Explain process scheduling in operating systems.",
#     "What is a context switch?",
#     "How does an OS handle deadlock?",
#     "What are the necessary conditions for deadlock?",
#     "Explain paging in operating systems.",
#     "What is segmentation in memory management?",
#     "Explain virtual memory.",
#     "What is demand paging?",
#     "What are page replacement algorithms?",
#     "Explain FIFO page replacement.",
#     "Explain LRU page replacement.",
#     "What is thrashing in operating systems?",
#     "Explain the concept of file system.",
#     "What is inode in file systems?",
#     "What is journaling in file systems?",
#     "Explain inter-process communication (IPC).",
#     "What is message passing in IPC?",
#     "What is shared memory in IPC?",
#     "Explain semaphores in OS.",
#     "Explain mutex in OS.",
#     "What is the difference between semaphore and mutex?",
#     "Explain critical section problem.",
#     "What is a monitor in operating systems?",
#     "What is priority scheduling?",
#     "Explain round-robin scheduling.",
#     "What is shortest job first scheduling?",
#     "Explain multilevel queue scheduling.",
#     "What is starvation in scheduling?",
#     "What is aging in scheduling?",
#     "Explain producer-consumer problem.",
#     "Explain readers-writers problem.",
#     "Explain dining philosophers problem.",
#     "What is process synchronization?",
#     "Explain time-sharing systems.",
#     "What is a real-time operating system?",
#     "What is the difference between hard and soft real-time OS?",
#     "What is a distributed operating system?",
#     "What is the difference between multiprogramming and multitasking?",
#     "Explain multiprocessing.",
#     "What is asymmetric multiprocessing?",
#     "What is symmetric multiprocessing?",
#     "What is the difference between multitasking and multithreading?",
#     "Explain hybrid kernel.",
#     "What is device driver in OS?",
#     "What is interrupt handling?",
#     "Explain the concept of system boot process.",
#     "What is BIOS?",
#     "What is UEFI?",
#     "What is bootloader?",
#     "What is swap space?",
#     "What is memory fragmentation?",
#     "Explain internal fragmentation.",
#     "Explain external fragmentation.",
#     "What is compaction in memory management?",
#     "What is the buddy system in memory allocation?",
#     "What are overlays in OS?",
#     "Explain demand segmentation.",
#     "What is cache memory?",
#     "What is the difference between primary memory and secondary memory?",
#     "Explain spooling in operating systems.",
#     "What is the difference between user mode and kernel mode?",
#     "What are system programs in OS?",
#     "What is the role of shell in an operating system?",
#     "What is the role of command interpreter?",
#     "Explain swap in Linux.",
#     "What is a zombie process?",
#     "What is an orphan process?",
#     "What is a daemon process?",
#     "Explain process states in OS.",
#     "What is fork() in Unix/Linux?",
#     "What is exec() in Unix/Linux?",
#     "What is wait() system call?",
#     "What is signal handling in OS?",
#     "Explain interleaving in concurrent processes.",
#     "What is atomic operation?",
#     "What is memory-mapped I/O?",
#     "What is polling in OS?",
#     "What is DMA (Direct Memory Access)?",
#     "What is the difference between kernel space and user space?",
#     "Explain file permissions in Unix/Linux.",
#     "What is virtual file system (VFS)?",
#     "What is log-structured file system?",
#     "Explain journaling vs non-journaling file systems.",
#     "What is SSD vs HDD from an OS perspective?",
#     "Explain buffer cache in OS.",
#     "What is a system resource?",
#     "Explain resource allocation graph.",
#     "What is banker’s algorithm?",
#     "What is a deadlock prevention technique?",
#     "What is deadlock avoidance?",
#     "What is deadlock detection?",
#     "What are system daemons in Linux?",
#     "What is the role of init/systemd?",
#     "What is kernel panic?",
#     "Explain the concept of OS security.",
#     "What is privilege escalation in OS?",
#     "What are system logs in Linux?",
#     "Explain the concept of containers vs virtual machines in OS."
# ]

# --- Function to be executed by each thread ---
def send_request(prompt, request_id):
    """Sends a single request to the LLM server and prints the response."""
    print(f"[Request {request_id}] Sending prompt: '{prompt[:30]}...'")

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, # Optional: Add other parameters
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
        print(f"Please ensure your sglang server is running on port {PORT}.")
        print(f"Error: {e}")
        print("---------------------------\n")


# --- Main execution block ---
if __name__ == "__main__":
    print(f"Starting {len(prompts)} concurrent requests to the sglang server...")
    start_time = time.time()

    threads = []
    # Create and start a thread for each prompt
    for i, prompt in enumerate(prompts):
        # The 'target' is the function to run in the thread.
        # The 'args' is a tuple of arguments to pass to that function.
        thread = threading.Thread(target=send_request, args=(prompt, i + 1))
        threads.append(thread)
        thread.start() # Starts the thread's execution

    # Wait for all threads to complete their execution
    # This is crucial! The main script will pause here until all requests are done.
    for thread in threads:
        thread.join()

    end_time = time.time()
    print(f"\nAll {len(prompts)} requests completed in {end_time - start_time:.2f} seconds.")


# nsys profile --gpu-metrics-device=all -o l2_mterics.qdrep python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.7 --disable-cuda-graph --host 0.0.0.0
# nsys stats --report gputrace --report gpukernsum --report cudaapisum --format csv,column --output .,- l2_mterics.nsys-rep
# ncu --target-processes all \
#     --set full \
#     --launch-skip 0 \
#     --launch-count 1 \
#     --metrics lts__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_hit_rate.pct \
#     --sampling-interval auto \
#     python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --mem-fraction-static 0.7 --disable-cuda-graph --host 0.0.0.0

# ncu --target-processes all \
#     --set full \
#     --launch-skip 0 \
#     --launch-count 1 \
#     --metrics lts__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_hit_rate.pct \
#     --sampling-interval auto \
#     python3 req.py
