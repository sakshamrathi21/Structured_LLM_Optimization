import requests
import threading
import time
# from transformers import AutoTokenizer

# --- Configuration ---
# Make sure this port matches your running sglang server
PORT = 30000
URL = f"http://localhost:{PORT}/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-0.5b-instruct" #"meta-llama/Llama-3.1-8B-Instruct"

# A list of prompts we want to send to the model concurrently
# prompts = [
    # "Please keep the answer under 1000 words.",
    # "What is the capitol of India?",
    # "What is the capitol of pakistan?",
    # "What is the capitol of USA?",
    # "Hi, how is life going?",
    # "Write a short, futuristic story about Mumbai in the year 2077.",
    # "What is the best way to cook paneer tikka?",
    # "How to cook paneer tikka?",
    # "What are the ingredients in paneer tikka?",
    # "Explain the concept of monsoons in simple terms.",
    # "Write a Python function that finds the factorial of a number.",
# ]

# tok = AutoTokenizer.from_pretrained(MODEL_NAME)
# # text = 'Hello, how are you?'

# for i in range(len(prompts)):
#     text = prompts[i]
#     print(tok(text)) #Display input_ids
#     print(tok.tokenize(text)) #Display tokens
#     print(tok.convert_ids_to_tokens(tok(text)["input_ids"]))

prompts = [
    "What is the main goal of MLFQ scheduling?",
    "Who first described the MLFQ scheduler and in what system?",
    "What are the two primary problems MLFQ tries to solve?",
    "Why can’t SJF or STCF be directly applied in practice?",
    "Why is Round Robin bad for turnaround time?",
    "What is the crux question behind MLFQ scheduling?",
    "How does MLFQ attempt to predict job behavior?",
    "What does 'learning from history' mean in MLFQ?",
    "Why does MLFQ use multiple queues?",
    "How are jobs assigned to queues in MLFQ?",
    "What is Rule 1 of MLFQ?",
    "What is Rule 2 of MLFQ?",
    "How does MLFQ handle jobs with equal priority?",
    "How does MLFQ decide which job runs first?",
    "Why do interactive jobs usually retain high priority?",
    "Why are CPU-intensive jobs demoted in MLFQ?",
    "What is a job’s allotment in MLFQ?",
    "What happens if a job uses up its allotment?",
    "What happens if a job relinquishes the CPU early?",
    "What happens to a single long-running job in MLFQ?",
    "How does MLFQ approximate SJF?",
    "How are short jobs favored in MLFQ?",
    "What happens when an interactive job frequently gives up the CPU?",
    "Why does MLFQ risk starvation of long-running jobs?",
    "What is gaming the scheduler in MLFQ?",
    "How can a job game the scheduler under old rules?",
    "What is the effect of priority boost in MLFQ?",
    "What problem does Rule 5 solve?",
    "How often should priority boosts occur?",
    "Why are such constants called 'voodoo constants'?",
    "What is Ousterhout’s Law?",
    "How can better accounting prevent gaming in MLFQ?",
    "What is the updated Rule 4 for MLFQ?",
    "Why does MLFQ need better accounting?",
    "What is starvation in MLFQ?",
    "How can MLFQ adapt to a job changing behavior over time?",
    "How many queues are typically used in MLFQ?",
    "Why are higher-priority queues given shorter time slices?",
    "Why are lower-priority queues given longer time slices?",
    "What is the Solaris MLFQ implementation called?",
    "How many queues does Solaris TS typically use?",
    "What does FreeBSD use instead of exact rules?",
    "What is decay-usage scheduling?",
    "Why might some queues be reserved for OS work?",
    "What does the 'nice' utility do in scheduling?",
    "What is the purpose of advice in scheduling?",
    "How does MLFQ handle both interactive and CPU-bound jobs?",
    "What is the fundamental advantage of MLFQ?",
    "Which real systems implement MLFQ variants?",
    "How does MLFQ balance fairness with performance?",
    "Why does MLFQ not require a priori job knowledge?",
    "What is turnaround time in scheduling?",
    "What is response time in scheduling?",
    "How does MLFQ optimize turnaround time?",
    "How does MLFQ minimize response time?",
    "What is the Compatible Time-Sharing System (CTSS)?",
    "What later system refined MLFQ after CTSS?",
    "Why was Corbato awarded the Turing Award?",
    "What is the significance of Multics in MLFQ history?",
    "Why does MLFQ assume new jobs might be short?",
    "What happens to long jobs over time in MLFQ?",
    "How does MLFQ avoid interactive job starvation?",
    "Why is starvation dangerous in a scheduling system?",
    "Why is security relevant in scheduling policies?",
    "How can malicious users exploit scheduling?",
    "What is the difference between Rule 4a and Rule 4b?",
    "Why were Rules 4a and 4b merged into Rule 4?",
    "How does I/O affect priority in MLFQ?",
    "Why should relinquishing CPU not always reset allotment?",
    "What happens in MLFQ if a process mixes CPU and I/O phases?",
    "Why must scheduling be secure against attacks?",
    "What happens when priority boost is too frequent?",
    "What happens when priority boost is too infrequent?",
    "Why is finding the right S value difficult?",
    "How does machine learning help tune scheduling?",
    "What does Figure 8.2 illustrate?",
    "What does Figure 8.3 illustrate?",
    "What does Figure 8.4 show with and without boosts?",
    "What does Figure 8.5 show about gaming?",
    "What does Figure 8.6 demonstrate about time slices?",
    "How does MLFQ differ from fixed-priority scheduling?",
    "Why is round robin used within queues?",
    "How does MLFQ ensure fairness?",
    "What happens to a job after being boosted?",
    "How does MLFQ handle mixed workloads?",
    "What is the role of history in MLFQ?",
    "Why does MLFQ provide good responsiveness?",
    "How does MLFQ prevent monopolization of CPU?",
    "Why does the OS rarely know job length?",
    "Why must MLFQ adapt dynamically?",
    "What is the advantage of multiple queues vs one?",
    "What is an allotment relative to time slice?",
    "How do queue lengths affect performance?",
    "Why is administrator tuning important in MLFQ?",
    "What is the relationship between MLFQ and SJF?",
    "How does MLFQ approximate STCF?",
    "What is the general principle behind feedback in scheduling?",
    "Why is advice optional for OS to follow?",
    "What are the five final rules of MLFQ?",
    "How does MLFQ prevent monopolization of CPU?",
]

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