import requests
import threading

port = 30000
url = f"http://localhost:{port}/v1/chat/completions"

model_name = "meta-llama/Llama-3.1-8B-Instruct"

def send_request(message):
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": message}],
    }
    try:
        response = requests.post(url, json=data)
        print(f"\n[Thread] Status code: {response.status_code}")
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print("[Thread] Model reply:", reply)
        else:
            print("[Thread] Error:", response.text)
    except Exception as e:
        print("[Thread] Exception:", e)

# Messages to send simultaneously
messages = [
    "Hi, how is life going?",
    "Tell me a fun fact.",
    "What’s the weather like on Mars?",
    "Write me a short poem about AI.",
    "Explain quantum computing in simple terms."
]

threads = []
for msg in messages:
    t = threading.Thread(target=send_request, args=(msg,))
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("\nAll requests completed.")
