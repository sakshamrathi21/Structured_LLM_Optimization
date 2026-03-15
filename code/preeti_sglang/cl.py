import requests

url = "http://localhost:30000/v1/chat/completions"

headers = {"Content-Type": "application/json"}

# Example payloads – you can add more prompts if you want
payloads = [
    {
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Write a short poem about GPUs."}],
        "max_tokens": 64,
        "temperature": 0.7
    },
    {
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "Explain what an L2 cache does in a GPU."}],
        "max_tokens": 64,
        "temperature": 0.7
    }
]

for i, payload in enumerate(payloads, 1):
    print(f"\n--- Sending request {i} ---")
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print(response.json()["choices"][0]["message"]["content"])
    else:
        print(f"Error {response.status_code}: {response.text}")
