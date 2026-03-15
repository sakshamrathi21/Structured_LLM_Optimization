import requests

url = "http://localhost:30000/v1/chat/completions"
data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}

response = requests.post(url, json=data)
print(response.json())