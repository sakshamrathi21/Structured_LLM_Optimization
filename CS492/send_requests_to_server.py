import requests

port = 30000
url = f"http://localhost:{port}/v1/chat/completions"

llm_input_message = "Hi, how is life going?"

data = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": llm_input_message}],
}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Full JSON response:", response.json())

if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print("\nModel reply:", reply)
else:
    print("Error:", response.text)
