from nicegui import ui
import requests

# Config
LLM_PORT = 30000
LLM_URL = f"http://localhost:{LLM_PORT}/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# State
chat_history = []

# UI Setup
ui.label("🤖 LLM Chat Interface").classes("text-h4")

chat_container = ui.column().classes("h-96 overflow-y-auto border p-2")

user_input = ui.input("Type your message...").props('clearable').on('keydown.enter', lambda e: send_message())
ui.button("Send", on_click=lambda: send_message())

# Message rendering
def render_message(role, content):
    prefix = "👤 You: " if role == "user" else "🤖 Bot: "
    with chat_container:
        ui.label(f"{prefix}{content}").classes("text-left w-full")

# Send handler
def send_message():
    message = user_input.value.strip()
    if not message:
        return

    render_message("user", message)
    chat_history.append({"role": "user", "content": message})
    user_input.value = ""

    payload = {
        "model": MODEL_NAME,
        "messages": chat_history
    }

    try:
        response = requests.post(LLM_URL, json=payload)
        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            chat_history.append({"role": "assistant", "content": reply})
            render_message("assistant", reply)
        else:
            render_message("assistant", f"❌ Error: {response.status_code}")
    except Exception as e:
        render_message("assistant", f"⚠️ Request failed: {str(e)}")

# Launch app
ui.run()
