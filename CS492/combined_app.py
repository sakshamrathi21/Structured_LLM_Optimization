from nicegui import ui
import requests
import time
import pynvml
import json
from pathlib import Path

# -----------------------
# CONFIG
# -----------------------
LLM_PORT = 30000
LLM_URL = f"http://localhost:{LLM_PORT}/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# -----------------------
# FILE STORAGE FOR LATENCY
# -----------------------
LAT_FILE = Path("latencies.json")

def save_latency(ts, lat):
    """Append a latency record to file (keep last 100 entries)."""
    data = []
    if LAT_FILE.exists():
        try:
            data = json.loads(LAT_FILE.read_text())
        except json.JSONDecodeError:
            data = []
    data.append((ts, lat))
    data = data[-100:]  # keep last 100
    LAT_FILE.write_text(json.dumps(data))

def load_average_latency():
    """Load latency records from file."""
    if LAT_FILE.exists():
        try:
            timestamp_latencies = json.loads(LAT_FILE.read_text())
            sum_latencies = 0
            for _, latency in timestamp_latencies[-10:]:
                sum_latencies += latency
            return sum_latencies/len(timestamp_latencies[-10:])
        except json.JSONDecodeError:
            return []
    return []

# -----------------------
# GPU Monitor init
# -----------------------
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_stats():
    name = pynvml.nvmlDeviceGetName(handle).decode()
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
    average_latency =round(load_average_latency(), 2)
    return {
        "GPU Name": name,
        "GPU Utilization": f"{util.gpu}%",
        "Memory Used": f"{mem.used // (1024 ** 2)} MB",
        "Memory Total": f"{mem.total // (1024 ** 2)} MB",
        "Temperature": f"{temp} °C",
        "Power Usage": f"{power:.1f} W",
        "Fan Speed": f"{fan_speed} %",
        "Average Inference Latency": f"{average_latency} ms"
    }

# -----------------------
# ROUTE: Chat Page
# -----------------------
@ui.page("/chat")
def chat_ui():
    ui.label("🤖 LLM Chat Interface").classes("text-h4")
    chat_container = ui.column().classes("h-96 overflow-y-auto border p-2")
    user_input = ui.input("Type your message...").props('clearable')

    def render_message(role, content):
        prefix = "👤 You: " if role == "user" else "🤖 Bot: "
        with chat_container:
            ui.label(f"{prefix}{content}").classes("text-left w-full")

    def send_message():
        message = user_input.value.strip()
        if not message:
            return
        render_message("user", message)

        payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": message}]}

        start_time = time.perf_counter()
        try:
            response = requests.post(LLM_URL, json=payload)
            latency_ms = (time.perf_counter() - start_time) * 1000
            save_latency(time.strftime("%H:%M:%S"), latency_ms)

            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
                render_message("assistant", reply)
            else:
                render_message("assistant", f"❌ Error: {response.status_code}")
        except Exception as e:
            render_message("assistant", f"⚠️ Request failed: {str(e)}")

        user_input.value = ""

    user_input.on('keydown.enter', lambda e: send_message())
    ui.button("Send", on_click=send_message)

# -----------------------
# ROUTE: GPU Monitor Page
# -----------------------
@ui.page("/gpu")
def gpu_ui():
    ui.label("📊 GPU Monitoring Dashboard").classes("text-2xl font-bold mb-4")
    stats_labels = {}
    with ui.card().classes("w-96 p-4"):
        for key in get_gpu_stats():
            with ui.row().classes("justify-between w-full"):
                ui.label(f"{key}:").classes("font-medium")
                stats_labels[key] = ui.label("")

    def update_stats():
        stats = get_gpu_stats()
        for key, value in stats.items():
            stats_labels[key].text = value

    ui.timer(2.0, update_stats)

# -----------------------
# App object for uvicorn
# -----------------------


ui.run(port=8080, reload=False)
