from nicegui import ui
import pynvml

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Function to fetch GPU stats
def get_gpu_stats():
    name = pynvml.nvmlDeviceGetName(handle).decode()
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert mW to W
    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)

    return {
        "GPU Name": name,
        "GPU Utilization": f"{util.gpu}%",
        "Memory Used": f"{mem.used // (1024 ** 2)} MB",
        "Memory Total": f"{mem.total // (1024 ** 2)} MB",
        "Temperature": f"{temp} °C",
        "Power Usage": f"{power:.1f} W",
        "Fan Speed": f"{fan_speed} %"
    }

# --- UI Setup ---

ui.label("📊 GPU Monitoring Dashboard").classes("text-2xl font-bold mb-4")

stats_labels = {}  # dictionary to store label objects

with ui.card().classes("w-96 p-4"):
    for key in get_gpu_stats():
        with ui.row().classes("justify-between w-full"):
            ui.label(f"{key}:").classes("font-medium")
            stats_labels[key] = ui.label("")

# Function to refresh stats
def update_stats():
    stats = get_gpu_stats()
    for key, value in stats.items():
        stats_labels[key].text = value

ui.timer(2.0, update_stats)  # update every 2 seconds

# Run the app
ui.run()
