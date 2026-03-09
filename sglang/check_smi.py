import subprocess
import time
import pycupti.profiler as cupti

# Your sglang command
sglang_cmd = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", "qwen/qwen2.5-0.5b-instruct",
    "--mem-fraction-static", "0.8",
    "--disable-cuda-graph",
    "--host", "0.0.0.0",
    "--attention-backend", "flashinfer"
]

# Start the LLM server
proc = subprocess.Popen(sglang_cmd)

# CUPTI metric set
metrics = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_average_data",
    "tensor__throughput.avg.pct_of_peak_sustained_elapsed"
]

profiler = cupti.Profiler(metrics)

try:
    profiler.start()
    while proc.poll() is None:
        data = profiler.read()
        print("==== CUPTI Profiling ====")
        for m, v in data.items():
            print(f"{m}: {v}")
        time.sleep(2)  # sample interval
finally:
    profiler.stop()
    proc.terminate()


# import pynvml
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# print(pynvml.nvmlDeviceGetName(handle))
# print(pynvml.nvmlDeviceGetCudaComputeCapability(handle))


# import torch
# import torch.profiler as profiler
# with profiler.profile(
#     activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
#     profile_memory=True,
#     with_stack=True,
#     with_flops=True
# ) as prof:
#     run_your_model()
# print(prof.key_averages().table(sort_by="cuda_time_total"))

