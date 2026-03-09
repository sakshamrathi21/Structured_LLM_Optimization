import subprocess
import time
import psutil
import os

MODEL = "qwen/qwen2.5-0.5b-instruct"
SERVER_CMD = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL,
    "--disable-radix-cache"
]

# Metrics you want
METRICS = "l2_tex_hit_rate,l2_tex_read_hit_rate,l2_tex_write_hit_rate"
NREP_FILE = "sglang_worker_profile"

def find_worker_pids(server_pid):
    """Finds all Python worker PIDs spawned by the sglang server."""
    pids = []
    try:
        parent = psutil.Process(server_pid)
        for child in parent.children(recursive=True):
            if "python" in child.name():
                pids.append(child.pid)
    except psutil.NoSuchProcess:
        pass
    return pids

if __name__ == "__main__":
    # 1. Start server
    print(">>> Starting sglang server...")
    server = subprocess.Popen(SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # 2. Wait for workers to appear
    worker_pid = None
    print(">>> Waiting for worker processes...")
    for _ in range(60):  # wait up to 60s
        pids = find_worker_pids(server.pid)
        if pids:
            worker_pid = pids[0]  # take the first worker
            break
        time.sleep(2)

    if not worker_pid:
        print("!!! No worker process found. Exiting.")
        server.terminate()
        exit(1)

    print(f">>> Found worker PID: {worker_pid}")

    # 3. Run ncu attached to worker
    ncu_cmd = [
        "ncu",
        "--target-processes", "application",
        "--metrics", METRICS,
        "--set", "full",
        "-o", NREP_FILE,
        "--pid", str(worker_pid)
    ]
    print(f">>> Running Nsight Compute on worker PID {worker_pid}...")
    ncu_proc = subprocess.Popen(ncu_cmd)

    # 4. Give ncu a moment to attach
    time.sleep(5)

    # 5. Launch client
    print(">>> Launching client requests...")
    subprocess.run(["python3", "req.py"])

    # 6. Give time for kernels to finish
    time.sleep(10)

    # 7. Stop profiling
    ncu_proc.terminate()
    print(">>> Profiling complete. Output:", NREP_FILE + ".ncu-rep")

    # 8. Stop server
    server.terminate()


# >>> Starting sglang server...
# >>> Waiting for worker processes...
# >>> Found worker PID: 64969
# >>> Running Nsight Compute on worker PID 64969...
# unrecognised option '--pid'