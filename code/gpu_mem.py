import time
import threading
import pynvml

# Globals
_logging_thread = None
_stop_flag = False


def _logging_worker(log_file, interval):
    global _stop_flag
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    mem_list = []
    next_time = time.time()

    while not _stop_flag:
        ts = time.time()
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_used = mem_info.used / (1024**2)
        mem_list.append((ts, gpu_mem_used))

        # drift-corrected sleep
        next_time += interval
        delay = next_time - time.time()
        if delay > 0:
            time.sleep(delay)

    # dump all at once
    with open(log_file, "w") as f:
        f.write("timestamp,gpu_mem_used_MB\n")
        f.writelines(f"{ts:.6f},{mem:.2f}\n" for ts, mem in mem_list)

    pynvml.nvmlShutdown()


def start_logging(log_file="gpu_mem_log.txt", interval=0.01):
    """Start background logging of GPU memory only (non-blocking)."""
    global _logging_thread, _stop_flag
    if _logging_thread is not None:
        print("Logging already running.")
        return
    _stop_flag = False
    _logging_thread = threading.Thread(target=_logging_worker, args=(log_file, interval), daemon=True)
    _logging_thread.start()
    print(f"Logging started → {log_file}")


def stop_logging():
    """Stop background logging."""
    global _logging_thread, _stop_flag
    if _logging_thread is None:
        print("No active logging.")
        return
    _stop_flag = True
    _logging_thread.join()
    _logging_thread = None
    print("Logging stopped.")


if __name__ == "__main__":
    start_logging(interval=0.01)  # 100 Hz sampling
    time.sleep(5)                 # workload
    stop_logging()
