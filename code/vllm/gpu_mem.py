import time
import multiprocessing as mp
import pynvml
import csv

# Globals
_processes = {}
_queues = {}
_stop_event = None
_manager = None
_interval = 0.01
_log_file = "gpu_metrics.csv"


def _gpu_logger(metric, interval, queue, stop_event):
    """Worker for logging a single metric into its queue."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    next_time = time.time()
    while not stop_event.is_set():
        ts = time.time()

        # ---- MEMORY USAGE ----
        if metric == "mem":
            val = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)  # MB

        # ---- PCIe Throughput ----
        elif metric == "pcie_rx":
            try:
                val = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                val = 0.0
        elif metric == "pcie_tx":
            try:
                val = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
            except pynvml.NVMLError:
                val = 0.0

        # ---- GPU + Memory Controller Utilization ----
        elif metric == "gpu_util":
            print(pynvml.nvmlDeviceGetUtilizationRates(handle))
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                val = util.gpu  # % of time GPU SMs are active
            except pynvml.NVMLError:
                val = 0.0
        elif metric == "mem_util":
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                val = util.memory  # % of time memory controller is active
            except pynvml.NVMLError:
                val = 0.0

        # ---- GPU Power Draw (W) ----
        elif metric == "power":
            try:
                val = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
            except pynvml.NVMLError:
                val = 0.0

        else:
            val = None

        queue.put((metric, ts, val))

        next_time += interval
        delay = next_time - time.time()
        if delay > 0:
            time.sleep(delay)

    pynvml.nvmlShutdown()


def _merge_and_write(queues, csv_file):
    """Merge results from multiple queues and align by timestamp."""
    data = {m: {} for m in queues}

    for metric, q in queues.items():
        while not q.empty():
            _, ts, val = q.get()
            data[metric][round(ts, 3)] = val  # round to ms for alignment

    all_ts = sorted(set().union(*[data[m].keys() for m in data]))

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "gpu_mem_used_MB",
                "pcie_rx_MBps",
                "pcie_tx_MBps",
                "gpu_util_percent",
                "mem_util_percent",
                "power_W",
            ]
        )
        for ts in all_ts:
            row = [
                f"{ts:.3f}",
                f"{data['mem'].get(ts, '-')}",
                f"{data['pcie_rx'].get(ts, '-')}",
                f"{data['pcie_tx'].get(ts, '-')}",
                f"{data['gpu_util'].get(ts, '-')}",
                f"{data['mem_util'].get(ts, '-')}",
                f"{data['power'].get(ts, '-')}",
            ]
            writer.writerow(row)


def start_logging(log_file="gpu_metrics.csv", interval=0.01):
    """Start logging processes for multiple GPU metrics."""
    global _processes, _queues, _stop_event, _manager, _interval, _log_file

    if _processes:
        print("Logging already running.")
        return

    _interval = interval
    _log_file = log_file
    _manager = mp.Manager()
    _stop_event = mp.Event()

    metrics = ["mem", "pcie_rx", "pcie_tx", "gpu_util", "mem_util", "power"]

    _queues = {m: _manager.Queue() for m in metrics}

    _processes = {
        m: mp.Process(target=_gpu_logger, args=(m, _interval, _queues[m], _stop_event))
        for m in _queues
    }

    for p in _processes.values():
        p.start()

    print(f"Logging started → {log_file}")


def stop_logging():
    """Stop all logging processes and merge results into a CSV."""
    global _processes, _queues, _stop_event, _manager, _log_file

    if not _processes:
        print("No active logging.")
        return

    _stop_event.set()
    for p in _processes.values():
        p.join()

    print("Processes stopped. Writing CSV...")
    _merge_and_write(_queues, _log_file)

    # cleanup
    _processes = {}
    _queues = {}
    _stop_event = None
    _manager = None

    print(f"Done → {_log_file}")


if __name__ == "__main__":
    start_logging(interval=0.01)
    time.sleep(10)  # workload here
    stop_logging()
