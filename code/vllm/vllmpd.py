import re
import matplotlib.pyplot as plt
import pandas as pd

# --- INPUT FILES ---
logfile = "out_infinite"  # LLM logs
gpulogfile = "gpu_metrics.csv"  # GPU/CPU metrics log from start_logging()

# --- REGEX PATTERNS FOR LLM LOG ---
gen_start_re = re.compile(r"Generation Start Time: ([\d.]+)")
add_re = re.compile(r"\[LOGGER\] Added request (\d+) at time ([\d.]+)")
prefill_re = re.compile(r"\[LOGGER\]\[Req (\d+)\].*Prefill time: ([\d.]+) sec")
token_re = re.compile(r"\[LOGGER\]\[Req (\d+)\].*latency: ([\d.]+) sec")

# --- PARSE LLM LOG ---
requests = {}
gen_start = None

with open(logfile, "r") as f:
    for line in f:
        if m := gen_start_re.search(line):
            gen_start = float(m.group(1))
        elif m := add_re.search(line):
            rid, t = int(m.group(1)), float(m.group(2))
            requests[rid] = {"add_time": t, "prefill": None, "tokens": []}
        elif m := prefill_re.search(line):
            rid, prefill = int(m.group(1)), float(m.group(2))
            if rid in requests:
                requests[rid]["prefill"] = prefill
        elif m := token_re.search(line):
            rid, latency = int(m.group(1)), float(m.group(2))
            if rid in requests:
                requests[rid]["tokens"].append(latency)

if gen_start is None:
    raise ValueError("No Generation Start Time found in log.")

# Normalize request times
for req in requests.values():
    req["add_time"] -= gen_start
    if req["prefill"] is not None:
        req["prefill"] -= gen_start
    req["tokens"] = [t - gen_start for t in req["tokens"]]

import pandas as pd

# --- PARSE GPU/CPU LOG ---
df = pd.read_csv(gpulogfile, na_values="-")  # treat "-" as NaN
df["timestamp"] = df["timestamp"] - gen_start  # align with gen_start

# Rename columns to match our log format if needed
df.rename(
    columns={"pcie_rx_MBps": "cpu_to_gpu_MBps", "pcie_tx_MBps": "gpu_to_cpu_MBps"},
    inplace=True,
)


# --- PLOTTING ---
fig, axes = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

# ---- (1) Request timeline ----
ax = axes[0]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
yticks, ylabels = [], []

for i, (rid, req) in enumerate(sorted(requests.items())):
    y = i * 10
    yticks.append(y)
    ylabels.append(f"Req {rid}")

    ax.scatter(
        req["add_time"],
        y,
        marker="*",
        color="black",
        s=100,
        label="Add time" if i == 0 else "",
    )
    ax.text(
        req["add_time"],
        y + 1.2,
        f"Add\n{req['add_time']:.3f}",
        fontsize=7,
        ha="center",
        color="black",
    )

    if req["prefill"] is not None:
        ax.scatter(
            req["prefill"],
            y,
            marker="s",
            color=colors[i % len(colors)],
            s=60,
            label="Prefill" if i == 0 else "",
        )
        ax.text(
            req["prefill"],
            y + 1.2,
            f"P\n{req['prefill']:.3f}",
            fontsize=7,
            ha="center",
            color=colors[i % len(colors)],
        )

    for idx, t in enumerate(req["tokens"]):
        ax.scatter(
            t,
            y,
            marker="o",
            color="gray",
            s=40,
            label="Token" if (i == 0 and idx == 0) else "",
        )

ax.set_ylabel("Requests")
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_title("LLM Request Timelines (normalized)")
ax.legend(loc="upper right")

# ---- (2) GPU Memory ----
mask = df["gpu_mem_used_MB"].notna()
axes[1].plot(
    df.loc[mask, "timestamp"],
    df.loc[mask, "gpu_mem_used_MB"],
    label="GPU Mem (MB)",
    color="tab:blue",
    marker="o",  # circle markers
    linestyle="-",  # keep line if you want; use "" for points only
)
axes[1].set_ylabel("GPU Mem (MB)")
axes[1].legend(loc="upper right")

# ---- (3) PCIe Bandwidth (CPU→GPU & GPU→CPU) ----
mask_rx = df["cpu_to_gpu_MBps"].notna()
axes[2].plot(
    df.loc[mask_rx, "timestamp"],
    df.loc[mask_rx, "cpu_to_gpu_MBps"],
    label="CPU→GPU BW",
    color="tab:green",
    marker="o",
    linestyle="-",
)

mask_tx = df["gpu_to_cpu_MBps"].notna()
axes[2].plot(
    df.loc[mask_tx, "timestamp"],
    df.loc[mask_tx, "gpu_to_cpu_MBps"],
    label="GPU→CPU BW",
    color="tab:red",
    marker="o",
    linestyle="-",
)

axes[2].set_ylabel("BW MB/s")
axes[2].legend(loc="upper right")


plt.tight_layout()
plt.savefig("try.png")
# plt.show()
