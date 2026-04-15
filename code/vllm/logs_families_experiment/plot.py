#!/usr/bin/env python3
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 22
})

RE_SCHEDULER = re.compile(
    r"Scheduler stats;.*?running batch size:\s*(\d+),.*?current time:\s*([\d.]+)"
)

RE_VLLM_THROUGHPUT = re.compile(r"Average throughput\s*:\s*([0-9.]+)")

log_dir = "."

data = {}

for log_file in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    filename = os.path.basename(log_file)
    m = re.match(r"families_(\d+)_default.log", filename)
    print(filename, m)
    if not m:
        continue
    n_families = int(m.group(1))
    policy = "default"

    batch_sizes = []
    throughput = None
    with open(log_file) as f:
        for line in f:
            m = RE_SCHEDULER.search(line)
            if m:
                batch_sizes.append(int(m.group(1)))
                continue
            m = RE_VLLM_THROUGHPUT.search(line)
            if m:
                throughput = float(m.group(1))
                continue

    data.setdefault(n_families, {})[policy] = {
        "batch_size": sum(batch_sizes) / len(batch_sizes),
        "throughput": throughput,
    }

family_counts = sorted(data.keys())
policy = "default"

throughput_vals = [data[n].get(policy, {}).get("throughput") for n in family_counts]
batch_vals      = [data[n].get(policy, {}).get("batch_size") for n in family_counts]

x = np.arange(len(family_counts))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis → Throughput bars
bars1 = ax1.bar(
    x - width / 2,
    throughput_vals,
    width,
    label="TP (toks/s)",
    color="#1f77b4",
    alpha=0.85,
    hatch="//",
    edgecolor="black"
)

ax1.set_xlabel("Prefix Groups", fontsize=30)
ax1.set_ylabel("TP (toks/s)", fontsize=30)
ax1.set_xticks(x)
ax1.set_xticklabels([str(n) for n in family_counts])
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.tick_params(axis="x", labelsize=30)
# Right axis → Batch size bars
ax2 = ax1.twinx()
bars2 = ax2.bar(
    x + width / 2,
    batch_vals,
    width,
    label="Avg Batch Size",
    color="#d62728",
    alpha=0.85,
    hatch="xx",
    edgecolor="black"
)

ax2.set_ylabel("Avg Batch Size", fontsize=30)

# ── Legend placed INSIDE the plot at upper right ──────────────────────────────
handles = [bars1, bars2]
legend_labels = ["TP (toks/s)", "Average Batch Size"]

ax1.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=True,
    framealpha=0.8,
    fontsize=30,
    columnspacing=0.1
)

# Leave space at the top for legend
plt.tight_layout(rect=[0, 0, 1, 0.85])

plt.savefig(
    "../../../results/hbm_evictions_fcfs_throughput_vs_batchsize_dual_axis.png",
    bbox_inches="tight"
)
plt.close()
print("Plot saved to:", "../../../results/hbm_evictions_fcfs_throughput_vs_batchsize_dual_axis.png")