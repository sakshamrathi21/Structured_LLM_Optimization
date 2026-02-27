#!/usr/bin/env python3
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 16,           # Base font size (everything scales from this)
    "axes.titlesize": 20,      # Title font size
    "axes.labelsize": 18,      # X and Y label font size
    "xtick.labelsize": 16,     # X tick labels
    "ytick.labelsize": 16,     # Y tick labels
    "legend.fontsize": 12,     # Legend text
    "figure.titlesize": 22     # Figure title (if used)
})

log_dir = "."  # Change to your log directory

# ── Parse logs ────────────────────────────────────────────────────────────────
data = {}  # data[n_families][policy] = {batch_size, throughput}

for log_file in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    filename = os.path.basename(log_file)
    # Extract n_families and policy from filename, e.g. families_10_family_load.log
    m = re.match(r"families_(\d+)_([\w]+(?:_[\w]+)*)\.log", filename)
    if not m:
        continue
    n_families = int(m.group(1))
    policy = m.group(2)

    batch_size = throughput = None
    with open(log_file) as f:
        for line in f:
            bs = re.search(r"current batch size:\s*([\d.]+)", line)
            if bs:
                batch_size = float(bs.group(1))
            tp = re.search(r"Average throughput\s*:\s*([\d.]+)\s*req/s", line)
            if tp:
                throughput = float(tp.group(1))

    data.setdefault(n_families, {})[policy] = {
        "batch_size": batch_size,
        "throughput": throughput,
    }

family_counts = sorted(data.keys())
policies      = ["default", "family_load"]
colors        = {"default": "#4C72B0", "family_load": "#DD8452"}
labels        = {"default": "FCFS", "family_load": "Batches grouped by prefixes"}

# ── Only default policy ───────────────────────────────────────────────────────
# ── Only default policy ───────────────────────────────────────────────────────
policy = "default"

throughput_vals = [data[n].get(policy, {}).get("throughput") for n in family_counts]
batch_vals      = [data[n].get(policy, {}).get("batch_size") for n in family_counts]

x = np.arange(len(family_counts))
width = 0.35

# ── Dual Y-Axis Grouped Bar Plot ──────────────────────────────────────────────
# ── Dual Y-Axis Grouped Bar Plot ──────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis → Throughput bars
bars1 = ax1.bar(
    x - width/2,
    throughput_vals,
    width,
    label="Throughput (req/s)",
    color="#1f77b4",      # <-- CHANGE COLOR HERE
    alpha=0.85
)

ax1.set_xlabel("Number of Families")
ax1.set_ylabel("Throughput (req/s)")
ax1.set_xticks(x)
ax1.set_xticklabels([str(n) for n in family_counts])
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Right axis → Batch size bars
ax2 = ax1.twinx()
bars2 = ax2.bar(
    x + width/2,
    batch_vals,
    width,
    label="Average Batch Size",
    color="#d62728",      # <-- CHANGE COLOR HERE
    alpha=0.85
)

ax2.set_ylabel("Average Batch Size")

# ── Combined Legend ───────────────────────────────────────────────────────────
handles = [bars1, bars2]
labels  = ["Throughput (req/s)", "Average Batch Size"]
ax1.legend(handles, labels, loc="best")

plt.tight_layout()
plt.savefig("../../../results/hbm_evictions_fcfs_throughput_vs_batchsize_dual_axis.png")
plt.close()