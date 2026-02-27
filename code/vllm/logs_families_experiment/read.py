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

# ── Helper to extract metric arrays ──────────────────────────────────────────
def get_values(metric):
    return {
        p: [data[n].get(p, {}).get(metric) for n in family_counts]
        for p in policies
    }

x       = np.arange(len(family_counts))
width   = 0.35
x_ticks = [str(n) for n in family_counts]

# ── Plot factory ──────────────────────────────────────────────────────────────
def make_plots(metric, ylabel, title_suffix, filename_suffix):
    vals = get_values(metric)

    # --- Grouped bar chart ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(policies):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals[p], width, label=labels[p], color=colors[p], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("Number of Families", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"../../../results/hbm_evictions_bar_{filename_suffix}.png", dpi=150)
    plt.close()
    print(f"Saved bar_{filename_suffix}.png")

# ── Generate plots ────────────────────────────────────────────────────────────
make_plots("throughput", "Throughput (req/s)", "Throughput vs Number of Families", "throughput")
make_plots("batch_size", "Average Batch Size",  "Batch Size vs Number of Families",  "batch_size")