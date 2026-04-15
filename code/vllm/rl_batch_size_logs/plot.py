#!/usr/bin/env python3
"""
Plot current running batch size as time passes for 4 log files,
all on a single figure. Truncates all series to the minimum final
timestamp found across all logs (i.e., aligned by actual time).
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/rl_batch_size_logs")
LOG_FILES = [
    "vllm_multiple_families_varying_workloads_5_families_5_rps_chunked_hash_tree_bandit.log",
    # "vllm_multiple_families_varying_workloads_5_families_5_rps_chunked_hash_tree_heuristic.log",
    # "vllm_multiple_families_varying_workloads_5_families_500_rps_chunked_hash_tree_bandit.log",
    # "vllm_multiple_families_varying_workloads_5_families_500_rps_chunked_hash_tree_qlearning.log",
]

LABELS = [
    "Bandit",
    "Q-Learning",
    "Bandit",
    "Q-Learning",
]

PALETTE    = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
MARKERS    = ["o", "s", "D", "^"]
LINESTYLES = ["-", "--", "-.", ":"]

RE_SCHEDULER = re.compile(
    r"Scheduler stats;.*?running batch size:\s*(\d+),.*?current time:\s*([\d.]+)"
)

def parse_log(path):
    batch_sizes = []
    timestamps  = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_SCHEDULER.search(line)
            if m:
                batch_sizes.append(int(m.group(1)))
                timestamps.append(float(m.group(2)))
    return np.array(timestamps), np.array(batch_sizes)

# ── Parse all logs first ─────────────────────────────────────────────────────
# ── Parse all logs first ─────────────────────────────────────────────────────
all_series = []

for fname, label, color, marker, ls in zip(
    LOG_FILES, LABELS, PALETTE, MARKERS, LINESTYLES
):
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        print(f"  MISSING: {fname}")
        continue
    ts, bs = parse_log(path)
    if len(ts) == 0:
        print(f"  NO DATA: {fname}")
        continue
    duration = ts[-1] - ts[0]          # real elapsed time, before normalisation
    ts = ts - ts[0]                     # normalise to start at 0
    all_series.append((label, color, marker, ls, ts, bs, duration))

# ── Min duration across all series ───────────────────────────────────────────
max_time = min(duration for *_, duration in all_series)
durations = [duration for *_, duration in all_series]
print("\nSeries durations (s):")
for (label, *_, duration) in all_series:
    print(f"  {label:25s}  {duration:.2f}")
# max_time = 200
print(f"\nTruncating all series to duration <= {max_time:.2f}s\n")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

for label, color, marker, ls, ts, bs, duration in all_series:
    mask = ts <= max_time
    ts = ts[mask]
    bs = bs[mask]

    print(f"  {label:25s}  points={len(ts)}  "
          f"t=[{ts[0]:.1f}, {ts[-1]:.1f}]  "
          f"batch=[{bs.min()}, {bs.max()}]")

    ax.plot(
        ts, bs,
        label=label,
        color=color,
        linestyle=ls,
        linewidth=5,
        marker=marker,
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.4,
        alpha=0.85,
    )

ax.set_xlabel("Time (s)", fontsize=35)
ax.set_ylabel("Running Batch Size", fontsize=35)
ax.tick_params(axis="both", labelsize=30)
# ax.legend(fontsize=25, framealpha=0.35)
ax.grid(True, linewidth=0.6, linestyle="--")
ax.set_ylim(bottom=0)
plt.tight_layout()

OUT = os.path.expanduser("~/CS496/results/rl_batch_size_plots/rl_policies_comparison.png")
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT}")