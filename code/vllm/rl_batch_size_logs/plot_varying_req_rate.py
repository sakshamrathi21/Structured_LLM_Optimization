#!/usr/bin/env python3
"""
Plot current running batch size as time passes for log files.

NEW FEATURE:
- Remove specified time ranges and smoothly connect remaining segments.
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/rl_batch_size_logs")

LOG_FILES = [
    "rl_batching.log",
]

LABELS = [
    "Bandit",
    "Q-Learning",
    "Bandit1",
    "Q-Learning",
]

PALETTE    = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
MARKERS    = ["o", "s", "D", "^"]
LINESTYLES = ["-", "--", "-.", ":"]

# ─────────────────────────────────────────────────────────────
# 🔥 USER CONFIG: RANGES TO REMOVE (in seconds)
# Example: remove time from 20–40s and 80–100s
REMOVE_RANGES = [
    # (start, end),
    (98, 148),
    (190, 235),
    (255, 290),
    (307, 350),
]
# ─────────────────────────────────────────────────────────────

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


def remove_time_ranges(ts, bs, remove_ranges):
    if not remove_ranges:
        return ts, bs

    remove_ranges = sorted(remove_ranges, key=lambda x: x[0])

    orig_ts = ts.copy()
    orig_bs = bs.copy()

    result_ts = []
    result_bs = []

    prev_end = orig_ts[0]

    ranges_with_sentinel = remove_ranges + [(orig_ts[-1] + 1, orig_ts[-1] + 1)]

    for (cut_start, cut_end) in ranges_with_sentinel:
        seg_mask = (orig_ts >= prev_end) & (orig_ts <= cut_start)
        seg_ts   = orig_ts[seg_mask]
        seg_bs   = orig_bs[seg_mask]

        if len(seg_ts) > 0:
            result_ts.append(seg_ts)
            result_bs.append(seg_bs)

        prev_end = cut_end

    final_ts = np.concatenate(result_ts)
    final_bs = np.concatenate(result_bs)

    # ── Shift timestamps after all cutting is done ────────────
    # Close the gaps by making each segment immediately follow the previous one.
    # ── Shift timestamps after all cutting is done ────────────
    final_ts = np.concatenate(result_ts)
    final_bs = np.concatenate(result_bs)

    # Just reassign contiguous timestamps — kills all connecting lines
    final_ts = np.linspace(0, max_time, len(final_ts))

    return final_ts, final_bs


# ── Parse all logs ────────────────────────────────────────────
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

    duration = ts[-1] - ts[0]

    ts = ts - ts[0]  # normalize

    all_series.append((label, color, marker, ls, ts, bs, duration))


# ── Min duration across all series ────────────────────────────
max_time = min(duration for *_, duration in all_series)

print("\nSeries durations (s):")
for (label, *_, duration) in all_series:
    print(f"  {label:25s}  {duration:.2f}")

print(f"\nTruncating all series to duration <= {max_time:.2f}s\n")


# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

for label, color, marker, ls, ts, bs, duration in all_series:

    # Truncate
    mask = ts <= max_time
    ts = ts[mask]
    bs = bs[mask]

    # 🔥 REMOVE + STITCH
    ts, bs = remove_time_ranges(ts, bs, REMOVE_RANGES)

    print(f"  {label:25s}  points={len(ts)}  "
          f"t=[{ts[0]:.1f}, {ts[-1]:.1f}]  "
          f"batch=[{bs.min()}, {bs.max()}]")

    ax.plot(
        ts, bs,
        label=label,
        color=color,
        linestyle=ls,
        linewidth=1.8,
        marker=marker,
        markersize=4,
        markeredgecolor="black",
        markeredgewidth=0.4,
        alpha=0.85,
    )

ax.set_xlabel("Time (s)", fontsize=26)
ax.set_ylabel("Running Batch Size", fontsize=26)
ax.tick_params(axis="both", labelsize=20)
ax.grid(True, linewidth=0.6, linestyle="--")
ax.set_ylim(bottom=0)

plt.tight_layout()

OUT = os.path.expanduser(
    "~/CS496/results/rl_batch_size_plots/batch_size_over_time.png"
)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {OUT}")