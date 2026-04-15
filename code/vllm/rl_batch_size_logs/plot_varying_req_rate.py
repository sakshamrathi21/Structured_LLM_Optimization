import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/rl_batch_size_logs")
LOG_FILE = "rl_batching.log"

REMOVE_RANGES = [
    (98, 148),
    (190, 235),
    (255, 290),
    (307, 350),
]

RATE_PHASES = [
    (0,   200, "5 req/s",  "#457B9D", "-",  "o"),
    (200, 312, "10 req/s", "#E9C46A", "--", "s"),
    (312, 400, "20 req/s", "#E63946", "-.", "D"),
]

RE_SCHEDULER = re.compile(
    r"Scheduler stats;.*?running batch size:\s*(\d+),.*?current time:\s*([\d.]+)"
)


def parse_log(path):
    batch_sizes, timestamps = [], []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_SCHEDULER.search(line)
            if m:
                batch_sizes.append(int(m.group(1)))
                timestamps.append(float(m.group(2)))
    return np.array(timestamps), np.array(batch_sizes)


def remove_time_ranges(ts, bs, remove_ranges):
    remove_ranges = sorted(remove_ranges, key=lambda x: x[0])
    result_ts, result_bs = [], []
    prev_end = ts[0]

    for (cut_start, cut_end) in remove_ranges + [(ts[-1] + 1, ts[-1] + 1)]:
        mask = (ts >= prev_end) & (ts <= cut_start)
        if mask.any():
            result_ts.append(ts[mask])
            result_bs.append(bs[mask])
        prev_end = cut_end

    final_ts = np.concatenate(result_ts)
    final_bs = np.concatenate(result_bs)
    final_ts = np.linspace(0, ts[-1] - ts[0], len(final_ts))
    return final_ts, final_bs


# ── Parse ─────────────────────────────────────────────────────
path = os.path.join(LOG_DIR, LOG_FILE)
ts, bs = parse_log(path)
ts -= ts[0]
ts, bs = remove_time_ranges(ts, bs, REMOVE_RANGES)
max_time = ts[-1]

print(f"points={len(ts)}  t=[{ts[0]:.1f}, {ts[-1]:.1f}]  batch=[{bs.min()}, {bs.max()}]")

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

for (t_start, t_end, rate_label, color, ls, marker) in RATE_PHASES:
    seg_mask = (ts >= t_start) & (ts <= t_end)
    seg_ts = ts[seg_mask]
    seg_bs = bs[seg_mask]

    if len(seg_ts) < 2:
        continue

    idx = np.where(seg_mask)[0]
    if idx[0] > 0:
        seg_ts = np.concatenate([[ts[idx[0] - 1]], seg_ts])
        seg_bs = np.concatenate([[bs[idx[0] - 1]], seg_bs])
    if idx[-1] < len(ts) - 1:
        seg_ts = np.concatenate([seg_ts, [ts[idx[-1] + 1]]])
        seg_bs = np.concatenate([seg_bs, [bs[idx[-1] + 1]]])

    ax.plot(seg_ts, seg_bs, color=color, linestyle=ls, linewidth=5,
            marker=marker, markersize=7, markeredgecolor="black",
            markeredgewidth=0.4, alpha=0.85)

for t_boundary in [200, 312]:
    ax.axvline(x=t_boundary, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

legend_handles = [
    mlines.Line2D([], [], color=color, linestyle=ls, marker=marker,
                  markersize=18, linewidth=5, label=rate_label)
    for (_, _, rate_label, color, ls, marker) in RATE_PHASES
]

ax.set_xlabel("Time (s)", fontsize=35)
ax.set_ylabel("Running Batch Size", fontsize=35)
ax.tick_params(axis="both", labelsize=30)
ax.grid(True, linewidth=0.6, linestyle="--")
ax.set_ylim(bottom=0)
ax.legend(handles=legend_handles, fontsize=30, loc="upper left")

plt.tight_layout()

OUT = os.path.expanduser("~/CS496/results/rl_batch_size_plots/batch_size_over_time.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT}")