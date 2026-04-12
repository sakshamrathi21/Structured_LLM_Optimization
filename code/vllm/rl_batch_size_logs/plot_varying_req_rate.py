import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/rl_batch_size_logs")

LOG_FILES = [
    "rl_batching.log",
]

LABELS = [
    "Bandit",
]

PALETTE    = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
MARKERS    = ["o", "s", "D", "^"]
LINESTYLES = ["-", "--", "-.", ":"]

# ─────────────────────────────────────────────────────────────
# 🔥 USER CONFIG: RANGES TO REMOVE (in seconds)
REMOVE_RANGES = [
    (98, 148),
    (190, 235),
    (255, 290),
    (307, 350),
]

# ─────────────────────────────────────────────────────────────
# 🎨 Rate phases on the FINAL (stitched) time axis
# Adjust t_start / t_end after checking printed t=[..., ...] output
RATE_PHASES = [
    (0,   200, "5 req/s",  "#457B9D"),
    (200, 300, "10 req/s", "#E9C46A"),
    (300, 400, "20 req/s", "#E63946"),   # 400 is a sentinel; clips to data end
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


def remove_time_ranges(ts, bs, remove_ranges, max_time):
    if not remove_ranges:
        # Still remap to [0, max_time]
        final_ts = np.linspace(0, max_time, len(ts))
        return final_ts, bs

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

    # Remap to contiguous [0, max_time] — eliminates gap-connecting lines
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
    ts = ts - ts[0]  # normalize to 0

    all_series.append((label, color, marker, ls, ts, bs, duration))


# ── Min duration across all series ────────────────────────────
max_time = min(duration for *_, duration in all_series)

print("\nSeries durations (s):")
for (label, *_, duration) in all_series:
    print(f"  {label:25s}  {duration:.2f}")

print(f"\nTruncating all series to duration <= {max_time:.2f}s\n")


# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))

legend_handles = []

for label, color, marker, ls, ts, bs, duration in all_series:

    # Truncate to max_time
    mask = ts <= max_time
    ts = ts[mask]
    bs = bs[mask]

    # Remove + stitch
    ts, bs = remove_time_ranges(ts, bs, REMOVE_RANGES, max_time)

    print(f"  {label:25s}  points={len(ts)}  "
          f"t=[{ts[0]:.1f}, {ts[-1]:.1f}]  "
          f"batch=[{bs.min()}, {bs.max()}]")

    # Draw each phase segment in its own colour
    for (t_start, t_end, rate_label, phase_color) in RATE_PHASES:
        seg_mask = (ts >= t_start) & (ts <= t_end)
        seg_ts = ts[seg_mask]
        seg_bs = bs[seg_mask]

        if len(seg_ts) < 2:
            continue

        # Extend by one point on each side to avoid gaps at phase boundaries
        idx = np.where(seg_mask)[0]
        if idx[0] > 0:
            seg_ts = np.concatenate([[ts[idx[0] - 1]], seg_ts])
            seg_bs = np.concatenate([[bs[idx[0] - 1]], seg_bs])
        if idx[-1] < len(ts) - 1:
            seg_ts = np.concatenate([seg_ts, [ts[idx[-1] + 1]]])
            seg_bs = np.concatenate([seg_bs, [bs[idx[-1] + 1]]])

        ax.plot(
            seg_ts, seg_bs,
            color=phase_color,
            linestyle=ls,
            linewidth=1.8,
            marker=marker,
            markersize=4,
            markeredgecolor="black",
            markeredgewidth=0.4,
            alpha=0.85,
        )

# Build legend handles (one per rate phase)
for (_, _, rate_label, phase_color) in RATE_PHASES:
    legend_handles.append(
        mlines.Line2D([], [],
                      color=phase_color,
                      linewidth=2.5,
                      label=f"{rate_label}")
    )

# Vertical dashed lines at phase boundaries
for t_boundary in [200, 300]:
    ax.axvline(x=t_boundary, color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7)

ax.set_xlabel("Time (s)", fontsize=26)
ax.set_ylabel("Running Batch Size", fontsize=26)
ax.tick_params(axis="both", labelsize=20)
ax.grid(True, linewidth=0.6, linestyle="--")
ax.set_ylim(bottom=0)
ax.legend(handles=legend_handles, fontsize=25, loc="upper left")

plt.tight_layout()

OUT = os.path.expanduser(
    "~/CS496/results/rl_batch_size_plots/batch_size_over_time.png"
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {OUT}")