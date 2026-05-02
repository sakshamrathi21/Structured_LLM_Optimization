#!/usr/bin/env python3
"""
Parse req_per_sec_logs/ and generate a single figure with 2 plots stacked
vertically:
  1. (top)    Throughput vs RPS
  2. (bottom) Avg Running Batch Size vs RPS

A single shared legend is placed to the right of both subplots.
SGLang:Feather hardcoded data is always included.
"""

import os
import re
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/req_per_sec_logs")

# ── Hardcoded SGLang:Feather data ────────────────────────────────────────────
FEATHER_SGLANG_THROUGHPUT = {
    1:   201.21,
    2:   401.51,
    5:   991.28,
    10: 1876.09,
    20: 2336.57,
    50: 2381.43,
    100: 2356.28,
}

FEATHER_SGLANG_AVG_BATCH = {
    1:   2.39888,
    2:   5.05219,
    5:   16.1344,
    10:  55.5741,
    20:  214.357,
    50:  250.25,
    100: 250.25,
}

# ── Regex patterns ────────────────────────────────────────────────────────────
RE_SCHEDULER = re.compile(
    r"Scheduler stats;.*?running batch size:\s*(\d+),.*?current time:\s*([\d.]+)"
)
RE_VLLM_THROUGHPUT = re.compile(
    r"Throughput Test.*?:\s*([\d.]+)\s*$"
)
RE_SGLANG_THROUGHPUT = re.compile(
    r"Throughput:\s*([\d.]+)\s*toks/s"
)
RE_FILENAME = re.compile(
    r"^(?P<engine>vllm|sglang)_(?P<policy>.+?)_rps(?P<rps>\d+)_"
)


def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("rps"))


def parse_log(path):
    batch_sizes = []
    timestamps  = []
    throughput  = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_SCHEDULER.search(line)
            if m:
                batch_sizes.append(int(m.group(1)))
                timestamps.append(float(m.group(2)))
                continue
            m = RE_VLLM_THROUGHPUT.search(line)
            if m:
                throughput = float(m.group(1))
                continue
            m = RE_SGLANG_THROUGHPUT.search(line)
            if m:
                throughput = float(m.group(1))

    avg_batch = np.mean(batch_sizes) if batch_sizes else None
    avg_diff  = None
    if len(timestamps) >= 2:
        avg_diff = np.mean(np.diff(sorted(timestamps)))

    return avg_batch, avg_diff, throughput


# ── Collect data ──────────────────────────────────────────────────────────────
data = defaultdict(lambda: defaultdict(dict))

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"  SKIP (unrecognised name): {os.path.basename(path)}")
        continue

    engine, policy, rps = parsed

    engine = {"vllm": "vLLM", "sglang": "SGLang"}.get(engine, engine)

    policy_map = {
        "dfs-weight":               "DFS-W",
        "lpm":                      "LPM",
        "dynamic_batching":         "DynBatch",
        "fcfs":                     "FCFS",
        "chunked_hash_tree_bandit": "Feather",
    }
    if policy == "chunked_hash_tree_heuristic":
        continue
    policy = policy_map.get(policy, policy)

    label = f"{engine}:{policy}"

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"  {label:45s} rps={rps:4d}  "
        f"tput={throughput}  batch={avg_batch}  dt={avg_diff}"
    )

    existing_tput = (data[label][rps].get("throughput") or 0) if rps in data[label] else 0
    new_tput = throughput or 0
    if new_tput <= existing_tput:
        continue

    data[label][rps] = dict(
        throughput=throughput,
        avg_batch=avg_batch,
        avg_diff=avg_diff,
    )

# ── Inject hardcoded SGLang:Feather data ─────────────────────────────────────
print("\nInjecting hardcoded SGLang:Feather data...")
for rps, tput in FEATHER_SGLANG_THROUGHPUT.items():
    data["SGLang:Feather"][rps]["throughput"] = tput
    data["SGLang:Feather"][rps]["avg_batch"]  = FEATHER_SGLANG_AVG_BATCH.get(rps)
    print(
        f"  SGLang:Feather  rps={rps:4d}  "
        f"tput={tput}  batch={FEATHER_SGLANG_AVG_BATCH.get(rps)}"
    )

# ── Plot styling ──────────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]
MARKERS    = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", "--", ":", ":", "-", "-", "--", ":", "-"]

# Consistent label ordering: everything alphabetically, SGLang:Feather last
policies = (
    [l for l in sorted(data.keys()) if l != "SGLang:Feather"]
    + (["SGLang:Feather"] if "SGLang:Feather" in data else [])
)

# ── Build figure: 2 rows, 1 col + right-hand legend column ───────────────────
#   We use gridspec with an extra narrow column that we hide (the legend sits
#   outside the axes via bbox_to_anchor, so no extra column is needed – we
#   just leave enough right margin).
fig, axes = plt.subplots(
    nrows=2, ncols=1,
    figsize=(16, 12),
    sharex=True,
)

ax_tp, ax_batch = axes

METRICS = [
    ("throughput", "TP (toks/s)",    ax_tp),
    ("avg_batch",  "Avg Batch Size", ax_batch),
]

y_formatter_tp = ticker.FuncFormatter(
    lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}"
)
y_formatter_batch = ticker.FuncFormatter(
    lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else (
        f"{x:.2f}" if 0 < x < 1 else f"{x:.0f}"
    )
)

line_handles = []
line_labels  = []

for i, label in enumerate(policies):
    color     = PALETTE[i % len(PALETTE)]
    marker    = MARKERS[i % len(MARKERS)]
    linestyle = LINESTYLES[i % len(LINESTYLES)]

    rps_vals = sorted(data[label].keys())

    for metric_key, ylabel, ax in METRICS:
        xs, ys = [], []
        for rps in rps_vals:
            v = data[label][rps].get(metric_key)
            if v is not None:
                xs.append(rps)
                ys.append(round(v, 4) if isinstance(v, float) else v)

        if xs:
            line, = ax.plot(
                xs, ys,
                marker=marker,
                linestyle=linestyle,
                linewidth=4,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=0.7,
                color=color,
                label=label,
            )
            # Collect handle/label only once (from the first axis)
            if ax is ax_tp:
                line_handles.append(line)
                line_labels.append(label)

# ── Axis formatting ───────────────────────────────────────────────────────────
for ax in axes:
    ax.grid(True, linewidth=0.7, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=28)

ax_tp.set_ylabel("TP (toks/s)",    fontsize=30)
ax_batch.set_ylabel("Avg Batch Size", fontsize=30)
ax_batch.set_xlabel("Input Workload (req/s)", fontsize=30)

ax_tp.yaxis.set_major_formatter(y_formatter_tp)
ax_batch.yaxis.set_major_formatter(y_formatter_batch)

# Vertical marker + annotation on throughput subplot
ax_tp.axvline(x=5, color='black', linestyle=':', linewidth=2)
ylim_tp = ax_tp.get_ylim()
ax_tp.annotate(
    "Homogeneity\nstarts here",
    xy=(5, ylim_tp[1] * 0.85),
    xytext=(9, ylim_tp[1] * 0.85),
    fontsize=24,
    color='black',
    arrowprops=dict(arrowstyle="->", color='black', lw=1.8),
    ha='left',
    va='center',
)

# ── Shared legend to the right of both subplots ───────────────────────────────
fig.legend(
    line_handles,
    line_labels,
    fontsize=30,
    framealpha=0.15,
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),   # just outside the right edge, vertically centred
    borderaxespad=0,
    ncol=1,
)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
OUT = os.path.expanduser("../../../results/req_per_sec_plots")
os.makedirs(OUT, exist_ok=True)
outpath = os.path.join(OUT, "stacked_tp_batch.png")

fig.savefig(
    outpath,
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close(fig)

print(f"\nDone! Saved stacked plot to: {outpath}")