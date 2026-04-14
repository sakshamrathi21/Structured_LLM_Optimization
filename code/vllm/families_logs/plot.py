#!/usr/bin/env python3
"""
Parse families_logs/ and generate 3 plots:
  1. Throughput vs Prefix Groups
  2. Avg Running Batch Size vs Prefix Groups
  3. Avg Consecutive Time Difference vs Prefix Groups
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

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/families_logs")

# ── Regex patterns ──────────────────────────────────────────────────────────
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
    r"^(?P<engine>vllm|sglang)_(?P<policy>.*?)_req.*?_fam(?P<fam>\d+)"
)


# ── Parse filename ──────────────────────────────────────────────────────────
def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("fam"))


# ── Parse log contents ──────────────────────────────────────────────────────
def parse_log(path):

    batch_sizes = []
    timestamps = []
    throughput = None

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

    if len(timestamps) >= 2:
        diffs = np.diff(sorted(timestamps))
        avg_diff = np.mean(diffs)
    else:
        avg_diff = None

    return avg_batch, avg_diff, throughput


# ── Collect data ────────────────────────────────────────────────────────────
data = defaultdict(lambda: defaultdict(dict))

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:

    parsed = parse_filename(path)

    if parsed is None:
        print(f"  SKIP (unrecognised name): {os.path.basename(path)}")
        continue

    engine, policy, fam = parsed
    if engine == "vllm":
        engine = "vLLM"
    if engine == "sglang":
        engine = "SGLang"
    if policy == "dfs-weight":
        policy = "DFS-W"
    if policy == "lpm":
        policy = "LPM"
    if policy == "fcfs":
        policy = "FCFS"
    if policy == "dynamic_batching":
        policy = "DynBatch"
    if policy == "chunked_hash_tree_bandit":
        policy = "Feather"
    if policy == "chunked_hash_tree_heuristic":
        continue
    label = f"{engine}:{policy}"

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"  {label:45s} fam={fam:4d}  "
        f"tput={throughput}  batch={avg_batch}  "
        f"dt={avg_diff}"
    )

    if label in data and fam in data[label]:
        existing_tput = data[label][fam].get("throughput") or 0
        new_tput = throughput or 0
        if new_tput <= existing_tput:
            continue

    data[label][fam] = dict(
        throughput=throughput,
        avg_batch=avg_batch,
        avg_diff=avg_diff
    )


# ── Plot styling (color-blind friendly) ─────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", "--", ":", ":", "-"]

all_fams = sorted({fam for policy in data for fam in data[policy]})


# ── Plot function ───────────────────────────────────────────────────────────
def make_plot(metric_key, ylabel, title, outpath):

    fig, ax = plt.subplots(figsize=(12, 6))

    policies = sorted(data.keys())

    for i, label in enumerate(policies):

        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        fam_vals = sorted(data[label].keys())
        xs, ys = [], []

        for fam in fam_vals:

            v = data[label][fam].get(metric_key)

            if v is not None:
                xs.append(fam)

                if isinstance(v, float):
                    v = round(v, 4)

                ys.append(v)

        if xs:
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                linewidth=5,
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=0.7,
                color=color,
                label=label
            )

    ax.set_xscale("log")
    ax.set_xticks(all_fams)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Prefix Groups", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    # y axis convert 1000 to 1K
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.tick_params(axis="both", labelsize=30)

    if metric_key == "throughput":
        ax.legend(fontsize=22, framealpha=0.1, ncol=2, columnspacing=0.1, bbox_to_anchor=(0.7, 0.7), loc="upper center")

    ax.grid(True, linewidth=0.7, linestyle="--")
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    fig.savefig(
        outpath,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(f"Saved: {outpath}")


# ── Generate plots ──────────────────────────────────────────────────────────
OUT = os.path.expanduser("../../../results/families_plots")
os.makedirs(OUT, exist_ok=True)

make_plot(
    "throughput",
    "TP (toks/s)",
    "Throughput vs Prefix Groups",
    os.path.join(OUT, "throughput_vs_fam.png")
)

make_plot(
    "avg_batch",
    "Avg Batch Size",
    "Average Batch Size vs Prefix Groups",
    os.path.join(OUT, "avg_batch_vs_fam.png")
)

make_plot(
    "avg_diff",
    "TBT (s)",
    "Average Consecutive Scheduler Timestamp Diff vs Prefix Groups",
    os.path.join(OUT, "avg_timediff_vs_fam.png")
)

print("\nDone! Plots written to:", OUT)