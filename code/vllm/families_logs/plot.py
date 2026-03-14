#!/usr/bin/env python3
"""
Parse req_per_sec_logs/ and generate 3 plots:
  1. Throughput vs RPS
  2. Avg Running Batch Size vs RPS
  3. Avg Consecutive Time Difference vs RPS
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

def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("fam"))

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

    if len(timestamps) >= 2:
        diffs = np.diff(sorted(timestamps))
        avg_diff = np.mean(diffs)
    else:
        avg_diff = None

    return avg_batch, avg_diff, throughput

# ── Collect data ────────────────────────────────────────────────────────────
# data[policy][rps] = {throughput, avg_batch, avg_diff}
data = defaultdict(lambda: defaultdict(dict))

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"  SKIP (unrecognised name): {os.path.basename(path)}")
        continue
    engine, policy, rps = parsed
    label = f"{engine}_{policy}"

    avg_batch, avg_diff, throughput = parse_log(path)
    print(f"  {label:45s} rps={rps:4d}  "
          f"tput={throughput}  batch={avg_batch}  "
          f"dt={avg_diff}")

    data[label][rps] = dict(throughput=throughput,
                            avg_batch=avg_batch,
                            avg_diff=avg_diff)

# ── Plotting helpers ─────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

all_fams = sorted({rps for policy in data for rps in data[policy]})

def make_plot(metric_key, ylabel, title, outpath):
    fig, ax = plt.subplots(figsize=(10, 6))

    color_cycle = iter(PALETTE)
    policies = sorted(data.keys())

    for label in policies:
        color = next(color_cycle)
        rps_vals = sorted(data[label].keys())
        xs, ys = [], []

        for rps in rps_vals:
            v = data[label][rps].get(metric_key)
            if v is not None:
                xs.append(rps)
                if isinstance(v, float):
                    v = round(v, 4)
                ys.append(v)

        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6,
                    color=color, label=label)

    # automatically scale x axis
    ax.set_xscale("log")
    ax.set_xticks(all_fams)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.set_xlabel("Number of Prefix Groups", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.tick_params(axis="both", labelsize=18)

    if metric_key == "throughput":
        ax.legend(fontsize=15, framealpha=0.3)

    ax.grid(True, linewidth=0.7, linestyle="--")

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

# ── Generate plots ──────────────────────────────────────────────────────────
OUT = os.path.expanduser("../../../results/families_plots")
os.makedirs(OUT, exist_ok=True)

make_plot("throughput",
          "Throughput (toks/s)",
          "Throughput vs Request Rate",
          os.path.join(OUT, "throughput_vs_fam.png"))

make_plot("avg_batch",
          "Avg Running Batch Size",
          "Average Running Batch Size vs Request Rate",
          os.path.join(OUT, "avg_batch_vs_fam.png"))

make_plot("avg_diff",
          "Time Between Tokens",
          "Average Consecutive Scheduler Timestamp Diff vs Request Rate",
          os.path.join(OUT, "avg_timediff_vs_fam.png"))

print("\nDone! Plots written to:", OUT)