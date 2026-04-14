#!/usr/bin/env python3
"""
Parse sys_prompt_tokens_logs/ and generate 3 plots:
  1. Throughput vs Shared Prefix Tokens
  2. Avg Running Batch Size vs Shared Prefix Tokens
  3. Avg Consecutive Time Difference vs Shared Prefix Tokens
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

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/sys_prompt_tokens_logs/")

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
    r"^(?P<engine>vllm|sglang)_(?P<policy>.*?)_req.*?sys(?P<tok>\d+)_(\d+)_(\d+).log"
)


# ── Parse filename ──────────────────────────────────────────────────────────
def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("tok"))


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

    engine, policy, tok = parsed
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
    if policy == "chunked_hash_tree_bandit":
        policy = "Feather"
    if policy == "chunked_hash_tree_heuristic":
        continue
    if policy == "dynamic_batching":
        policy = "DynBatch"
    label = f"{engine}:{policy}"

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"  {label:45s} tok={tok:4d}  "
        f"tput={throughput}  batch={avg_batch}  "
        f"dt={avg_diff}"
    )

    data[label][tok] = dict(
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

all_tokens = sorted({tok for policy in data.values() for tok in policy})


# ── Plot helper ─────────────────────────────────────────────────────────────
def make_plot(metric_key, ylabel, title, outpath):

    fig, ax = plt.subplots(figsize=(12, 6))

    policies = sorted(data.keys())

    for i, label in enumerate(policies):

        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        xs = []
        ys = []

        for tok in all_tokens:

            v = data[label].get(tok, {}).get(metric_key)

            if v is not None and isinstance(v, float):
                v = round(v, 4)

            xs.append(tok)
            ys.append(v if v is not None else np.nan)

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

    ax.set_xlabel("Shared Prefix Tokens", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)

    ax.set_xticks(all_tokens)
    ax.set_xticklabels(all_tokens, rotation=30)
    # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.tick_params(axis="both", labelsize=27)

    # set 1000 to 1K
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.1f}K"))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.1f}K"))

    if metric_key == "throughput":
        ax.legend(fontsize=21, framealpha=0.3, ncol=2, columnspacing=0.1)

    ax.grid(True, linewidth=0.7, linestyle="--")

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
OUT = os.path.expanduser("../../../results/sys_prompt_tokens_plots")
os.makedirs(OUT, exist_ok=True)

make_plot(
    "throughput",
    "TP (toks/s)",
    "Throughput vs Shared Prefix Tokens",
    os.path.join(OUT, "throughput_vs_sys_prompt_tokens.png")
)

make_plot(
    "avg_batch",
    "Avg Batch Size",
    "Average Batch Size vs Shared Prefix Tokens",
    os.path.join(OUT, "avg_batch_vs_sys_prompt_tokens.png")
)

make_plot(
    "avg_diff",
    "Time Between Tokens (s)",
    "Average Consecutive Scheduler Timestamp Diff vs Shared Prefix Tokens",
    os.path.join(OUT, "avg_timediff_vs_sys_prompt_tokens.png")
)

print("\nDone! Plots written to:", OUT)