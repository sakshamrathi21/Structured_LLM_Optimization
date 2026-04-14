#!/usr/bin/env python3
"""
Parse decode_tokens_logs/ and generate 3 plots:
  1. Throughput vs Decode Tokens
  2. Avg Running Batch Size vs Decode Tokens
  3. Avg Consecutive Time Difference vs Decode Tokens

Changes:
  - avg_batch and avg_diff are extrapolated leftward from the first
    available value (flat extension) for points that have no data.
  - X-axis ticks are thinned to avoid clutter.
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

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/decode_tokens_logs/")

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
    r"^(?P<engine>vllm|sglang)_(?P<policy>.*?)_req.*?tok(?P<tok>\d+)_fam"
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
    if policy == "dynamic_batching":
        policy = "DynBatch"
    if policy == "chunked_hash_tree_bandit":
        policy = "Feather"
    if policy == "chunked_hash_tree_heuristic":
        continue
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


# collect all token counts
all_tokens = sorted({tok for policy in data.values() for tok in policy})

# Thin x-axis ticks: keep at most ~6 evenly-spaced ticks on a log scale
def select_display_ticks(tokens, max_ticks=6):
    if len(tokens) <= max_ticks:
        return tokens
    # Pick indices evenly spaced in log space
    log_vals = np.log10(tokens)
    indices = np.round(np.linspace(0, len(tokens) - 1, max_ticks)).astype(int)
    return [tokens[i] for i in sorted(set(indices))]

display_ticks = select_display_ticks(all_tokens)


# ── Plot function ───────────────────────────────────────────────────────────
def make_plot(metric_key, ylabel, title, outpath):

    fig, ax = plt.subplots(figsize=(12, 6))
    policies = sorted(data.keys())

    for i, label in enumerate(policies):

        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        xs = []
        ys = []

        # For avg_batch / avg_diff: find the first token where a real value exists
        # so we can extrapolate leftward (flat extension).
        first_real_tok = None
        first_real_val = None
        if metric_key in ("avg_batch", "avg_diff"):
            for tok in all_tokens:
                v = data[label].get(tok, {}).get(metric_key)
                if v is not None:
                    first_real_tok = tok
                    first_real_val = round(v, 4) if isinstance(v, float) else v
                    break

        for tok in all_tokens:
            v = data[label].get(tok, {}).get(metric_key)

            if v is not None and isinstance(v, float):
                v = round(v, 4)

            # Leftward extrapolation: fill missing points before first real value
            if (
                v is None
                and metric_key in ("avg_batch", "avg_diff")
                and first_real_tok is not None
                and tok < first_real_tok
            ):
                v = first_real_val

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
    ax.set_xlabel("Decode Tokens", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)

    # Use thinned ticks to avoid clutter
    ax.set_xticks(display_ticks)
    ax.set_xticklabels(display_ticks, rotation=30)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax.tick_params(axis="both", labelsize=30)
    # convert 1000 to 1k
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}"))

    if metric_key == "throughput":
        ax.legend(fontsize=25, framealpha=0, ncol=1)

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
OUT = os.path.expanduser("../../../results/decode_tokens_plots")
os.makedirs(OUT, exist_ok=True)

make_plot(
    "throughput",
    "TP (toks/s)",
    "Throughput vs Decode Tokens",
    os.path.join(OUT, "throughput_vs_tok.png")
)

make_plot(
    "avg_batch",
    "Avg Batch Size",
    "Average Batch Size vs Decode Tokens",
    os.path.join(OUT, "avg_batch_vs_tok.png")
)

make_plot(
    "avg_diff",
    "TBT (s)",
    "Average Consecutive Scheduler Timestamp Diff vs Decode Tokens",
    os.path.join(OUT, "avg_timediff_vs_tok.png")
)

print("\nDone! Plots written to:", OUT)