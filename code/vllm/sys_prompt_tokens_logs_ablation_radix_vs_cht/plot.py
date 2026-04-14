#!/usr/bin/env python3
import os
import re
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/sys_prompt_tokens_logs_ablation_radix_vs_cht/")

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

FOCUS_POLICIES = {"radix_cost", "chunked_hash_tree_heuristic"}


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

    # Only keep our two focus policies
    if policy not in FOCUS_POLICIES:
        continue

    if engine == "vllm":
        engine = "vLLM"
    if engine == "sglang":
        engine = "SGLang"

    display_policy = "CHT" if policy == "chunked_hash_tree_heuristic" else "RadixTree"
    label = f"{display_policy}"

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


# ── Bar chart styling ────────────────────────────────────────────────────────
PALETTE = {
    "RadixTree": "#8338EC",
    "CHT":   "#3A86FF",
}

token_sets = [set(data[label].keys()) for label in data]
all_tokens = sorted(set.intersection(*token_sets) if token_sets else set())
labels_sorted = sorted(data.keys())

# ── Bar plot helper ──────────────────────────────────────────────────────────
def make_bar_plot(metric_key, ylabel, outpath):

    fig, ax = plt.subplots(figsize=(12, 6))

    n_groups = len(all_tokens)
    n_bars = len(labels_sorted)
    bar_width = 0.35
    group_gap = 0.1

    # Total width per group
    group_width = n_bars * bar_width + group_gap
    x_centers = np.arange(n_groups) * group_width

    for i, label in enumerate(labels_sorted):
        color = PALETTE.get(label, "#888888")

        ys = []
        tbt_vals = []
        for tok in all_tokens:
            v = data[label].get(tok, {}).get(metric_key)
            if v is not None:
                v = round(v, 4)
            ys.append(v if v is not None else 0.0)

            tbt = data[label].get(tok, {}).get("avg_diff")
            tbt_vals.append(tbt)

        offsets = x_centers + (i - (n_bars - 1) / 2) * bar_width

        bars = ax.bar(
            offsets,
            ys,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            label=label,
            zorder=3
        )

        # Annotate TBT on top of each bar
        for bar, tbt in zip(bars, tbt_vals):
            if tbt is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f"{tbt:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=24,
                    color="black",
                )

    ax.set_xlabel("Shared Prefix Tokens", fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(t) for t in all_tokens], rotation=0, fontsize=32)
    ax.tick_params(axis="both", labelsize=32)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.2f}")
    )

    # Build legend: bar colours + TBT annotation marker
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=PALETTE.get(lbl, "#888888"),
                       edgecolor="black", linewidth=0.7,
                       label=lbl)
        for lbl in labels_sorted
    ]
    tbt_handle = Line2D(
        [], [],
        color="none",
        marker="",
        label="Bar annotation: TBT (s)",
        linestyle="none",
    )
    # Use a small text marker to represent the annotation
    from matplotlib.patches import Patch
    tbt_proxy = Line2D(
        [0], [0],
        color="black",
        linewidth=0,
        marker=r"$\mathtt{s}$",
        markersize=12,
        label="Bar top: TBT (s)",
    )
    ax.legend(
        handles=bar_handles + [tbt_proxy],
        fontsize=32,
        framealpha=0.3,
        ncol=3,
        columnspacing=0.6,
        handletextpad=0.3,
        labelspacing=0.3,
        borderpad=0.4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
    )

    ax.grid(True, axis="y", linewidth=0.7, linestyle="--", zorder=0)

    # Add a little headroom so TBT labels don't get clipped
    ymax = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=ymax * 1.18)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Generate plots ───────────────────────────────────────────────────────────
OUT = os.path.expanduser("../../../results/sys_prompt_tokens_plots_ablation")
os.makedirs(OUT, exist_ok=True)

make_bar_plot(
    "throughput",
    "TP (toks/s)",
    os.path.join(OUT, "throughput_vs_sys_prompt_tokens_bar.png")
)