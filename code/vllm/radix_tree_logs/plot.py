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

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/radix_tree_logs")

# ── Regex patterns ──────────────────────────────────────────────────────────
RE_SCHEDULER = re.compile(
    r"Scheduler stats;.*?running batch size:\s*(\d+),.*?current time:\s*([\d.]+)"
)

RE_RESULTS_THROUGHPUT = re.compile(
    r"\[RESULTS\].*Throughput:\s*([\d.]+)\s*req/sec"
)

# ── Filename parser ─────────────────────────────────────────────────────────
RE_FILENAME = re.compile(
    r"^(?P<engine>vllm|sglang):(?P<policy>.+?)_req.*_sys(?P<sys>\d+)_mode(?P<mode>\d+)"
)


def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("sys")), int(m.group("mode"))


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

            m = RE_RESULTS_THROUGHPUT.search(line)
            if m:
                throughput = float(m.group(1))
                continue

    avg_batch = np.mean(batch_sizes) if batch_sizes else None

    if len(timestamps) >= 2:
        diffs = np.diff(sorted(timestamps))
        avg_diff = np.mean(diffs)
    else:
        avg_diff = None
    if throughput is None:
        return avg_batch, avg_diff, throughput
    return avg_batch, avg_diff, throughput*200


# ── Collect data: data[label][mode] = {metrics} ─────────────────────────────
# data[label][mode] = dict(throughput, avg_batch, avg_diff)
data = defaultdict(lambda: defaultdict(dict))

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"  SKIP (unrecognised name): {os.path.basename(path)}")
        continue

    engine, policy, sys_val, mode = parsed
    if mode == 5:
        continue
    if engine == "vllm":
        engine = "vLLM"
    if engine == "sglang":
        engine = "SGLang"
    if policy == "dfs-weight":
        policy = "DFS-W"
    if policy == "lpm":
        policy = "LPM"
    if policy == "dynamic_batching":
        policy = "DynBatch"
    if policy == "fcfs":
        policy = "FCFS"
    if policy == "chunked_hash_tree_bandit":
        policy = "Feather"
    if policy == "chunked_hash_tree_heuristic":
        policy = "Feather"

    label = f"{engine}:{policy}"
    if "fcfs" in label.lower() and "sglang" in label.lower():
        continue

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"  {label:45s} mode={mode}  "
        f"tput={throughput}  batch={avg_batch}  dt={avg_diff}"
    )

    # Keep best throughput if duplicate
    existing = data[label][mode]
    if existing:
        existing_tput = existing.get("throughput") or 0
        new_tput = throughput or 0
        if new_tput <= existing_tput:
            continue

    data[label][mode] = dict(
        throughput=throughput,
        avg_batch=avg_batch,
        avg_diff=avg_diff,
        sys_val=sys_val,
    )


# ── Determine axis dimensions ────────────────────────────────────────────────
all_modes  = sorted({mode for policy in data.values() for mode in policy})
all_labels = sorted(data.keys())

print(f"\nPolicies found : {all_labels}")
print(f"Modes found    : {all_modes}")

# ── Color-blind-friendly palette ─────────────────────────────────────────────
PALETTE = [
    "#E63946", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

HATCH_PATTERNS = ["", "//", "xx", "..", "\\\\", "oo", "++", "**"]


# ── Generate one set of 3 bar charts per sys value ──────────────────────────
OUT = os.path.expanduser("../../../results/radix_tree_mode_barcharts")
os.makedirs(OUT, exist_ok=True)

all_sys = sorted({sys_val for label in data for mode in data[label] for sys_val in [data[label][mode].get("sys_val")] if sys_val is not None})

# Rebuild a per-sys view: per_sys[sys_val][label][mode] = metrics
per_sys = defaultdict(lambda: defaultdict(dict))
for label in data:
    for mode in data[label]:
        entry = data[label][mode]
        sv = entry.get("sys_val")
        if sv is not None:
            per_sys[sv][label][mode] = entry

for sv in sorted(per_sys.keys()):
    sys_data   = per_sys[sv]
    sys_modes  = sorted({m for lbl in sys_data.values() for m in lbl})
    sys_labels = sorted(sys_data.keys())

    def make_barchart_for_sys(metric_key, ylabel, title, outpath):
        n_modes  = len(sys_modes)
        n_labels = len(sys_labels)
        if n_modes == 0 or n_labels == 0:
            print(f"  No data for sys={sv}, metric={metric_key}, skipping.")
            return

        bar_width = 0.8 / n_labels
        group_pos = np.arange(n_modes)

        fig, ax = plt.subplots(figsize=(max(12, n_modes * 2.2), 6))

        for i, label in enumerate(sys_labels):
            color = PALETTE[i % len(PALETTE)]
            hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
            offsets = (i - (n_labels - 1) / 2.0) * bar_width
            xs = group_pos + offsets
            ys = []
            for mode in sys_modes:
                v = sys_data[label][mode].get(metric_key) if mode in sys_data[label] else None
                ys.append(v if v is not None else 0.0)
            # bold the hatches
            ax.bar(
                xs, ys,
                width=bar_width * 0.92,
                color=color,
                hatch=hatch,
                edgecolor="black",
                linewidth=1,
                label=label,
                zorder=3,
            )

        ax.set_xticks(group_pos)
        def mode_labels_func(m):
            if m == 1:
                return "1 level"
            if m == 9:
                return "2 levels"
            if m == 10:
                return "3 levels"
            if m == 11:
                return "4 levels"
            return None
        mode_labels = [mode_labels_func(m) for m in all_modes]
        ax.set_xticklabels(mode_labels, fontsize=30)
        ax.set_xlabel("Sharing Pattern", fontsize=35)
        ax.set_ylabel(ylabel, fontsize=35)
        ax.tick_params(axis="both", labelsize=28)
        # set 1000 to 1K
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
        if metric_key == "throughput":
            # reduce the gap between rows in the legend
            ax.legend(fontsize=25, framealpha=0.35, ncol=3,
                      loc="upper center", bbox_to_anchor=(0, 1.30), columnspacing=0.1, handletextpad=0.3, borderpad=0.3)
        ax.grid(True, axis="y", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plt.subplots_adjust(top=0.78)
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {outpath}")

    sys_dir = os.path.join(OUT, f"sys{sv}")
    os.makedirs(sys_dir, exist_ok=True)

    make_barchart_for_sys(
        "throughput", "TP (tok/s)",
        "Throughput by Mode",
        os.path.join(sys_dir, "throughput_by_mode.png"),
    )
    make_barchart_for_sys(
        "avg_batch", "Avg Running Batch Size",
        "Average Running Batch Size by Mode",
        os.path.join(sys_dir, "avg_batch_by_mode.png"),
    )
    make_barchart_for_sys(
        "avg_diff", "Time Between Tokens (s)",
        "Avg Consecutive Scheduler Timestamp Diff by Mode",
        os.path.join(sys_dir, "avg_timediff_by_mode.png"),
    )

print("\nDone! Bar charts written to:", OUT)