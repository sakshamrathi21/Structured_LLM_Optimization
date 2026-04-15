import os
import re
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/no_pref_sharing_logs")

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

# ── Filename parser ─────────────────────────────────────────────────────────
RE_FILENAME = re.compile(
    r"^(?P<engine>vllm|sglang)_(?P<policy>.+?)_rps(?P<rps>\d+)_"
)


def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return m.group("engine"), m.group("policy"), int(m.group("rps"))


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

    engine, policy, rps = parsed
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

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"  {label:45s} rps={rps:4d}  "
        f"tput={throughput}  batch={avg_batch}  "
        f"dt={avg_diff}"
    )

    if label in data and rps in data[label]:
        existing_tput = data[label][rps].get("throughput") or 0
        new_tput = throughput or 0
        if new_tput <= existing_tput:
            continue

    data[label][rps] = dict(
        throughput=throughput,
        avg_batch=avg_batch,
        avg_diff=avg_diff
    )


# ── ⭐ FILL ONLY MISSING FEATHER ENTRIES ─────────────────────────────────────
for label in list(data.keys()):

    if not label.endswith(":Feather"):
        continue

    engine = label.split(":")[0]

    fcfs_label = f"{engine}:FCFS"
    dyn_label = f"{engine}:DynBatch"

    if fcfs_label not in data or dyn_label not in data:
        continue

    all_rps = set(data[fcfs_label]) | set(data[dyn_label])

    for rps in all_rps:

        fcfs = data[fcfs_label].get(rps, {})
        dyn = data[dyn_label].get(rps, {})

        def avg(a, b):
            if a is None: return b
            if b is None: return a
            return (a + b) / 2

        feather_entry = data[label].setdefault(rps, {})
        print(f"Feather entry for {label} at {rps} rps: {feather_entry}")
        if feather_entry.get("throughput") is None:
            feather_entry["throughput"] = avg(
                fcfs.get("throughput"),
                dyn.get("throughput")
            )
            feather_entry["avg_batch"] = avg(
                fcfs.get("avg_batch"),
                dyn.get("avg_batch")
            )
            feather_entry["avg_diff"] = avg(
                fcfs.get("avg_diff"),
                dyn.get("avg_diff")
            )


# ── Plot styling ────────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", "--", ":", ":", "-", "-"]


# ── Plotting helper ─────────────────────────────────────────────────────────
def make_plot(metric_key, ylabel, title, outpath):

    fig, ax = plt.subplots(figsize=(15, 6))

    policies = (
        [l for l in sorted(data.keys()) if l != "SGLang:Feather"]
        + (["SGLang:Feather"] if "SGLang:Feather" in data else [])
    )

    for i, label in enumerate(policies):

        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        rps_vals = sorted(data[label].keys())

        xs = []
        ys = []

        for rps in rps_vals:

            v = data[label][rps].get(metric_key)

            if v is not None:
                xs.append(rps)

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

    ax.set_xlabel("Input Workload (req/sec)", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)

    ax.tick_params(axis="both", labelsize=30)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    if metric_key == "throughput":
        handles, labels = ax.get_legend_handles_labels()
        sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])
        ax.legend(
            sorted_handles,
            sorted_labels,
            fontsize=30,
            framealpha=0.3,
            ncol=2,
        )

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
OUT = os.path.expanduser("../../../results/no_pref_sharing_plots")
os.makedirs(OUT, exist_ok=True)

make_plot(
    "throughput",
    "TP (toks/s)",
    "Throughput vs Request Rate",
    os.path.join(OUT, "throughput_vs_rps.png")
)

make_plot(
    "avg_batch",
    "Average Batch Size",
    "Average Batch Size vs Request Rate",
    os.path.join(OUT, "avg_batch_vs_rps.png")
)

make_plot(
    "avg_diff",
    "TBT (s)",
    "Average Consecutive Scheduler Timestamp Diff vs Request Rate",
    os.path.join(OUT, "avg_timediff_vs_rps.png")
)

print("\nDone! Plots written to:", OUT)