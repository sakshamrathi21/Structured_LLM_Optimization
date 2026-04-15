import os
import re
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

HATCHES = ["//", "\\\\", "xx", "--", "++", "oo", ".."]

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/models_logs/")
MODEL_SHORT = {
    "qwen_qwen2.5-0.5b-instruct": "Qwen-0.5B",
    "Qwen_Qwen2.5-1.5B": "Qwen-1.5B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-8B",
    "qwen_qwen3-8b": "Qwen-8B",
}


# ─────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────
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
    r"^(?P<engine>vllm|sglang)_(?P<policy>.+?)_(?P<model>"
    r"qwen_qwen2\.5-0\.5b-instruct|"
    r"Qwen_Qwen2\.5-1\.5B|"
    r"meta-llama_Llama-3\.1-8B-Instruct|"
    r"qwen_qwen3-8b"
    r")_req\d+"
)


# ─────────────────────────────────────────────────────────────
# Filename parser
# ─────────────────────────────────────────────────────────────
def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None

    engine = m.group("engine")
    policy = m.group("policy")
    model = MODEL_SHORT.get(m.group("model"), m.group("model"))

    return engine, policy, model


# ─────────────────────────────────────────────────────────────
# Log parser
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Collect Data
# data[model][policy] = metrics
# ─────────────────────────────────────────────────────────────
data = defaultdict(dict)

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:

    parsed = parse_filename(path)

    if parsed is None:
        print("Skipping:", os.path.basename(path))
        continue

    engine, policy, model = parsed
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
    if policy == "dynamic_batching":
        policy = "DynBatch"
    if policy == "family_load":
        policy = "Feather"
    if policy == "chunked_hash_tree_heuristic":
        continue
    label = f"{engine}:{policy}"

    avg_batch, avg_diff, throughput = parse_log(path)

    print(
        f"{label:40s} model={model:20s}"
        f" tput={throughput} batch={avg_batch} dt={avg_diff}"
    )

    data[model][label] = dict(
        throughput=throughput,
        avg_batch=avg_batch,
        avg_diff=avg_diff,
    )


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

PALETTE = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#8338EC",
    "#06D6A0",
]

def model_sort_key(name):
    if "Llama" in name:
        return (999, 0)  # always last

    # extract number before B
    m = re.search(r"(\d+\.?\d*)B", name)
    size = float(m.group(1)) if m else 0

    return (0, size)


def make_bar_plot(metric_key, ylabel, title, outpath):

    models = sorted(data.keys(), key=model_sort_key)
    policies = sorted({p for m in data.values() for p in m})

    x = np.arange(len(models))
    width = 0.8 / len(policies)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, policy in enumerate(policies):

        ys = []

        for model in models:
            v = data[model].get(policy, {}).get(metric_key)

            if v is not None:
                v = round(v, 4)

            ys.append(v if v is not None else 0)

        offset = (i - len(policies) / 2) * width + width / 2

        ax.bar(
            x + offset,
            ys,
            width,
            label=policy,
            color=PALETTE[i % len(PALETTE)],
            hatch=HATCHES[i % len(HATCHES)],  # ← add this
            edgecolor="black",                # ← recommended for visibility
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_xlabel("Model", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    if metric_key == "throughput":
        ax.legend(fontsize=25)
    ax.set_xticklabels(models, fontsize=25)
    ax.tick_params(axis="y", labelsize=30)

    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # set y-axis from 1000 to 1K
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x)) if x > 1 else str(f"{x:2f}".rstrip("0").rstrip(".")) if x > 0 else "0")

    # ── Optional (clean paper-style plots) ──
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    plt.tight_layout()

    plt.savefig(
        outpath,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    plt.close()

    print("Saved:", outpath)


# ─────────────────────────────────────────────────────────────
# Generate plots
# ─────────────────────────────────────────────────────────────

OUT = os.path.expanduser("../../../results/model_bar_plots")
os.makedirs(OUT, exist_ok=True)

make_bar_plot(
    "throughput",
    "TP (toks/s)",
    "Throughput Across Models",
    os.path.join(OUT, "throughput_model_bar.png"),
)

make_bar_plot(
    "avg_batch",
    "Average Batch Size",
    "Average Batch Size Across Models",
    os.path.join(OUT, "avg_batch_model_bar.png"),
)

make_bar_plot(
    "avg_diff",
    "TBT (s)",
    "Scheduler Token Time Across Models",
    os.path.join(OUT, "avg_timediff_model_bar.png"),
)

print("\nDone! Plots written to:", OUT)