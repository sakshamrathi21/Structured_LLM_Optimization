#!/usr/bin/env python3
"""
Tensor Core Utilization (%) vs Request Rate (req/s)
OLD-style plotting + engine-aware labels (vLLM / SGLang)
+ JSON caching: skip re-parsing log files if cache entry exists
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
LOG_DIR = "."
OUT_DIR = "../../../results/tenso_policy_plots/"
CACHE_FILE = "dcgmi_cache.json"
DPI = 180
os.makedirs(OUT_DIR, exist_ok=True)

# ── OLD STYLE ─────────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", "--", ":", ":", "-", "-"]

# ── Rename policies ───────────────────────────────────────────────────────
def rename_policy(policy: str) -> str:
    if policy == "fcfs":
        return "FCFS"
    if policy == "dynamic_batching":
        return "DynBatch"
    if policy == "chunked_hash_tree_bandit":
        return "Feather"
    if policy == "dfs-weight":
        return "DFS-W"
    if policy == "lpm":
        return "LPM"
    return policy.replace("_", " ").title()


# ═════════════════════════════════════════════════════════════════════════
# 1. Filename parser
# ═════════════════════════════════════════════════════════════════════════

def extract_info(filename: str):
    rps_m = re.search(r"rps-([\d.]+)", filename)
    policy_m = re.search(
        r"policy-([^_\.]+(?:_[^_\.]+)*?)(?=_maxnew|\.log|$)", filename
    )
    engine = None
    if "vllm" in filename:
        engine = "vLLM"
    elif "sglang" in filename:
        engine = "SGLang"

    rps = float(rps_m.group(1)) if rps_m else None
    policy = policy_m.group(1) if policy_m else None
    return engine, policy, rps


# ═════════════════════════════════════════════════════════════════════════
# 2. dcgmi reader
# ═════════════════════════════════════════════════════════════════════════

def read_dcgmi(filepath: str) -> dict:
    cols = {"smact": [], "smocc": [], "tenso": [], "drama": []}
    col_order = None

    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("ID"):
                continue
            if line.startswith("#Entity"):
                headers = line.split()
                col_order = [h.lower() for h in headers[1:]]
                continue
            parts = line.split()
            values = parts[2:]
            if col_order is None or len(values) < len(col_order):
                continue
            for key, v in zip(col_order, values):
                if key in cols:
                    try:
                        cols[key].append(float(v))
                    except ValueError:
                        pass

    return {k: np.array(v) for k, v in cols.items()}


# ═════════════════════════════════════════════════════════════════════════
# 3. Cache helpers
# ═════════════════════════════════════════════════════════════════════════

def load_cache(cache_path: str) -> dict:
    """Load existing cache from JSON, or return empty dict."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Cache file corrupt, starting fresh: {cache_path}")
    return {}


def save_cache(cache: dict, cache_path: str):
    """Persist cache to JSON."""
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════
# 4. Data collection (with caching)
# ═════════════════════════════════════════════════════════════════════════

def collect_data(log_dir: str, cache_path: str) -> dict:
    cache = load_cache(cache_path)
    cache_dirty = False                    # track if we need to re-save
    raw = defaultdict(dict)

    for fname in sorted(os.listdir(log_dir)):
        if not (fname.startswith("dcgmi_") and fname.endswith(".log")):
            continue

        engine, policy, rps = extract_info(fname)
        if engine is None or policy is None or rps is None:
            print(f"[SKIP] {fname}")
            continue

        # ── Cache lookup ──────────────────────────────────────────────
        if fname in cache:
            tenso_mean = cache[fname]
            print(f"[CACHE] {fname}  →  tenso={tenso_mean:.2f}%")
        else:
            fpath = os.path.join(log_dir, fname)
            gpu_stats = read_dcgmi(fpath)
            tenso_arr = gpu_stats.get("tenso", np.array([]))

            if tenso_arr.size == 0:
                print(f"[WARN] no tenso data: {fname}")
                continue

            tenso_mean = float(np.mean(tenso_arr)) * 100
            cache[fname] = tenso_mean      # store in cache
            cache_dirty = True
            print(f"[READ] {fname}  →  tenso={tenso_mean:.2f}%")

        policy_name = rename_policy(policy)
        label = f"{engine}:{policy_name}"
        raw[label][rps] = tenso_mean

    if cache_dirty:
        save_cache(cache, cache_path)
        print(f"[CACHE] Saved → {cache_path}")

    return raw


# ═════════════════════════════════════════════════════════════════════════
# 5. Plot
# ═════════════════════════════════════════════════════════════════════════

def plot_tenso_vs_rps(data: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(15, 6))
    policies = sorted(data.keys())

    for i, label in enumerate(policies):
        rps_dict = data[label]
        xs = sorted(rps_dict.keys())
        ys = [rps_dict[x] for x in xs]

        color = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        ax.plot(
            xs, ys,
            marker=marker,
            linestyle=linestyle,
            linewidth=5,
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=0.7,
            color=color,
            label=label,
        )

    ax.set_xlabel("Input Workload (req/sec)", fontsize=35)
    ax.set_ylabel("Tensor Util (%)", fontsize=35)
    ax.tick_params(axis="both", labelsize=30)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.set_ylim(0, 105)

    handles, labels = ax.get_legend_handles_labels()
    sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    if sorted_pairs:
        sorted_labels, sorted_handles = zip(*sorted_pairs)
    else:
        sorted_labels, sorted_handles = [], []

    ax.legend(
        sorted_handles, sorted_labels,
        fontsize=29, framealpha=0.1, ncol=3,
        loc="upper left", columnspacing=0.1,
    )

    ax.grid(True, linewidth=0.7, linestyle="--")
    plt.tight_layout()

    outpath = os.path.join(out_dir, "plot_tenso_vs_rps_policies.png")
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved → {outpath}")


# ═════════════════════════════════════════════════════════════════════════
# 6. Entry
# ═════════════════════════════════════════════════════════════════════════

def main():
    data = collect_data(LOG_DIR, CACHE_FILE)

    if not data:
        print("No data found.")
        return

    print("\n── Summary ─────────────────────────────────────")
    for label in sorted(data):
        for rps in sorted(data[label]):
            print(f"{label:<20} rps={rps:>6}  {data[label][rps]:.2f}%")

    plot_tenso_vs_rps(data, OUT_DIR)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()