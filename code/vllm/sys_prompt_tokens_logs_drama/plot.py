#!/usr/bin/env python3
"""
Single script that:
  1. Reads .log files from LOG_DIR, extracts DRAMA values per file.
  2. Computes both mean and sum of DRAMA, stores both in results.json.
  3. Generates two plots:
       - Mean DRAM Activity vs Shared Prefix Tokens
       - Sum  DRAM Activity vs Shared Prefix Tokens (normalised to max)

results.json structure:
  { model -> engine -> policy -> sptokens (str) -> {"mean": float|null, "sum": float|null} }
"""

import os
import re
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR     = "."
OUTPUT_FILE = "results.json"
OUT_DIR     = os.path.expanduser("../../../results/sys_prompt_tokens_plots")

# ── Plot styling (colour-blind friendly) ──────────────────────────────────────
PALETTE = [
    "#E63946", "#2A9D8F", "#E9C46A", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]
MARKERS    = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", ":", "-"]

# ── Policy / engine mapping ───────────────────────────────────────────────────
VLLM_POLICIES   = {"fcfs", "dynamic_batching", "chunked_hash_tree_bandit", "family_load"}
SGLANG_POLICIES = {"lpm", "dfs-weight", "fcfs"}
SKIP_POLICIES   = {"chunked_hash_tree_heuristic", "fcfs", "family_load"}

MODEL_PATTERN    = r"model-(.+?)_families-"
POLICY_PATTERN   = r"policy-([a-zA-Z0-9\-_]+?)_maxnew"
SPTOKENS_PATTERN = r"sptokens-(\d+)"


def infer_engine(policy: str) -> str:
    """
    Infer the engine from the policy name.
    'fcfs' is ambiguous — treat as sglang per spec.
    """
    if policy == "fcfs":
        return "sglang"
    if policy in VLLM_POLICIES:
        return "vllm"
    if policy in SGLANG_POLICIES:
        return "sglang"
    return "unknown"


def normalise_policy(policy: str) -> str:
    mapping = {
        "dfs-weight":               "DFS-W",
        "lpm":                      "LPM",
        "fcfs":                     "FCFS",
        "chunked_hash_tree_bandit": "Feather",
        "dynamic_batching":         "DynBatch",
        "family_load":              "FamilyLoad",
    }
    return mapping.get(policy, policy)


def normalise_engine(engine: str) -> str:
    return {"vllm": "vLLM", "sglang": "SGLang"}.get(engine, engine)


# ── DRAMA reader ──────────────────────────────────────────────────────────────
def read_drama(filename: str) -> np.ndarray:
    drama = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ID"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                drama.append(float(parts[-3]))
            except ValueError:
                continue
    drama = np.array(drama)
    nz = np.nonzero(drama)[0]
    if len(nz) == 0:
        raise ValueError(f"{filename}: DRAMA column is all zeros")
    return drama


# ── Metadata extraction ───────────────────────────────────────────────────────
def extract_metadata(filename: str):
    model_match    = re.search(MODEL_PATTERN,    filename)
    policy_match   = re.search(POLICY_PATTERN,   filename)
    sptokens_match = re.search(SPTOKENS_PATTERN, filename)

    if not (model_match and policy_match and sptokens_match):
        return None, None, None, None

    model    = model_match.group(1)
    policy   = policy_match.group(1)
    sptokens = sptokens_match.group(1)
    engine   = infer_engine(policy)
    return model, engine, policy, sptokens


# ── Step 1: Process logs → results.json ───────────────────────────────────────
def build_results() -> dict:
    results = {}

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue

        model, engine, policy, sptokens = extract_metadata(filename)
        if model is None:
            print(f"Skipping (no metadata): {filename}")
            continue

        filepath = os.path.join(LOG_DIR, filename)
        print(f"Processing : {filename}")
        print(f"  model={model}, engine={engine}, policy={policy}, sptokens={sptokens}")

        try:
            drama    = read_drama(filepath)
            mean_val = float(np.mean(drama))
            sum_val  = float(np.sum(drama))
            print(f"  mean={mean_val:.4f}  sum={sum_val:.4f}  (n={len(drama)})")
            entry = {"mean": mean_val*100, "sum": sum_val*960/1000000}
        except Exception as e:
            print(f"  ERROR: {e}")
            entry = {"mean": None, "sum": None}

        # Nested: model → engine → policy → sptokens → {"mean": …, "sum": …}
        results \
            .setdefault(model, {}) \
            .setdefault(engine, {}) \
            .setdefault(policy, {})[sptokens] = entry

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Saved results to {OUTPUT_FILE}")
    return results


# ── Step 2: Flatten JSON → per-metric label series ────────────────────────────
def flatten_data(raw: dict, metric: str) -> dict:
    """
    Returns: label -> {tok_int -> float|None}
    metric is 'mean' or 'sum'.
    """
    data = defaultdict(dict)

    for model, engines in raw.items():
        for engine, policies in engines.items():
            eng_label = normalise_engine(engine)
            for policy, sptokens_map in policies.items():
                if policy in SKIP_POLICIES:
                    continue
                pol_label = normalise_policy(policy)
                label = f"{eng_label}:{pol_label}"
                for tok_str, entry in sptokens_map.items():
                    tok = int(tok_str)
                    val = entry.get(metric) if isinstance(entry, dict) else entry

                    if tok in data[label]:
                        existing = data[label][tok]
                        if existing is None or val is None:
                            data[label][tok] = None
                        else:
                            data[label][tok] = (existing + val) / 2.0
                    else:
                        data[label][tok] = val

    return data


# ── Step 2b: Normalise sum data to fraction of global max ─────────────────────
def normalise_to_max(data: dict) -> dict:
    """
    Rescales all values in `data` so that the global maximum across all
    labels and token points equals 1.0.  None / NaN values are preserved.

    Returns a new dict with the same structure.
    """
    # Collect every finite value to find the global max
    all_vals = [
        v
        for series in data.values()
        for v in series.values()
        if v is not None and not np.isnan(v)
    ]
    if not all_vals:
        return data  # nothing to normalise

    global_max = max(all_vals)
    if global_max == 0:
        return data  # avoid division by zero

    normalised = {}
    for label, series in data.items():
        normalised[label] = {
            tok: (v / global_max if (v is not None and not np.isnan(v)) else v)
            for tok, v in series.items()
        }
    return normalised


# ── Step 3: Plot one metric ───────────────────────────────────────────────────
def plot_metric(data: dict, metric: str, out_dir: str, normalised: bool = False) -> None:
    all_tokens = sorted({tok for series in data.values() for tok in series})

    ylabel_map = {
        "mean": "Mean (%)",
        "sum":  "Sum",
    }
    fname_map = {
        "mean": "mean_dram_vs_sys_prompt_tokens.png",
        "sum":  "sum_dram_vs_sys_prompt_tokens.png",
    }

    print(f"\nPlotting {metric.upper()} — Labels : {sorted(data.keys())}")
    print(f"Tokens  : {all_tokens}")

    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, fname_map[metric])

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, label in enumerate(sorted(data.keys())):
        color     = PALETTE[i % len(PALETTE)]
        marker    = MARKERS[i % len(MARKERS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]

        xs, ys = [], []
        for tok in all_tokens:
            v = data[label].get(tok)
            if isinstance(v, float):
                v = round(v, 4)
            xs.append(tok)
            ys.append(v if v is not None else np.nan)

        ax.plot(
            xs, ys,
            marker=marker,
            linestyle=linestyle,
            linewidth=5,
            markersize=18,
            markeredgecolor="black",
            markeredgewidth=0.7,
            color=color,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Shared Prefix Tokens", fontsize=35)
    ax.set_ylabel(ylabel_map[metric],               fontsize=35)

    ax.set_xticks(all_tokens)
    ax.set_xticklabels(all_tokens, rotation=30)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="both", labelsize=30)

    if metric == "sum":
        # Show as 0.00–1.00 on y-axis
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(0, 1.05)
    else:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(bottom=0)

    if metric == "mean":
        ax.legend(fontsize=28, framealpha=0.3, ncol=1)
    ax.grid(True, linewidth=0.7, linestyle="--")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # 1. Build / save results.json
    raw = build_results()

    # 2. Load from file so plots also work standalone if results.json exists
    with open(OUTPUT_FILE, "r") as f:
        raw = json.load(f)

    # 3. Generate one plot per metric
    for metric in ("mean", "sum"):
        data = flatten_data(raw, metric)
        if metric == "sum":
            data = normalise_to_max(data)
        plot_metric(data, metric, OUT_DIR)


if __name__ == "__main__":
    main()