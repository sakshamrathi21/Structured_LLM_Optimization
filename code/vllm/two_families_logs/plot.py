#!/usr/bin/env python3
"""
Reads two_families_logs/ and produces 3 plots, all vs fraction:
  1. Throughput (req/s)           — from vllm_two_fam_frac*.log
  2. Mean DRAM Activity (GB/s)    — from dcgmi_twofamily_*.log
  3. Total DRAM Data Accessed (PB)— from dcgmi_twofamily_*.log

Usage:
    cd ~/CS496/code/vllm/two_families_logs
    python plot_fractions.py
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR     = "."          # run from two_families_logs/
OUTPUT_JSON = "results_fractions.json"
OUT_DIR     = "../../../results/two_family_plots/"          # save plots here; change if needed

# ── Plot styling ──────────────────────────────────────────────────────────────
MARKER     = "o"
LINEWIDTH  = 2.4
MARKERSIZE = 8

# ── DCGMI: DRAMA reader ───────────────────────────────────────────────────────
def read_drama(filepath: str) -> np.ndarray:
    drama = []
    with open(filepath, "r") as f:
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
        raise ValueError(f"{filepath}: DRAMA column is all zeros")
    return drama


# ── vLLM log: throughput reader ───────────────────────────────────────────────
def read_throughput(filepath: str) -> float:
    pattern = re.compile(r"Average throughput\s*:\s*([\d.]+)\s*req/s", re.IGNORECASE)
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))*200
    raise ValueError(f"No throughput found in {filepath}")


# ── Extract fraction from filename ───────────────────────────────────────────
FRAC_PATTERN = re.compile(r"frac[-_]?([\d.]+)")

def extract_frac(filename: str) -> float | None:
    m = FRAC_PATTERN.search(filename)
    return float(m.group(1)) if m else None


# ── Step 1: Build results dict ────────────────────────────────────────────────
def build_results() -> dict:
    results = {}   # frac (float) -> {"throughput": float|None,
                   #                  "mean": float|None,
                   #                  "sum":  float|None}

    for filename in sorted(os.listdir(LOG_DIR)):
        filepath = os.path.join(LOG_DIR, filename)
        if not filename.endswith(".log"):
            continue

        frac = extract_frac(filename)
        if frac is None:
            print(f"Skipping (no frac): {filename}")
            continue

        entry = results.setdefault(frac, {"throughput": None, "mean": None, "sum": None})

        # ── vLLM throughput log ───────────────────────────────────────────────
        if filename.startswith("vllm_two_fam_"):
            print(f"vLLM  log  frac={frac}: {filename}")
            try:
                tp = read_throughput(filepath)
                entry["throughput"] = tp
                print(f"  throughput = {tp} req/s")
            except Exception as e:
                print(f"  ERROR: {e}")

        # ── DCGMI log ─────────────────────────────────────────────────────────
        elif filename.startswith("dcgmi_twofamily_"):
            print(f"DCGMI log  frac={frac}: {filename}")
            try:
                drama    = read_drama(filepath)
                mean_val = float(np.mean(drama)) * 960          # GB/s
                sum_val  = float(np.sum(drama))  * 960 / 1e6   # PB
                entry["mean"] = mean_val
                entry["sum"]  = sum_val
                print(f"  mean={mean_val:.4f} GB/s  sum={sum_val:.6f} PB  (n={len(drama)})")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Serialise with str keys for JSON
    serialisable = {str(k): v for k, v in sorted(results.items())}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"\n✅ Saved results to {OUTPUT_JSON}")
    return results          # keep float keys internally


# ── Step 2: Generic line plotter (equally spaced x-axis) ─────────────────────
def plot_series(
    fracs:   list,
    values:  list,
    ylabel:  str,
    title:   str,
    outpath: str,
    color:   str,
) -> None:
    # Only keep points where we actually have data
    paired = [(x, y) for x, y in zip(fracs, values) if y is not None]
    if not paired:
        print(f"  No data for {title}, skipping.")
        return

    frac_labels, ys = zip(*paired)

    # Equally spaced integer positions — one per data point
    xs = list(range(len(frac_labels)))

    # Format labels: strip trailing zeros but keep "0" and "1" clean
    def fmt(v: float) -> str:
        if v == 0:
            return "0"
        if v == 1:
            return "1"
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s

    labels = [fmt(v) for v in frac_labels]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        xs, ys,
        marker=MARKER,
        linestyle="-",
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        markeredgecolor="black",
        markeredgewidth=0.8,
        color=color,
    )

    ax.set_xlabel("Fraction", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right")

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linewidth=0.7, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, len(xs) - 0.5)   # small padding on both sides

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = build_results()

    fracs = sorted(results.keys())
    tps   = [results[f]["throughput"] for f in fracs]
    means = [results[f]["mean"]       for f in fracs]
    sums  = [results[f]["sum"]        for f in fracs]

    plot_series(
        fracs, tps,
        ylabel  = "Throughput (tok/s)",
        title   = "Throughput vs Shared Prefix Fraction",
        outpath = os.path.join(OUT_DIR, "throughput_vs_fraction.png"),
        color   = "#3A86FF",
    )

    plot_series(
        fracs, means,
        ylabel  = "Mean DRAM Bandwidth Utilization (GB/s)",
        title   = "Mean DRAM Activity vs Shared Prefix Fraction",
        outpath = os.path.join(OUT_DIR, "mean_dram_vs_fraction.png"),
        color   = "#E63946",
    )

    plot_series(
        fracs, sums,
        ylabel  = "Total DRAM Data Accessed (PB)",
        title   = "Total DRAM Activity vs Shared Prefix Fraction",
        outpath = os.path.join(OUT_DIR, "sum_dram_vs_fraction.png"),
        color   = "#2A9D8F",
    )


if __name__ == "__main__":
    main()