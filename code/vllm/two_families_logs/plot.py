#!/usr/bin/env python3
"""
Reads two_families_logs/ and produces 3 plots, all vs fraction:
  1. Throughput (req/s)            — from vllm_two_fam_frac*.log  (latest timestamp wins)
  2. Mean DRAM Activity (GB/s)     — DCGMI vs NSYS on same axes
  3. Total DRAM Data Accessed (PB) — DCGMI vs NSYS on same axes

For each frac, if multiple vLLM logs exist the one with the latest
timestamp (the _YYYYMMDD_HHMMSS_ suffix) is used.

NSYS numbers are embedded as the last two lines of each vLLM log:
    <mean_dram_activity>|<sum_dram_activity>
e.g.
    24.1172946941933|98.0

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
LOG_DIR     = "."
OUTPUT_JSON = "results_fractions.json"
OUT_DIR     = "../../../results/two_family_plots/"

# ── Plot styling ──────────────────────────────────────────────────────────────
MARKER     = "o"
LINEWIDTH  = 2.4
MARKERSIZE = 8

# ── DCGMI: DRAM reader ────────────────────────────────────────────────────────
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
                return float(m.group(1)) * 200
    raise ValueError(f"No throughput found in {filepath}")


# ── vLLM log: NSYS numbers (last non-empty line containing "|") ───────────────
def read_nsys(filepath: str) -> tuple[float, float] | tuple[None, None]:
    """
    Reads the last 3 non-empty lines from the vLLM log:
        Line -3: <mean_dram_activity>|<ignored>
        Line -2: (ignored separator or duplicate)
        Line -1: <sum_dram_activity>  (raw sum, standalone float)
    Returns (mean_GB_s, sum_PB) or (None, None) if not found.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Grab last 3 non-empty lines
    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) < 3:
        return None, None

    mean_line = non_empty[-2]   # "24.117...|98.0"
    sum_line  = non_empty[-1]   # "138055860.0"

    mean_pattern = re.compile(r"^([\d.]+)\|[\d.]+\s*$")
    sum_pattern  = re.compile(r"^([\d.]+)\s*$")

    m_mean = mean_pattern.match(mean_line)
    m_sum  = sum_pattern.match(sum_line)

    if not m_mean or not m_sum:
        return None, None

    mean_raw = float(m_mean.group(1)) * 960 / 100
    sum_raw  = float(m_sum.group(1))  * 960 / 100
    sum_pb   = sum_raw / 1e6

    return mean_raw, sum_pb


# ── Timestamp extraction from vLLM filename ───────────────────────────────────
TS_PATTERN   = re.compile(r"(\d{8}_\d{6})")
FRAC_PATTERN = re.compile(r"frac[-_]?([\d.]+)")

def extract_frac(filename: str) -> float | None:
    m = FRAC_PATTERN.search(filename)
    return float(m.group(1)) if m else None

def extract_ts(filename: str) -> str:
    """Return timestamp string 'YYYYMMDD_HHMMSS' for sorting, or '' if absent."""
    m = TS_PATTERN.search(filename)
    return m.group(1) if m else ""


# ── Step 1: Build results dict ────────────────────────────────────────────────
def build_results() -> dict:
    """
    results[frac] = {
        "throughput":   float | None,
        "nsys_mean":    float | None,   # GB/s
        "nsys_sum":     float | None,   # PB
        "dcgmi_mean":   float | None,   # GB/s
        "dcgmi_sum":    float | None,   # PB
    }
    For vLLM logs with the same frac, only the latest timestamp is kept.
    """
    results: dict[float, dict] = {}

    # Collect vLLM logs grouped by frac so we can pick the latest
    vllm_by_frac: dict[float, list[tuple[str, str]]] = {}   # frac -> [(ts, filename)]

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue

        frac = extract_frac(filename)
        if frac is None:
            print(f"Skipping (no frac): {filename}")
            continue

        if filename.startswith("vllm_two_fam_"):
            ts = extract_ts(filename)
            vllm_by_frac.setdefault(frac, []).append((ts, filename))

        elif filename.startswith("dcgmi_twofamily_"):
            filepath = os.path.join(LOG_DIR, filename)
            print(f"DCGMI log  frac={frac}: {filename}")
            entry = results.setdefault(frac, _empty_entry())
            try:
                drama    = read_drama(filepath)
                entry["dcgmi_mean"] = float(np.mean(drama)) * 960
                entry["dcgmi_sum"]  = float(np.sum(drama))  * 960 / 1e6
                print(f"  dcgmi_mean={entry['dcgmi_mean']:.4f} GB/s  "
                      f"dcgmi_sum={entry['dcgmi_sum']:.6f} PB  (n={len(drama)})")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Process vLLM logs — pick latest timestamp per frac
    for frac, ts_files in vllm_by_frac.items():
        ts_files.sort(key=lambda x: x[0], reverse=True)   # latest first
        chosen_ts, chosen_name = ts_files[0]
        if len(ts_files) > 1:
            skipped = [n for _, n in ts_files[1:]]
            print(f"vLLM  frac={frac}: choosing {chosen_name} (ts={chosen_ts}), "
                  f"skipping {skipped}")
        else:
            print(f"vLLM  log  frac={frac}: {chosen_name}")

        filepath = os.path.join(LOG_DIR, chosen_name)
        entry = results.setdefault(frac, _empty_entry())

        try:
            tp = read_throughput(filepath)
            entry["throughput"] = tp
            print(f"  throughput = {tp:.2f} tok/s")
        except Exception as e:
            print(f"  throughput ERROR: {e}")

        nsys_mean, nsys_sum = read_nsys(filepath)
        if nsys_mean is not None:
            entry["nsys_mean"] = nsys_mean
            entry["nsys_sum"]  = nsys_sum
            print(f"  nsys_mean={nsys_mean:.4f} GB/s  nsys_sum={nsys_sum:.6f} PB")
        else:
            print(f"  WARNING: no NSYS line found in {chosen_name}")

    # Serialise
    serialisable = {str(k): v for k, v in sorted(results.items())}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(serialisable, f, indent=4)
    print(f"\n✅ Saved results to {OUTPUT_JSON}")
    return results


def _empty_entry() -> dict:
    return {
        "throughput": None,
        "nsys_mean":  None,
        "nsys_sum":   None,
        "dcgmi_mean": None,
        "dcgmi_sum":  None,
    }


# ── Step 2: Single-series plot (throughput) ───────────────────────────────────
def plot_single(
    fracs:   list,
    values:  list,
    ylabel:  str,
    outpath: str,
    color:   str,
) -> None:
    paired = [(x, y) for x, y in zip(fracs, values) if y is not None]
    if not paired:
        print(f"  No data for {outpath}, skipping.")
        return
    frac_labels, ys = zip(*paired)
    xs = list(range(len(frac_labels)))

    def fmt(v):
        if v == 0: return "0"
        if v == 1: return "1"
        return f"{v:.3f}".rstrip("0").rstrip(".")

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(xs, ys, marker=MARKER, linestyle="-", linewidth=LINEWIDTH,
            markersize=MARKERSIZE, markeredgecolor="black",
            markeredgewidth=0.8, color=color)
    ax.set_xlabel("Fraction", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt(v) for v in frac_labels], rotation=40, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linewidth=0.7, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Step 3: Dual-series plot (DCGMI vs NSYS) ─────────────────────────────────
def plot_dual(
    fracs:        list,
    values_dcgmi: list,
    values_nsys:  list,
    ylabel:       str,
    outpath:      str,
    color_dcgmi:  str = "#E63946",
    color_nsys:   str = "#FF9F1C",
) -> None:
    def fmt(v):
        if v == 0: return "0"
        if v == 1: return "1"
        return f"{v:.3f}".rstrip("0").rstrip(".")

    # Build a unified x-axis from fracs that have at least one series
    valid_fracs = [
        f for f, d, n in zip(fracs, values_dcgmi, values_nsys)
        if d is not None or n is not None
    ]
    if not valid_fracs:
        print(f"  No data for {outpath}, skipping.")
        return

    xs = list(range(len(valid_fracs)))
    frac_map = {f: i for i, f in enumerate(valid_fracs)}

    def series_xy(values):
        xs_s, ys_s = [], []
        for f, v in zip(fracs, values):
            if f in frac_map and v is not None:
                xs_s.append(frac_map[f])
                ys_s.append(v)
        return xs_s, ys_s

    xs_d, ys_d = series_xy(values_dcgmi)
    xs_n, ys_n = series_xy(values_nsys)

    fig, ax = plt.subplots(figsize=(11, 7))

    plot_kwargs = dict(
        marker=MARKER, linestyle="-", linewidth=LINEWIDTH,
        markersize=MARKERSIZE, markeredgecolor="black", markeredgewidth=0.8,
    )

    # if xs_d:
    #     ax.plot(xs_d, ys_d, color=color_dcgmi, label="DCGMI", **plot_kwargs)
    if xs_n:
        ax.plot(xs_n, ys_n, color=color_nsys,  label="NSYS",  **plot_kwargs)

    ax.set_xlabel("Fraction", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt(v) for v in valid_fracs], rotation=40, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linewidth=0.7, linestyle="--")
    # ax.set_ylim(bottom=0)
    # ax.set_yscale("log")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.legend(fontsize=16, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Entry point ───────────────────────────────────────────────────────────────
# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = build_results()
    fracs   = sorted(results.keys())

    tps         = [results[f]["throughput"] for f in fracs]
    dcgmi_means = [results[f]["dcgmi_mean"] for f in fracs]
    dcgmi_sums  = [results[f]["dcgmi_sum"]  for f in fracs]
    nsys_means  = [results[f]["nsys_mean"]  for f in fracs]
    nsys_sums   = [results[f]["nsys_sum"]   for f in fracs]

    # 1. Throughput
    plot_single(
        fracs, tps,
        ylabel  = "Throughput (tok/s)",
        outpath = os.path.join(OUT_DIR, "throughput_vs_fraction.png"),
        color   = "#3A86FF",
    )

    # 2. DCGMI Mean DRAM Bandwidth
    plot_single(
        fracs, dcgmi_means,
        ylabel  = "Mean DRAM Bandwidth Utilization (GB/s)",
        outpath = os.path.join(OUT_DIR, "dcgmi_mean_dram_vs_fraction.png"),
        color   = "#E63946",
    )

    # 3. NSYS Mean DRAM Bandwidth
    plot_single(
        fracs, nsys_means,
        ylabel  = "Mean DRAM Bandwidth Utilization (GB/s)",
        outpath = os.path.join(OUT_DIR, "nsys_mean_dram_vs_fraction.png"),
        color   = "#FF9F1C",
    )

    # 4. DCGMI Total DRAM Data
    plot_single(
        fracs, dcgmi_sums,
        ylabel  = "Total DRAM Data Accessed (PB)",
        outpath = os.path.join(OUT_DIR, "dcgmi_sum_dram_vs_fraction.png"),
        color   = "#E63946",
    )

    # 5. NSYS Total DRAM Data
    plot_single(
        fracs, nsys_sums,
        ylabel  = "Total DRAM Data Accessed (PB)",
        outpath = os.path.join(OUT_DIR, "nsys_sum_dram_vs_fraction.png"),
        color   = "#FF9F1C",
    )


if __name__ == "__main__":
    main()