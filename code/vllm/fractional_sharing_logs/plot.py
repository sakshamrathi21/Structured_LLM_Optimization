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
OUT_DIR     = "../../../results/fractional_sharing_plots/"

# ── Plot styling ──────────────────────────────────────────────────────────────
MARKER     = "o"
LINEWIDTH  = 2.4
MARKERSIZE = 12


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
    with open(filepath, "r") as f:
        lines = f.readlines()

    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) < 3:
        return None, None

    mean_line = non_empty[-2]
    sum_line  = non_empty[-1]

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


# ── Timestamp / fraction extraction ──────────────────────────────────────────
TS_PATTERN   = re.compile(r"(\d{8}_\d{6})")
FRAC_PATTERN = re.compile(r"frac([\d.]+)")

def extract_frac(filename: str) -> float | None:
    m = FRAC_PATTERN.search(filename)
    return float(m.group(1)) if m else None

def extract_ts(filename: str) -> str:
    m = TS_PATTERN.search(filename)
    return m.group(1) if m else ""


# ── Step 1: Build results dict ────────────────────────────────────────────────
def build_results() -> dict:
    results: dict[float, dict] = {}

    # ↓ CHANGED: prefix updated to match your actual filenames
    vllm_by_frac: dict[float, list[tuple[str, str]]] = {}

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue

        frac = extract_frac(filename)
        if frac is None:
            print(f"Skipping (no frac): {filename}")
            continue

        # ↓ CHANGED: was "vllm_two_fam_", now "vllm_frac_share_"
        if filename.startswith("vllm_frac_share_"):
            ts = extract_ts(filename)
            vllm_by_frac.setdefault(frac, []).append((ts, filename))

    for frac, ts_files in vllm_by_frac.items():
        ts_files.sort(key=lambda x: x[0], reverse=True)
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


# ── Step 2: Single-series plot ────────────────────────────────────────────────
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
    ax.set_xlabel("Fraction", fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    ax.tick_params(axis="both", labelsize=35)
    ax.set_xticks(xs)
    ax.set_xticklabels([fmt(v) for v in frac_labels], rotation=40, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, linewidth=0.7, linestyle="--")
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.5, len(xs) - 0.5)

    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = build_results()
    fracs   = sorted(results.keys())

    tps        = [results[f]["throughput"] for f in fracs]
    nsys_means = [results[f]["nsys_mean"]  for f in fracs]
    nsys_sums  = [results[f]["nsys_sum"]   for f in fracs]

    plot_single(
        fracs, tps,
        ylabel  = "TP (tok/s)",
        outpath = os.path.join(OUT_DIR, "throughput_vs_fraction.png"),
        color   = "#3A86FF",
    )

    plot_single(
        fracs, nsys_means,
        ylabel  = "Mean DRAM Bandwidth Utilization (GB/s)",
        outpath = os.path.join(OUT_DIR, "nsys_mean_dram_vs_fraction.png"),
        color   = "#FF9F1C",
    )

    plot_single(
        fracs, nsys_sums,
        ylabel  = "Total DRAM Data Accessed (PB)",
        outpath = os.path.join(OUT_DIR, "nsys_sum_dram_vs_fraction.png"),
        color   = "#FF9F1C",
    )


if __name__ == "__main__":
    main()