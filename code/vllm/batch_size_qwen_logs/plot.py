"""
plot_batch_size.py
==================
Reads dcgmi_*.log files and vllm_*.log files from the current directory,
then produces two publication-quality plots:

  Plot 1  –  Throughput (req/s)  vs  Tensor Core Utilization (%)   [dual y-axis]
  Plot 2  –  SM Activity (%)     vs  SM Occupancy (%)               [dual y-axis]

X-axis for both: batch size  (extracted from the filename).

File-name patterns expected:
  dcgmi  : dcgmi_vllm_..._batch-<B>_..._sptokens-<S>_policy-<P>_...
  vllm   : vllm_<policy>_batch_size<B>_req<R>_tok<T>_fam<F>_sys<S>_<timestamp>.log

Usage:
  python plot_batch_size.py            # runs in current working directory
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR  = "."
OUT_DIR  = "../../../results/batch_size_qwen_plots/"          # save plots alongside the script / logs
DPI      = 180
os.makedirs(OUT_DIR, exist_ok=True)
# ── Colour palette (colour-blind-friendly) ────────────────────────────────────
C_THROUGHPUT = "#2A9D8F"   # teal
C_TENSOR     = "#E63946"   # red
C_SMACT      = "#3A86FF"   # blue
C_SMOCC      = "#FB8500"   # orange

MARKER_LEFT  = "o"
MARKER_RIGHT = "s"

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Readers
# ═════════════════════════════════════════════════════════════════════════════

def read_dcgmi(filepath: str) -> dict:
    """
    Parse a dcgmi log file.
    Returns dict with keys: smact, smocc, tenso, drama
    each holding a np.ndarray of per-sample floats.
    """
    cols = {"smact": [], "smocc": [], "tenso": [], "drama": []}
    col_order = None          # will be set from the header line

    with open(filepath) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("ID"):
                continue

            # Header line starts with '#Entity'
            if line.startswith("#Entity"):
                headers = line.split()
                # headers = ['#Entity', 'SMACT', 'SMOCC', 'TENSO', 'DRAMA']
                col_order = [h.lower() for h in headers[1:]]
                continue

            parts = line.split()
            # Data lines: "GPU 0   val  val  val  val"
            # Skip the first two tokens ("GPU", "0")
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


def extract_batch_from_dcgmi_name(filename: str):
    """batch-<B>  →  int"""
    m = re.search(r"batch-(\d+)", filename)
    return int(m.group(1)) if m else None


def read_throughput_from_vllm_log(filepath: str) -> float | None:
    """
    Scan a vllm log for the summary line:
        vllm: Throughput Test with ... : 188.55359742522432
    and return the trailing float (req/s).
    """
    # Matches the trailing ": <number>" at the end of the throughput summary line
    pattern = re.compile(r"Throughput Test with .+?:\s*([0-9]+\.?[0-9]+)\s*$")

    with open(filepath) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def extract_batch_from_vllm_name(filename: str):
    """
    Handles both:
      vllm_fcfs_batch_size10_...   →  10
      vllm_fcfs_batch_size_10_...  →  10
    """
    m = re.search(r"batch_size_?(\d+)", filename)
    return int(m.group(1)) if m else None


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Data collection
# ═════════════════════════════════════════════════════════════════════════════

def collect_data(log_dir: str) -> dict:
    """
    Returns:
      {
        batch_size (int): {
          "smact":  float (mean, %),
          "smocc":  float (mean, %),
          "tenso":  float (mean, %),
          "drama":  float (mean, %),
          "throughput": float | None,
        },
        ...
      }
    """
    data = defaultdict(lambda: {
        "smact": [], "smocc": [], "tenso": [], "drama": [], "throughput": None
    })

    for fname in sorted(os.listdir(log_dir)):
        fpath = os.path.join(log_dir, fname)

        # ── dcgmi log ─────────────────────────────────────────────────────
        if fname.startswith("dcgmi_") and fname.endswith(".log"):
            bs = extract_batch_from_dcgmi_name(fname)
            if bs is None:
                print(f"[SKIP] cannot parse batch size: {fname}")
                continue
            print(f"[DCGMI]  batch={bs:>4}  ←  {fname}")
            gpu_stats = read_dcgmi(fpath)
            for key in ("smact", "smocc", "tenso", "drama"):
                arr = gpu_stats.get(key, np.array([]))
                if arr.size:
                    data[bs][key].extend(arr.tolist())

        # ── vllm throughput log ───────────────────────────────────────────
        elif fname.startswith("vllm_") and fname.endswith(".log"):
            bs = extract_batch_from_vllm_name(fname)
            if bs is None:
                print(f"[SKIP] cannot parse batch size: {fname}")
                continue
            tput = read_throughput_from_vllm_log(fpath)
            print(f"[VLLM ]  batch={bs:>4}  throughput={tput}  ←  {fname}")
            if tput is not None:
                data[bs]["throughput"] = tput

    # ── Collapse lists → means ────────────────────────────────────────────
    result = {}
    for bs, rec in data.items():
        result[bs] = {
            "smact":      float(np.mean(rec["smact"]))  * 100 if rec["smact"]  else np.nan,
            "smocc":      float(np.mean(rec["smocc"]))  * 100 if rec["smocc"]  else np.nan,
            "tenso":      float(np.mean(rec["tenso"]))  * 100 if rec["tenso"]  else np.nan,
            "drama":      float(np.mean(rec["drama"]))  * 100 if rec["drama"]  else np.nan,
            "throughput": rec["throughput"] if rec["throughput"] is not None else None
        }
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _base_fig():
    fig, ax_l = plt.subplots(figsize=(10, 5.5))
    ax_r = ax_l.twinx()
    return fig, ax_l, ax_r


def _finish(fig, ax_l, ax_r,
            xlabel, ylabel_l, ylabel_r,
            legend_l, legend_r,
            outpath):
    ax_l.set_xlabel(xlabel,   fontsize=25, labelpad=8)
    ax_l.set_ylabel(ylabel_l, fontsize=25, color=ax_l.get_lines()[0].get_color())
    ax_r.set_ylabel(ylabel_r, fontsize=25, color=ax_r.get_lines()[0].get_color())
    ax_l.tick_params(axis="both", labelsize=25)
    ax_r.tick_params(axis="y",    labelsize=25)
    ax_l.grid(True, linewidth=0.6, linestyle="--", alpha=0.55)

    # Combined legend
    lines  = ax_l.get_lines() + ax_r.get_lines()
    labels = [legend_l, legend_r]
    if "throughput" in legend_l.lower():
        loc = "lower right"
    else:
        loc = "center right"
    ax_l.legend(lines, labels, fontsize=25, framealpha=0.35, loc=loc)

    plt.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def _sort_xy(data, key):
    xs = sorted(k for k in data if not np.isnan(data[k].get(key, np.nan) or np.nan))
    ys = [data[x][key] for x in xs]
    return xs, ys


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Plot 1  –  Throughput  &  Tensor Core Utilisation
# ═════════════════════════════════════════════════════════════════════════════

def plot_throughput_tensor(data: dict, out_dir: str) -> None:
    fig, ax_l, ax_r = _base_fig()

    # Left axis: Throughput
    bs_vals = sorted(data.keys())
    tput    = [data[b]["throughput"] or np.nan for b in bs_vals]

    ax_l.plot(bs_vals, tput,
              color=C_THROUGHPUT, marker=MARKER_LEFT,
              linewidth=2.4, markersize=10,
              markeredgecolor="white", markeredgewidth=0.8,
              label="Throughput (toks/s)")
    ax_l.set_ylim(bottom=0)
    ax_l.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Right axis: Tensor Core Utilisation
    tenso = [data[b]["tenso"] for b in bs_vals]

    ax_r.plot(bs_vals, tenso,
              color=C_TENSOR, marker=MARKER_RIGHT,
              linewidth=2.4, markersize=10, linestyle="--",
              markeredgecolor="white", markeredgewidth=0.8,
              label="Tensor Core Util (%)")
    ax_r.set_ylim(0, 105)
    ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax_l.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    _finish(fig, ax_l, ax_r,
            xlabel   = "Batch Size",
            ylabel_l = "Throughput (toks/s)",
            ylabel_r = "Tensor Core Util. (%)",
            legend_l = "Throughput (toks/s)",
            legend_r = "Tensor Core Util (%)",
            outpath  = os.path.join(out_dir, "plot1_throughput_tensor.png"))


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Plot 2  –  SM Activity  &  SM Occupancy
# ═════════════════════════════════════════════════════════════════════════════

def plot_sm_activity_occupancy(data: dict, out_dir: str) -> None:
    fig, ax_l, ax_r = _base_fig()

    bs_vals = sorted(data.keys())
    smact   = [data[b]["smact"] for b in bs_vals]
    smocc   = [data[b]["smocc"] for b in bs_vals]

    ax_l.plot(bs_vals, smact,
              color=C_SMACT, marker=MARKER_LEFT,
              linewidth=2.4, markersize=10,
              markeredgecolor="white", markeredgewidth=0.8,
              label="SM Activity (%)")
    ax_l.set_ylim(0, 105)
    ax_l.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax_r.plot(bs_vals, smocc,
              color=C_SMOCC, marker=MARKER_RIGHT,
              linewidth=2.4, markersize=10, linestyle="--",
              markeredgecolor="white", markeredgewidth=0.8,
              label="SM Occupancy (%)")
    ax_r.set_ylim(0, 105)
    ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax_l.set_xticks(bs_vals)
    ax_l.set_xticklabels([str(b) for b in bs_vals])

    _finish(fig, ax_l, ax_r,
            xlabel   = "Batch Size",
            ylabel_l = "SM Activity (%)",
            ylabel_r = "SM Occupancy (%)",
            legend_l = "SM Activity (%)",
            legend_r = "SM Occupancy (%)",
            outpath  = os.path.join(out_dir, "plot2_sm_activity_occupancy.png"))


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect_data(LOG_DIR)

    if not data:
        print("No data found — check that log files are in:", os.path.abspath(LOG_DIR))
        return

    print("\n── Collected summary ──────────────────────────────────────────────")
    for bs in sorted(data):
        d = data[bs]
        print(f"  batch={bs:>4}  tput={d['throughput']}  "
              f"smact={d['smact']:.2f}%  smocc={d['smocc']:.2f}%  "
              f"tenso={d['tenso']:.2f}%  drama={d['drama']:.2f}%")
    print()

    plot_throughput_tensor(data,       OUT_DIR)
    plot_sm_activity_occupancy(data,   OUT_DIR)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()