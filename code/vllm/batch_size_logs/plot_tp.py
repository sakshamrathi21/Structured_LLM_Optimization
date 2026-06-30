import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR_HETERO = "."
LOG_DIR_HOMO   = "../homo_batch_size_logs"
OUT_DIR        = "../../../results/merged_batch_size_plots/"
CACHE_FILE     = "./data_cache.json"
DPI            = 180
THROUGHPUT_CAP_HETERO = 420     # cap applied to hetero throughput (set None to disable)
THROUGHPUT_CAP_HOMO   = None    # no cap for homo

os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_TPUT_HETERO = "#2A9D8F"
C_TPUT_HOMO   = "#57CC99"

MARKER_HETERO_LEFT = "o"   # hetero throughput
MARKER_HOMO_LEFT   = "^"   # homo throughput

LW = 5


# ═════════════════════════════════════════════════════════════════════════════
# 1.  JSON Cache helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_cache(cache_path: str) -> dict:
    if os.path.isfile(cache_path):
        with open(cache_path) as fh:
            try:
                cache = json.load(fh)
                print(f"[CACHE] Loaded from {cache_path}")
                return cache
            except json.JSONDecodeError:
                print(f"[CACHE] Corrupt cache file – ignoring: {cache_path}")
    return {"hetero": {}, "homo": {}}


def save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)
    print(f"[CACHE] Saved → {cache_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Readers
# ═════════════════════════════════════════════════════════════════════════════

def read_throughput_from_vllm_log(filepath: str):
    pattern = re.compile(r"Throughput Test with .+?:\s*([0-9]+\.?[0-9]+)\s*$")
    with open(filepath) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def extract_batch_from_vllm_name(filename: str):
    m = re.search(r"batch_size_?(\d+)", filename)
    return int(m.group(1)) if m else None


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Data collection  (cache-aware, throughput only)
# ═════════════════════════════════════════════════════════════════════════════

def collect_throughput(log_dir: str, cache_section: dict, throughput_cap=None, label: str = "") -> dict:
    raw: dict = defaultdict(lambda: None)

    cached_bs = set()
    for bs_str, entry in cache_section.items():
        if entry.get("throughput") is not None:
            cached_bs.add(int(bs_str))
            print(f"[{label}/CACHE]  batch={int(bs_str):>4}  (cached – skipping logs)")

    for fname in sorted(os.listdir(log_dir)):
        if not (fname.startswith("vllm_") and fname.endswith(".log")):
            continue
        fpath = os.path.join(log_dir, fname)
        bs = extract_batch_from_vllm_name(fname)
        if bs is None:
            print(f"[SKIP] cannot parse batch size: {fname}")
            continue
        if bs in cached_bs:
            continue
        tput = read_throughput_from_vllm_log(fpath)
        print(f"[{label}/VLLM ]  batch={bs:>4}  throughput={tput}  ←  {fname}")
        if tput is not None:
            raw[bs] = tput

    result = {}

    for bs_str, entry in cache_section.items():
        bs = int(bs_str)
        if bs in cached_bs:
            tput = entry.get("throughput")
            if tput is not None and throughput_cap is not None:
                tput = min(tput, throughput_cap)
            result[bs] = tput

    for bs, tput in raw.items():
        if tput is not None and throughput_cap is not None:
            tput = min(tput, throughput_cap)
        result[bs] = tput
        cache_section[str(bs)] = {"throughput": tput}

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Plotting
# ═════════════════════════════════════════════════════════════════════════════

def _xy(data: dict):
    bs_vals = sorted(b for b in data if data[b] is not None)
    ys = [data[b] for b in bs_vals]
    return bs_vals, ys


def plot_throughput_only(hetero: dict, homo: dict, out_dir: str) -> None:
    fig, ax_l = plt.subplots(figsize=(10, 5.5))

    kw_base = dict(linewidth=LW, markersize=15,
                   markeredgecolor="white", markeredgewidth=0.8)

    xs, ys = _xy(hetero)
    ax_l.plot(xs, ys, color=C_TPUT_HETERO, marker=MARKER_HETERO_LEFT,
              linestyle="-",  label="Throughput - 5 groups", **kw_base)

    xs, ys = _xy(homo)
    ax_l.plot(xs, ys, color=C_TPUT_HOMO,   marker=MARKER_HOMO_LEFT,
              linestyle="--", label="Throughput - 1 group", **kw_base)

    ax_l.set_ylim(bottom=0)
    ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.1f}K"))
    ax_l.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

    ax_l.set_xlabel("Batch Size", fontsize=35, labelpad=8)
    ax_l.set_ylabel("TP (toks/s)", fontsize=35)
    ax_l.tick_params(axis="both", labelsize=30)
    ax_l.grid(True, linewidth=0.6, linestyle="--", alpha=0.55)
    ax_l.legend(fontsize=20, framealpha=0.35, loc="best")

    plt.tight_layout()
    outpath = os.path.join(out_dir, "plot1_throughput_only_merged.png")
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_cache(CACHE_FILE)
    cache.setdefault("hetero", {})
    cache.setdefault("homo",   {})

    print("=== Collecting HETERO throughput ===")
    hetero = collect_throughput(LOG_DIR_HETERO,
                                 cache_section=cache["hetero"],
                                 throughput_cap=THROUGHPUT_CAP_HETERO,
                                 label="HETERO")

    print("\n=== Collecting HOMO throughput ===")
    homo = collect_throughput(LOG_DIR_HOMO,
                               cache_section=cache["homo"],
                               throughput_cap=THROUGHPUT_CAP_HOMO,
                               label="HOMO")

    if not hetero and not homo:
        print("No data found — check LOG_DIR_HETERO / LOG_DIR_HOMO.")
        return

    save_cache(cache, CACHE_FILE)

    print("\n── Hetero throughput ──")
    for bs in sorted(hetero):
        print(f"  batch={bs:>4}  tput={hetero[bs]}")

    print("\n── Homo throughput ──")
    for bs in sorted(homo):
        print(f"  batch={bs:>4}  tput={homo[bs]}")

    plot_throughput_only(hetero, homo, OUT_DIR)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
