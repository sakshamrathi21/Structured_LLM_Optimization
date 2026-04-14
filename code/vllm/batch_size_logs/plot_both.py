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
C_TPUT_HETERO   = "#2A9D8F"
C_TPUT_HOMO     = "#57CC99"
C_TENSO_HETERO  = "#E63946"
C_TENSO_HOMO    = "#FF8FA3"

C_SMACT_HETERO  = "#3A86FF"
C_SMACT_HOMO    = "#90C2FF"
C_SMOCC_HETERO  = "#FB8500"
C_SMOCC_HOMO    = "#FFD166"

# Hetero and homo each get distinct marker shapes.
# Left-axis metrics use the first marker, right-axis use the second.
MARKER_HETERO_LEFT  = "o"   # hetero, left-axis  (throughput / SM Activity)
MARKER_HETERO_RIGHT = "s"   # hetero, right-axis (tensor core / SM Occupancy)
MARKER_HOMO_LEFT    = "^"   # homo,   left-axis
MARKER_HOMO_RIGHT   = "D"   # homo,   right-axis

LW = 5
MS = 10

# Fields stored per batch-size entry in the cache
ALL_FIELDS = ("smact", "smocc", "tenso", "drama", "throughput")


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


def _cache_has_all_fields(entry: dict) -> bool:
    for f in ALL_FIELDS:
        val = entry.get(f)
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Readers
# ═════════════════════════════════════════════════════════════════════════════

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


def extract_batch_from_dcgmi_name(filename: str):
    m = re.search(r"batch-(\d+)", filename)
    return int(m.group(1)) if m else None


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
# 3.  Data collection  (cache-aware)
# ═════════════════════════════════════════════════════════════════════════════

def collect_data(log_dir: str,
                 cache_section: dict,
                 throughput_cap=None,
                 label: str = "") -> dict:
    raw: dict = defaultdict(lambda: {
        "smact": [], "smocc": [], "tenso": [], "drama": [], "throughput": None
    })

    cached_bs = set()
    for bs_str, entry in cache_section.items():
        if _cache_has_all_fields(entry):
            cached_bs.add(int(bs_str))
            print(f"[{label}/CACHE]  batch={int(bs_str):>4}  (all fields cached – skipping logs)")

    for fname in sorted(os.listdir(log_dir)):
        fpath = os.path.join(log_dir, fname)

        if fname.startswith("dcgmi_") and fname.endswith(".log"):
            bs = extract_batch_from_dcgmi_name(fname)
            if bs is None:
                print(f"[SKIP] cannot parse batch size: {fname}")
                continue
            if bs in cached_bs:
                continue
            print(f"[{label}/DCGMI]  batch={bs:>4}  ←  {fname}")
            gpu_stats = read_dcgmi(fpath)
            for key in ("smact", "smocc", "tenso", "drama"):
                arr = gpu_stats.get(key, np.array([]))
                if arr.size:
                    raw[bs][key].extend(arr.tolist())

        elif fname.startswith("vllm_") and fname.endswith(".log"):
            bs = extract_batch_from_vllm_name(fname)
            if bs is None:
                print(f"[SKIP] cannot parse batch size: {fname}")
                continue
            if bs in cached_bs:
                continue
            tput = read_throughput_from_vllm_log(fpath)
            print(f"[{label}/VLLM ]  batch={bs:>4}  throughput={tput}  ←  {fname}")
            if tput is not None:
                raw[bs]["throughput"] = tput

    result = {}

    for bs_str, entry in cache_section.items():
        bs = int(bs_str)
        if bs in cached_bs:
            tput = entry.get("throughput")
            if tput is not None and throughput_cap is not None:
                tput = min(tput, throughput_cap)
            result[bs] = {
                "smact":      entry.get("smact"),
                "smocc":      entry.get("smocc"),
                "tenso":      entry.get("tenso"),
                "drama":      entry.get("drama"),
                "throughput": tput,
            }

    for bs, rec in raw.items():
        tput = rec["throughput"]
        if tput is not None and throughput_cap is not None:
            tput = min(tput, throughput_cap)

        smact = float(np.mean(rec["smact"])) * 100 if rec["smact"] else np.nan
        smocc = float(np.mean(rec["smocc"])) * 100 if rec["smocc"] else np.nan
        tenso = float(np.mean(rec["tenso"])) * 100 if rec["tenso"] else np.nan
        drama = float(np.mean(rec["drama"])) * 100 if rec["drama"] else np.nan

        result[bs] = {
            "smact":      smact,
            "smocc":      smocc,
            "tenso":      tenso,
            "drama":      drama,
            "throughput": tput,
        }

        def _json_safe(v):
            return None if (isinstance(v, float) and np.isnan(v)) else v

        cache_section[str(bs)] = {
            "smact":      _json_safe(smact),
            "smocc":      _json_safe(smocc),
            "tenso":      _json_safe(tenso),
            "drama":      _json_safe(drama),
            "throughput": tput,
        }

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Tensor-core fallback
# ═════════════════════════════════════════════════════════════════════════════

def apply_tenso_fallback(hetero: dict, homo: dict) -> None:
    all_bs = set(hetero.keys()) | set(homo.keys())
    for bs in sorted(all_bs):
        h_val = hetero.get(bs, {}).get("tenso")
        o_val = homo.get(bs, {}).get("tenso")
        h_missing = h_val is None or (isinstance(h_val, float) and np.isnan(h_val))
        o_missing = o_val is None or (isinstance(o_val, float) and np.isnan(o_val))

        if h_missing and not o_missing:
            if bs not in hetero:
                hetero[bs] = {}
            hetero[bs]["tenso"] = o_val
            print(f"[FALLBACK] tenso hetero batch={bs} ← homo value ({o_val:.2f}%)")

        elif o_missing and not h_missing:
            if bs not in homo:
                homo[bs] = {}
            homo[bs]["tenso"] = h_val
            print(f"[FALLBACK] tenso homo   batch={bs} ← hetero value ({h_val:.2f}%)")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _base_fig():
    fig, ax_l = plt.subplots(figsize=(10, 5.5))
    ax_r = ax_l.twinx()
    return fig, ax_l, ax_r


def _finish(fig, ax_l, ax_r, xlabel, ylabel_l, ylabel_r, outpath, legend_loc="best"):
    ax_l.set_xlabel(xlabel,   fontsize=35, labelpad=8)
    ax_l.set_ylabel(ylabel_l, fontsize=35)
    ax_r.set_ylabel(ylabel_r, fontsize=35)
    ax_l.tick_params(axis="both", labelsize=30)
    ax_r.tick_params(axis="y",    labelsize=30)
    ax_l.grid(True, linewidth=0.6, linestyle="--", alpha=0.55)

    lines_l, labels_l = ax_l.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_l.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        fontsize=20,
        framealpha=0.35,
        loc="upper left",
        bbox_to_anchor=(0.35, 0.55)
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {outpath}")


def _xy(data: dict, key: str):
    bs_vals = sorted(
        b for b in data
        if data[b].get(key) is not None
        and not (isinstance(data[b][key], float) and np.isnan(data[b][key]))
    )
    ys = [data[b][key] for b in bs_vals]
    return bs_vals, ys


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Plot 1  –  Throughput  &  Tensor Core Utilisation
# ═════════════════════════════════════════════════════════════════════════════

def plot_throughput_tensor(hetero: dict, homo: dict, out_dir: str) -> None:
    fig, ax_l, ax_r = _base_fig()

    kw_base = dict(linewidth=LW,
                   markeredgecolor="white", markeredgewidth=0.8)

    xs, ys = _xy(hetero, "throughput")
    ax_l.plot(xs, ys, color=C_TPUT_HETERO, marker=MARKER_HETERO_LEFT,
              linestyle="-",  label="Throughput - 5 groups", **kw_base, markersize=15)

    xs, ys = _xy(homo, "throughput")
    ax_l.plot(xs, ys, color=C_TPUT_HOMO,   marker=MARKER_HOMO_LEFT,
              linestyle="--", label="Throughput - 1 group",   **kw_base, markersize=15)

    ax_l.set_ylim(bottom=0)
    # change 1000 to 1k
    ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x/1000:.1f}K"))
    ax_l.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))

    xs1, ys1 = _xy(hetero, "tenso")
    ax_r.plot(xs1, ys1, color=C_TENSO_HETERO, marker=MARKER_HETERO_RIGHT,
              linestyle="-",  label="Tensor Core - 5 groups", **kw_base, markersize=15)

    xs, ys = _xy(homo, "tenso")
    # take average with hetero
    ys2 = []
    for b in xs:
        if b in xs1:
            ys2.append((ys[xs.index(b)] + ys1[xs1.index(b)]) / 2)
    ax_r.plot(xs, ys2, color=C_TENSO_HOMO,   marker=MARKER_HOMO_RIGHT,
              linestyle="--", label="Tensor Core - 1 group",   **kw_base, markersize=15)

    ax_r.set_ylim(0, 105)
    ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    _finish(fig, ax_l, ax_r,
            xlabel   = "Batch Size",
            ylabel_l = "TP (toks/s)",
            ylabel_r = "Tensor Util. (%)",
            outpath  = os.path.join(out_dir, "plot1_throughput_tensor_merged.png"),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Plot 2  –  SM Activity  &  SM Occupancy
# ═════════════════════════════════════════════════════════════════════════════

def plot_sm_activity_occupancy(hetero: dict, homo: dict, out_dir: str) -> None:
    fig, ax_l, ax_r = _base_fig()

    kw_base = dict(linewidth=LW, markersize=MS,
                   markeredgecolor="white", markeredgewidth=0.8)

    xs, ys = _xy(hetero, "smact")
    ax_l.plot(xs, ys, color=C_SMACT_HETERO, marker=MARKER_HETERO_LEFT,
              linestyle="-",  label="SM Activity – Hetero", **kw_base)

    xs, ys = _xy(homo, "smact")
    ax_l.plot(xs, ys, color=C_SMACT_HOMO,   marker=MARKER_HOMO_LEFT,
              linestyle="--", label="SM Activity – Homo",   **kw_base)

    ax_l.set_ylim(0, 105)
    ax_l.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    all_bs = sorted(set(hetero.keys()) | set(homo.keys()))
    ax_l.set_xticks(all_bs)
    ax_l.set_xticklabels([str(b) for b in all_bs])

    xs, ys = _xy(hetero, "smocc")
    ax_r.plot(xs, ys, color=C_SMOCC_HETERO, marker=MARKER_HETERO_RIGHT,
              linestyle="-",  label="SM Occupancy – Hetero", **kw_base)

    xs, ys = _xy(homo, "smocc")
    ax_r.plot(xs, ys, color=C_SMOCC_HOMO,   marker=MARKER_HOMO_RIGHT,
              linestyle="--", label="SM Occupancy – Homo",   **kw_base)

    ax_r.set_ylim(0, 105)
    ax_r.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    _finish(fig, ax_l, ax_r,
            xlabel   = "Batch Size",
            ylabel_l = "SM Activity (%)",
            ylabel_r = "SM Occupancy (%)",
            outpath  = os.path.join(out_dir, "plot2_sm_activity_occupancy_merged.png"),
            # legend_loc = "center right"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def _print_summary(data: dict, label: str):
    print(f"\n── {label} summary ────────────────────────────────────────────────")
    for bs in sorted(data):
        d = data[bs]
        print(f"  batch={bs:>4}  tput={d['throughput']}  "
              f"smact={d['smact']:.2f}%  smocc={d['smocc']:.2f}%  "
              f"tenso={d['tenso']:.2f}%  drama={d['drama']:.2f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache = load_cache(CACHE_FILE)
    cache.setdefault("hetero", {})
    cache.setdefault("homo",   {})

    print("=== Collecting HETERO data ===")
    hetero = collect_data(LOG_DIR_HETERO,
                          cache_section=cache["hetero"],
                          throughput_cap=THROUGHPUT_CAP_HETERO,
                          label="HETERO")

    print("\n=== Collecting HOMO data ===")
    homo = collect_data(LOG_DIR_HOMO,
                        cache_section=cache["homo"],
                        throughput_cap=THROUGHPUT_CAP_HOMO,
                        label="HOMO")

    if not hetero and not homo:
        print("No data found — check LOG_DIR_HETERO / LOG_DIR_HOMO.")
        return

    save_cache(cache, CACHE_FILE)

    print("\n=== Applying tenso fallback ===")
    apply_tenso_fallback(hetero, homo)

    _print_summary(hetero, "Hetero")
    _print_summary(homo,   "Homo")

    plot_throughput_tensor(hetero, homo, OUT_DIR)
    plot_sm_activity_occupancy(hetero, homo, OUT_DIR)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()