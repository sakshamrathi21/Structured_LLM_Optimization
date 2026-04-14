import os
import re
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

LOG_DIR = os.path.expanduser("~/CS496/code/vllm/sys_prompt_tokens_logs_microbenchmark/")

# ── Regex patterns ──────────────────────────────────────────────────────────
RE_PROFILE = re.compile(
    r"\[PROFILE\]\s+(?P<func>[\w.]+)\s*\|.*?total=(?P<total>[\d.]+)\s*ns"
)

# Matches: "Throughput: 1234.56 toks/s"  (SGLang style)
RE_THROUGHPUT = re.compile(
    r"Throughput[\w\s]*:\s*([\d.]+)\s*toks/s"
)

# Also catch vLLM style: "Throughput Test ...: 1234.56"
RE_THROUGHPUT_VLLM = re.compile(
    r"Throughput Test.*?:\s*([\d.]+)\s*$"
)

# Filename: vllm_family_load_req5000_tok200_fam5_sys1000_20260319_095824.log
RE_FILENAME = re.compile(
    r"^(?P<engine>vllm|sglang)_(?P<policy>.+?)_req(?P<req>\d+)_tok(?P<tok_per_req>\d+)_fam\d+_sys(?P<sys_tok>\d+)_\d+_\d+\.log$"
)

POLICY_MAP = {
    "dfs-weight": "DFS-W",
    "lpm": "LPM",
    "fcfs": "FCFS",
    "chunked_hash_tree_bandit": "Feather",
    "chunked_hash_tree_heuristic": "Feather",
    "family_load": "Oracle",
}

# ── Parse filename ──────────────────────────────────────────────────────────
def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    engine       = {"vllm": "vLLM", "sglang": "SGLang"}.get(m.group("engine"), m.group("engine"))
    policy       = POLICY_MAP.get(m.group("policy"), m.group("policy"))
    req          = int(m.group("req"))
    tok_per_req  = int(m.group("tok_per_req"))
    sys_tok      = int(m.group("sys_tok"))
    label        = f"{engine}:{policy}"
    return label, sys_tok, req, tok_per_req


# ── Parse log ───────────────────────────────────────────────────────────────
def parse_log(path):
    """
    Returns:
      func_totals : dict  func_name -> last total_ns
      throughput  : float | None  (toks/s)
    """
    func_totals = {}
    throughput  = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_PROFILE.search(line)
            if m:
                func_totals[m.group("func")] = float(m.group("total"))
                continue

            m = RE_THROUGHPUT.search(line)
            if m:
                throughput = float(m.group(1))
                continue

            m = RE_THROUGHPUT_VLLM.search(line)
            if m:
                throughput = float(m.group(1))

    return func_totals, throughput


# ── Collect data ────────────────────────────────────────────────────────────
# summed_profile[label][sys_tok] = total_ns  (sum over all functions)
# exec_time[label][sys_tok]      = ms        (from throughput)
summed_profile = defaultdict(dict)
exec_time      = defaultdict(dict)

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"  SKIP (unrecognised name): {os.path.basename(path)}")
        continue

    label, sys_tok, req, tok_per_req = parsed
    func_totals, throughput = parse_log(path)

    if not func_totals:
        print(f"  SKIP (no [PROFILE] lines): {os.path.basename(path)}")
        continue

    total_ns = sum(func_totals.values())
    summed_profile[label][sys_tok] = total_ns

    if throughput and throughput > 0:
        total_tokens = req * tok_per_req          # e.g. 5000 * 200 = 1,000,000
        exec_ms = (total_tokens / throughput) * 1000.0
        exec_time[label][sys_tok] = exec_ms

    print(
        f"  {label:40s} sys={sys_tok:6d}  "
        f"profile_sum={total_ns/1e6:.1f} ms  "
        f"exec_time={exec_time[label].get(sys_tok, float('nan')):.1f} ms  "
        f"funcs={list(func_totals.keys())}"
    )


# ── Plot styling ─────────────────────────────────────────────────────────────
PALETTE = [
    "#E63946", "#2A9D8F", "#8338EC", "#FB5607",
    "#F4A261",  "#06D6A0",
    "#3A86FF", "#FF006E",
]
MARKERS    = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["--", "--", "-", ":", ":"]

all_sys_toks = sorted({tok for d in summed_profile.values() for tok in d})
print(f"\nSys-token counts found: {all_sys_toks}")
print(f"Labels found: {sorted(summed_profile.keys())}\n")


# ── Single combined plot: one line per label ──────────────────────────────────
OUT = os.path.expanduser("../../../results/")

fig, ax = plt.subplots(figsize=(12, 6))

labels = sorted(summed_profile.keys())

for i, label in enumerate(labels):
    color     = PALETTE[i % len(PALETTE)]
    marker    = MARKERS[i % len(MARKERS)]
    linestyle = LINESTYLES[i% len(LINESTYLES)]

    xs, ys = [], []
    for tok in all_sys_toks:
        ns = summed_profile[label].get(tok)
        xs.append(tok)
        ys.append(ns / 1e6 if ns is not None else np.nan)   # ns → ms

    ax.plot(
        xs, ys,
        marker=marker,
        linestyle=linestyle,
        linewidth=5,
        markersize=15,
        markeredgecolor="black",
        markeredgewidth=0.7,
        color=color,
        label=f"{label}",
    )

    # ── Overlay execution time for FamilyLoad ─────────────────────────────
    if "Oracle" in label and label in exec_time:
        xs_e, ys_e = [], []
        for tok in all_sys_toks:
            ms = exec_time[label].get(tok)
            xs_e.append(tok)
            ys_e.append(ms if ms is not None else np.nan)

        ax.plot(
            xs_e, ys_e,
            marker=marker,
            linestyle=":",          # dotted to distinguish
            linewidth=5,
            markersize=15,
            markeredgecolor="black",
            markeredgewidth=0.9,
            color=color,
            alpha=0.65,
            label=f"{label}\nGPU time",
        )

ax.set_xscale("log")
ax.set_yscale("log")  # <-- LOG SCALE ON Y-AXIS
ax.set_xlabel("Shared Prefix Tokens", fontsize=35)
ax.set_ylabel("Time (ms)", fontsize=35)

ax.set_xticks(all_sys_toks)
ax.set_xticklabels([str(t) for t in all_sys_toks], rotation=30)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.tick_params(axis="both", labelsize=22)
# Use LogFormatter for y-axis ticks since it's now log scale
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))

# reduce the gap between legend columns
ax.legend(fontsize=24, framealpha=0.3,ncol=3, columnspacing=0.5)
ax.grid(True, which="major", linewidth=0.7, linestyle="--")  # grid on both major/minor ticks
plt.tight_layout()
outpath = os.path.join(OUT, "profile_sum_vs_sys_prompt_tokens.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outpath}")