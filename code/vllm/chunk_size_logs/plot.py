
#!/usr/bin/env python3

import os
import re
import glob
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────
LOG_DIR = os.path.expanduser("~/CS496/code/vllm/chunk_size_logs/")
OUT     = os.path.expanduser("../../../results/")

# ── REGEX ─────────────────────────────────────────────────────────────────
RE_PROFILE = re.compile(
    r"\[PROFILE\]\s+(?P<func>[\w.]+)\s*\|.*?total=(?P<total>[\d.]+)\s*ns"
)

RE_THROUGHPUT = re.compile(
    r"Throughput[\w\s]*:\s*([\d.]+)\s*toks/s"
)

RE_THROUGHPUT_VLLM = re.compile(
    r"Throughput Test.*?:\s*([\d.]+)\s*$"
)

RE_FILENAME = re.compile(
    r"sys(?P<sys_tok>\d+)_chunk(?P<chunk>\d+)_"
)

# ── HELPERS ───────────────────────────────────────────────────────────────
def parse_filename(fname):
    m = RE_FILENAME.match(os.path.basename(fname))
    if not m:
        return None
    return int(m.group("sys_tok")), int(m.group("chunk"))


def parse_log(path):
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


# ── DATA COLLECTION ───────────────────────────────────────────────────────
summed_profile  = defaultdict(dict)
throughput_data = defaultdict(dict)

log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
print(f"Found {len(log_files)} log files")

for path in log_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"SKIP filename: {os.path.basename(path)}")
        continue

    sys_tok, chunk = parsed
    func_totals, throughput = parse_log(path)

    if not func_totals:
        print(f"SKIP no PROFILE: {os.path.basename(path)}")
        continue

    total_ns = sum(func_totals.values())
    summed_profile[chunk][sys_tok] = total_ns

    if throughput is not None:
        throughput_data[chunk][sys_tok] = throughput

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]

MARKERS    = ["o", "s", "D", "^", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

all_sys_toks = sorted({
    tok for d in summed_profile.values() for tok in d
})

labels = sorted(summed_profile.keys())

# ──────────────────────────────────────────────────────────────────────────
# 📊 PLOT 1: FUNCTION TOTALS
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for i, label in enumerate(labels):
    xs, ys = [], []

    for tok in all_sys_toks:
        ns = summed_profile[label].get(tok)
        xs.append(tok)
        ys.append(ns / 1e6 if ns else np.nan)  # ns → ms

    ax.plot(
        xs, ys,
        marker=MARKERS[i % len(MARKERS)],
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        color=PALETTE[i % len(PALETTE)],
        linewidth=7,
        label=f"{label}",
        markersize=18
    )

ax.set_xscale("log")
ax.set_xlabel("Shared Prefix Tokens", fontsize=32)
ax.set_ylabel("Scheduler Time (ms)", fontsize=32)

ax.set_xticks(all_sys_toks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.2f}")
)
ax.legend(fontsize=30, ncol=3)
ax.tick_params(axis="both", labelsize=30)
ax.grid(True, which="both", linestyle="--")
ax.set_ylim(bottom=0)

plt.tight_layout()
out1 = os.path.join(OUT, "functotals_vs_systok_chunks.png")
fig.savefig(out1, dpi=150)
plt.close(fig)

print(f"Saved: {out1}")

# ──────────────────────────────────────────────────────────────────────────
# 🚀 PLOT 2: THROUGHPUT
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for i, label in enumerate(labels):
    xs, ys = [], []

    for tok in all_sys_toks:
        val = throughput_data[label].get(tok)
        xs.append(tok)
        ys.append(val if val else np.nan)

    ax.plot(
        xs, ys,
        marker=MARKERS[i % len(MARKERS)],
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        color=PALETTE[i % len(PALETTE)],
        linewidth=5,
        label=f"{label}",
        markersize=18
    )

ax.set_xscale("log")
ax.set_xlabel("Shared Prefix Tokens", fontsize=32)
ax.set_ylabel("TP (toks/s)", fontsize=32)

ax.set_xticks(all_sys_toks)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}K" if x >= 1000 else f"{x:.2f}")
)

ax.legend(fontsize=30, ncol=3)
ax.tick_params(axis="both", labelsize=30)
ax.grid(True, which="both", linestyle="--")
ax.set_ylim(bottom=0)

plt.tight_layout()
out2 = os.path.join(OUT, "throughput_vs_systok_chunks.png")
fig.savefig(out2, dpi=150)
plt.close(fig)

print(f"Saved: {out2}")
print("\nDone!")