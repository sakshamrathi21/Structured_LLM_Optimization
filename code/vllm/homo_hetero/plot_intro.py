import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

hatches = ['/', '\\', 'x', '-', '+', 'o', '.', '*']

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="/home/saksham/CS496/code/vllm/homo_hetero/summary.csv")
parser.add_argument("--outdir", type=str, default="../../../results/")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.csv)
df["avg_throughput_req_s"] = pd.to_numeric(df["avg_throughput_req_s"], errors="coerce")

# Multiply throughput by decode tokens
df["throughput_tok_s"] = df["avg_throughput_req_s"] * df["max_new_tokens"]

df = df[df["max_new_tokens"] == 100].copy()
df = df.sort_values(["max_new_tokens", "mode"]).reset_index(drop=True)

SETTINGS = [
    "1x800\n(8 fam)",
    "2x400\n(4 fam)",
    "4x200\n(2 fam)",
    "8x100\n(1 fam)",
    "16x50\n(1 fam)",
    "32x25\n(1 fam)",
]

# Batch sizes corresponding to each setting (total requests per batch)
BATCH_SIZES = [800, 400, 200, 100, 50, 25]

tokens = sorted(df["max_new_tokens"].unique())
num_tokens = len(tokens)
num_configs = len(SETTINGS)

x = np.arange(num_configs)
bar_width = 0.8 / num_tokens

fig, ax = plt.subplots(figsize=(12, 6))

for i, tok in enumerate(tokens):
    subset = df[df["max_new_tokens"] == tok].sort_values("mode").reset_index(drop=True)

    if len(subset) != num_configs:
        print(f"[WARN] max_new_tokens={tok} has {len(subset)} rows (expected {num_configs}), skipping.")
        continue

    throughputs = subset["throughput_tok_s"].tolist()
    offset = (i - num_tokens / 2) * bar_width + bar_width / 2
    bars = ax.bar(
        x + offset,
        throughputs,
        width=bar_width,
        label=f"{tok} tokens",
        edgecolor="black",
        linewidth=0.7,
        hatch=hatches[i % len(hatches)]
    )

    # Add execution time label on top of each bar
    for bar, tp_tok_s, batch_size in zip(bars, throughputs, BATCH_SIZES):
        if tp_tok_s > 0:
            # tok/s → req/s (200 toks per request)
            tp_req_s = tp_tok_s / 100.0
            # time to process full batch
            exec_time = batch_size / tp_req_s
            label = f"{exec_time:.1f}s"
        else:
            label = "N/A"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.01,  # small padding above bar
            label,
            ha='center', va='bottom',
            fontsize=15, fontweight='bold', color='black',
        )

ax.set_xticks(x)
ax.set_xticklabels(SETTINGS, fontsize=20)
ax.set_yticklabels(ax.get_yticks(), fontsize=20)
ax.set_ylabel("Throughput (toks/s)", fontsize=20)
ax.set_xlabel("Batch Configuration", fontsize=20)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Extra top margin so rotated labels don't get clipped
ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
ax.text(
    0.99, 0.97,
    "Bold text = batch execution time (s)",
    transform=ax.transAxes,
    ha='right', va='top',
    fontsize=15, fontstyle='italic', color='black',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8)
)
plt.tight_layout()
out_path = os.path.join(args.outdir, "prefill_decode_homo_hetero_grouped_throughput_tok100.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved → {out_path}")
