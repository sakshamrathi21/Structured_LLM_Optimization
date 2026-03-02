import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="/home/saksham/CS496/code/vllm/homo_hetero/summary.csv")
parser.add_argument("--outdir", type=str, default="../../../results/")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.csv)
df["avg_throughput_req_s"] = pd.to_numeric(df["avg_throughput_req_s"], errors="coerce")

# Keep only max_new_tokens = 1, 10, 100
df = df[df["max_new_tokens"].isin([1, 10, 100])].copy()

# Optional: ensure sorted order
df = df.sort_values(["max_new_tokens", "mode"]).reset_index(drop=True)

SETTINGS = [
    "1x800\n(8 fam/batch)",
    "2x400\n(4 fam/batch)",
    "4x200\n(2 fam/batch)",
    "8x100\n(1 fam/batch)",
    "16x50\n(1 fam/batch)",
    "32x25\n(1 fam/batch)",
]

# Sort tokens for consistent legend ordering
tokens = sorted(df["max_new_tokens"].unique())
num_tokens = len(tokens)
num_configs = len(SETTINGS)

# Bar positioning
x = np.arange(num_configs)
bar_width = 0.8 / num_tokens  # total width capped at 0.8

fig, ax = plt.subplots(figsize=(12, 6))

for i, tok in enumerate(tokens):
    subset = df[df["max_new_tokens"] == tok].sort_values("mode").reset_index(drop=True)

    if len(subset) != num_configs:
        print(f"[WARN] max_new_tokens={tok} has {len(subset)} rows (expected {num_configs}), skipping.")
        continue

    throughputs = subset["avg_throughput_req_s"].tolist()

    offset = (i - num_tokens / 2) * bar_width + bar_width / 2
    bars = ax.bar(
        x + offset,
        throughputs,
        width=bar_width,
        label=f"{tok} tokens",
        edgecolor="white",
        linewidth=0.7
    )

    # Optional: annotate bars
    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=14
        )

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(SETTINGS, fontsize=13)
ax.set_ylabel("Throughput (req/s)", fontsize=14)
ax.set_xlabel("Batch Configuration", fontsize=14)
# ax.set_title("Throughput vs Batch Configuration (Grouped by Decode Tokens)", fontsize=15, fontweight="bold", pad=14)

ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(title="max_new_tokens", fontsize=12, title_fontsize=12)

plt.tight_layout()

out_path = os.path.join(args.outdir, "prefill_decode_homo_hetero_grouped_throughput.png")
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\nSaved → {out_path}")