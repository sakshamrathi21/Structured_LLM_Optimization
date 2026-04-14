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

# 🔹 Multiply throughput by decode tokens
df["throughput_tok_s"] = df["avg_throughput_req_s"] * df["max_new_tokens"]

# Keep only selected decode lengths
df = df[df["max_new_tokens"].isin([1, 10, 50, 100])].copy()
# df = df[df["max_new_tokens"] == 100].copy()

# Optional: ensure sorted order
df = df.sort_values(["max_new_tokens", "mode"]).reset_index(drop=True)

SETTINGS = [
    "1x800\n(8 groups)",
    "2x400\n(4 groups)",
    "4x200\n(2 groups)",
    "8x100\n(1 group)",
    "16x50\n(1 group)",
    "32x25\n(1 group)",
]

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

    # 🔹 Use token throughput instead of req throughput
    throughputs = subset["throughput_tok_s"].tolist()

    offset = (i - num_tokens / 2) * bar_width + bar_width / 2
    bars = ax.bar(
        x + offset,
        throughputs,
        width=bar_width,
        label=f"{tok} tokens",
        edgecolor="black",   # better visibility for hatches
        linewidth=0.7,
        hatch=hatches[i % len(hatches)]  # 🔹 add hatch here
    )

ax.set_xticks(x)
ax.set_xticklabels(SETTINGS, fontsize=21)
ax.set_yticklabels(ax.get_yticks(), fontsize=30)
ax.set_ylabel("TP (toks/s)", fontsize=35)  # 🔹 Updated label
ax.set_xlabel("Batch Configuration", fontsize=35)

ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=24, title_fontsize=20, ncol=2, columnspacing=0.1, loc="upper left", bbox_to_anchor=(-0.01, 1))

plt.tight_layout()

out_path = os.path.join(args.outdir, "prefill_decode_homo_hetero_grouped_throughput.png")
plt.savefig(out_path, dpi=150)
plt.close()

print(f"\nSaved → {out_path}")