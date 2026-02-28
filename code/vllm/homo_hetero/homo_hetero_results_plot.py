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

SETTINGS = [
    "1x800\n(8 fam/batch)",
    "2x400\n(4 fam/batch)",
    "4x200\n(2 fam/batch)",
    "8x100\n(1 fam/batch)",
    "16x50\n(1 fam/batch)",
    "32x25\n(1 fam/batch)",
]

# Highlight the best bar in a different color
BASE_COLOR = "#4C72B0"
BEST_COLOR = "#4C72B0"

for tok in sorted(df["max_new_tokens"].unique()):
    subset = df[df["max_new_tokens"] == tok].sort_values("mode").reset_index(drop=True)

    if len(subset) != 6:
        print(f"[WARN] max_new_tokens={tok} has {len(subset)} rows (expected 6), skipping.")
        continue

    throughputs = subset["avg_throughput_req_s"].tolist()
    best_idx = int(np.argmax(throughputs))
    colors = [BEST_COLOR if i == best_idx else BASE_COLOR for i in range(len(throughputs))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(SETTINGS, throughputs, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=15, fontweight="bold", color="#222222"
        )

    ax.set_title(f"Throughput vs Batch Configuration  |  max_new_tokens = {tok}", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylabel("Throughput (req/s)", fontsize=15)
    ax.set_xlabel("Batch Configuration", fontsize=15)
    ax.set_ylim(0, max(throughputs) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    out_path = os.path.join(args.outdir, f"homo_hetero_throughput_barchart_no_warmup_tok{tok:03d}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")

print("\nAll plots saved to:", args.outdir)