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
df["throughput_tok_s"] = df["avg_throughput_req_s"] * df["max_new_tokens"]
df = df[df["max_new_tokens"] == 100].copy()
df = df.sort_values(["max_new_tokens", "mode"]).reset_index(drop=True)

SETTINGS = [
    "1x800\n(8 groups)",
    "2x400\n(4 groups)",
    "4x200\n(2 groups)",
    "8x100\n(1 group)",
    "16x50\n(1 group)",
    "32x25\n(1 group)",
]

BATCH_SIZES = [800, 400, 200, 100, 50, 25]

tokens = sorted(df["max_new_tokens"].unique())
num_tokens = len(tokens)
num_configs = len(SETTINGS)

x = np.arange(num_configs)
bar_width = 0.8 / num_tokens

fig, ax = plt.subplots(figsize=(12, 6))

# We'll store one bar's info for the arrow annotation (first bar of first token group)
arrow_bar = None
arrow_label = None

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
    )

    for j, (bar, tp_tok_s, batch_size) in enumerate(zip(bars, throughputs, BATCH_SIZES)):
        if tp_tok_s > 0:
            tp_req_s = tp_tok_s / 100.0
            exec_time = batch_size / tp_req_s
            # multiplier: how many batches needed vs the largest batch (800)
            multiplier = 800 // batch_size
            label = f"{multiplier}×{exec_time:.1f}s"
        else:
            label = "N/A"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.01,
            label,
            ha='center', va='bottom',
            fontsize=20, fontweight='bold', color='black',
        )

        # Pick the 3rd bar (index 2, "4x200") of the first token group for the arrow
        if i == 0 and j == 3 and tp_tok_s > 0:
            arrow_bar = bar
            arrow_label = label
            arrow_bar_height = bar.get_height()
            arrow_bar_x = bar.get_x() + bar.get_width()

ax.set_xticks(x)
ax.set_xticklabels(SETTINGS, fontsize=21)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)
ax.set_ylabel("TP (toks/s)", fontsize=25)
ax.set_xlabel("Batch Configuration", fontsize=25)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Extra top margin so labels don't get clipped
ax.set_ylim(top=ax.get_ylim()[1] * 1.35)

# --- Arrow annotation pointing to the chosen bar's execution time label ---
if arrow_bar is not None:
    label_y = arrow_bar_height + max(df["throughput_tok_s"]) * 0.01
    # Place annotation text above and to the right, with an arrow pointing down to the label
    annotation_x = arrow_bar_x + 0.3
    annotation_y = label_y + ax.get_ylim()[1] * 0.15

    ax.annotate(
        "Total Execution Time",
        xy=(arrow_bar_x, label_y + ax.get_ylim()[1] * 0.04),  # tip of arrow: just above label
        xytext=(annotation_x, annotation_y),                   # text position
        fontsize=22,
        color='black',
        ha='left', va='bottom',
        arrowprops=dict(
            arrowstyle="->",
            color='black',
            lw=1.8,
            connectionstyle="arc3,rad=-0.2",
        ),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8),
    )

plt.tight_layout()
out_path = os.path.join(args.outdir, "prefill_decode_homo_hetero_grouped_throughput_tok100.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved → {out_path}")