import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =========================
# CONFIGURATION
# =========================
LOG_FILE = "/home/saksham/CS496/code/vllm/radix_tree_experiment_results.txt"

RENAMING_MAP = {
    1: "1 level",
    5: "No sharing",
    9: "2 levels",
    10: "3 levels",
    11: "4 levels",
}

# =========================
# PARSE LOG FILE
# =========================
pattern = re.compile(
    r"Mode (\d+), L=(\d+).*?Throughput: ([\d.]+) req/sec"
)

data = defaultdict(dict)
all_L_values = set()
all_modes = set()

with open(LOG_FILE, "r") as f:
    for line in f:
        if line == "\n":
            break
        match = pattern.search(line)
        if match:
            mode = int(match.group(1))
            if mode == 5:
                continue
            L = int(match.group(2))
            throughput = float(match.group(3))
            data[mode][L] = throughput
            all_L_values.add(L)
            all_modes.add(mode)

all_L_values = sorted(all_L_values)
all_modes = sorted(all_modes)

# =========================
# PREPARE DATA FOR PLOTTING
# =========================
num_modes = len(all_modes)
num_L = len(all_L_values)

x = np.arange(num_modes)
bar_width = 0.8 / num_L

# Define hatches and colors
HATCHES = ['/', '\\', 'x', 'o', '+', '-', '|', '.', '*', 'O']
COLORS = plt.cm.coolwarm(np.linspace(0.1, 0.9, num_L))

plt.figure(figsize=(10, 6))

for i, L in enumerate(all_L_values):
    throughputs = []
    for mode in all_modes:
        throughputs.append(data[mode].get(L, 0) * 50)

    offset = (i - num_L / 2) * bar_width + bar_width / 2
    bars = plt.bar(
        x + offset,
        throughputs,
        width=bar_width,
        label=f"L={L}",
        color=COLORS[i],
        hatch=HATCHES[i % len(HATCHES)],
        edgecolor='black',
        linewidth=0.6,
    )

mode_labels = [
    RENAMING_MAP.get(mode, f"Mode {mode}")
    for mode in all_modes
]

plt.xticks(x, mode_labels, fontsize=30)

plt.yticks(fontsize=30)
plt.ylabel("TP(toks/s)", fontsize=35)
plt.xlabel("Sharing Pattern", fontsize=35)
# change 1000 to 1K
ax = plt.gca()
ax.set_yticklabels([f"{(tick/1000):.1f}K" for tick in ax.get_yticks()])
plt.legend(
    fontsize=30,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.20),
    ncol=len(all_L_values),
    frameon=False,
    columnspacing=0.1,
)

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("../../results/tp_different_modes_sharing_at_top.png")
print("Plot saved to ../../results/tp_different_modes_sharing_at_top.png")