import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =========================
# CONFIGURATION
# =========================

LOG_FILE = "/home/saksham/CS496/code/vllm/radix_tree_experiment_results.txt"  # <-- change to your log file path

# Fill this with your custom names
RENAMING_MAP = {
    1: "Complete sharing",
    5: "No sharing",
    9: "2 levels",
    10: "3 levels",
    11: "4 levels",
    # Add more if needed
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
bar_width = 0.8 / num_L  # keep bars compact

plt.figure(figsize=(10, 6))

for i, L in enumerate(all_L_values):
    throughputs = []
    for mode in all_modes:
        throughputs.append(data[mode].get(L, 0))  # 0 if missing

    offset = (i - num_L / 2) * bar_width + bar_width / 2
    bars = plt.bar(x + offset, throughputs, width=bar_width, label=f"L={L}")

    # === ADD THIS BLOCK TO PRINT VALUES ON TOP ===
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # avoid printing 0 bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",     # format (change decimals if needed)
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=0,
            )

# =========================
# LABELING
# =========================

mode_labels = [
    RENAMING_MAP.get(mode, f"Mode {mode}")
    for mode in all_modes
]

plt.xticks(x, mode_labels)
plt.ylabel("Throughput (req/sec)")
plt.xlabel("Sharing Pattern")
# plt.title("Throughput Across Sharing for Different L Values")
plt.legend(fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("../../results/tp_different_modes_sharing_at_top_5L.png")