import re
import matplotlib.pyplot as plt

# ====== CONFIGURE THIS ======
log_file = "out_batch_size"
tenso_file = "tenso"
fractions = [
    5, 10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900
]
iters_per_fraction = 11
# =============================

# --- Parse throughput data ---
start_pattern = re.compile(r"Average throughput\s*:\s*([0-9.]+)")
throughput_values = []

with open(log_file, "r") as f:
    for line in f:
        match = start_pattern.search(line)
        if match:
            throughput_values.append(float(match.group(1)) * 50)

# --- Parse tensor core utilization data ---
tenso_fractions = []
means = []
current_frac = None

with open(tenso_file, "r") as f:
    for line in f:
        frac_match = re.search(r"batch-([0-9.]+)_", line)
        if frac_match:
            current_frac = float(frac_match.group(1))

        mean_match = re.search(r"Mean\s*:\s*([0-9.]+)", line)
        if mean_match and current_frac is not None:
            means.append(float(mean_match.group(1)))
            tenso_fractions.append(current_frac)

# Sort tensor data by fraction
sorted_data = sorted(zip(tenso_fractions, means), key=lambda x: x[0])
tenso_fractions, means = zip(*sorted_data)

# --- Dual-axis plot ---
fig, ax1 = plt.subplots(figsize=(12, 6))

color_throughput = "steelblue"
color_tensor = "darkorange"

# Left axis: Throughput
ax1.set_xlabel("Batch Size", fontsize=20)
ax1.set_ylabel("Throughput (toks/sec)", fontsize=22, color=color_throughput)
line1, = ax1.plot(fractions, throughput_values, marker='o', color=color_throughput, label="Throughput", markersize=12)
ax1.tick_params(axis='y', labelcolor=color_throughput, labelsize=20)
ax1.tick_params(axis='x', labelsize=20)
ax1.set_ylim(0, max(throughput_values) + 500)

# Right axis: Tensor Core Utilization
ax2 = ax1.twinx()
ax2.set_ylabel("Mean Tensor Core Utilization", fontsize=22, color=color_tensor)
line2, = ax2.plot(tenso_fractions, means, marker='s', color=color_tensor, linestyle='--', label="Tensor Core Util.", markersize=12)
ax2.tick_params(axis='y', labelcolor=color_tensor, labelsize=20)
ax2.set_ylim(0, None)

# Combined legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=22, loc="lower right")

ax1.grid(True, alpha=0.4)
fig.tight_layout()
plt.savefig("../../../results/throughput_and_tensor_vs_batch_size_llama8b.png", dpi=150)
print("../../../results/throughput_and_tensor_vs_batch_size_llama8b.png")
plt.close()