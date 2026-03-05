import re
import matplotlib.pyplot as plt

# Path to your single log file
LOG_FILE = "small_requests/smocc"

fractions = []
means = []

current_frac = None

with open(LOG_FILE, "r") as f:
    for line in f:
        # Detect fraction from filename lines
        frac_match = re.search(r"batch-([0-9.]+)_", line)
        if frac_match:
            current_frac = float(frac_match.group(1))
        # Extract Mean
        mean_match = re.search(r"Mean\s*:\s*([0-9.]+)", line)
        if mean_match and current_frac is not None:
            means.append(float(mean_match.group(1)))
            fractions.append(current_frac)

# Sort everything by fraction
sorted_data = sorted(zip(fractions, means), key=lambda x: x[0])
print(fractions, means)
fractions, means = zip(*sorted_data)

# -------------------------
# Plot Mean vs Fraction
# -------------------------
plt.figure()
plt.plot(fractions, means, marker='o')
plt.xlabel("Batch Size", fontweight="bold", fontsize=20)
plt.ylabel("Mean SM Occupancy", fontweight="bold", fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, None)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../../results/mean_smocc_vs_batch_size_llama8b_small_requests.png")