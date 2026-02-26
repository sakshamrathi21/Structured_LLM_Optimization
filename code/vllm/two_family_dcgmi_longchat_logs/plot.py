import re
import matplotlib.pyplot as plt

# Path to your single log file
LOG_FILE = "out_two_family_qwen"

fractions = []
means = []
sums = []

current_frac = None

with open(LOG_FILE, "r") as f:
    for line in f:
        # Detect fraction from filename lines
        frac_match = re.search(r"frac-([0-9.]+)_", line)
        if frac_match:
            current_frac = float(frac_match.group(1))

        # Extract Sum
        sum_match = re.search(r"Sum\s*:\s*([0-9.]+)", line)
        if sum_match and current_frac is not None:
            sums.append(float(sum_match.group(1)))

        # Extract Mean
        mean_match = re.search(r"Mean\s*:\s*([0-9.]+)", line)
        if mean_match and current_frac is not None:
            means.append(float(mean_match.group(1)))
            fractions.append(current_frac)

# Sort everything by fraction
sorted_data = sorted(zip(fractions, sums, means), key=lambda x: x[0])
fractions, sums, means = zip(*sorted_data)

# -------------------------
# Plot Mean vs Fraction
# -------------------------
plt.figure()
plt.plot(fractions, means, marker='o')
plt.xlabel("Fraction", fontweight="bold", fontsize=15)
plt.ylabel("Mean DRAM Activity", fontweight="bold", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0, None)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../../results/mean_dram_activity_vs_fraction_two_fam_longchat13b.png")

# -------------------------
# Plot Sum vs Fraction
# -------------------------
plt.figure()
plt.plot(fractions, sums, marker='o')
plt.xlabel("Fraction", fontweight="bold", fontsize=15)
plt.ylabel("Sum of DRAM Activity", fontweight="bold", fontsize=15)
plt.ylim(0, None)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../../results/sum_dram_activity_vs_fraction_two_fam_longchat13b.png")