import re
import matplotlib.pyplot as plt

# ====== CONFIGURE THIS ======
log_file = "out_batch_size"   # <-- CHANGE THIS
fractions = [
    5, 10, 20, 25, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900
]
iters_per_fraction = 11
# =============================

start_pattern = re.compile(r"Average throughput     : \s*([0-9.]+)")

start_times = []

# Read log file
with open(log_file, "r") as f:
    for line in f:
        start_match = start_pattern.search(line)
        if start_match:
            start_times.append(float(start_match.group(1))*50)

# Plot
plt.figure()
# print(fractions)
plt.plot(fractions, start_times, marker='o')
plt.xlabel("Batch Size", fontweight="bold", fontsize=20)
plt.ylabel("Throughput (toks/sec)", fontweight="bold", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, max(start_times) + 500)
plt.tight_layout()
plt.grid(True)
# plt.title("Average Scheduler Duration vs Fraction")
plt.savefig("../../../results/throughput_vs_batch_size_llama8b.png")