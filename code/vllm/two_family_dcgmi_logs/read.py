import re
import matplotlib.pyplot as plt

# ====== CONFIGURE THIS ======
log_file = "out_two_family"   # <-- CHANGE THIS
fractions = [
    0, 0.002, 0.01, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.998, 1.0
]
iters_per_fraction = 11
# =============================

start_pattern = re.compile(r"Scheduler start time:\s*([0-9.]+)")
end_pattern = re.compile(r"Scheduler end time:\s*([0-9.]+)")

start_times = []
end_times = []

# Read log file
with open(log_file, "r") as f:
    for line in f:
        start_match = start_pattern.search(line)
        if start_match:
            start_times.append(float(start_match.group(1)))

        end_match = end_pattern.search(line)
        if end_match:
            end_times.append(float(end_match.group(1)))

# Sanity check
if len(start_times) != len(end_times):
    raise ValueError("Mismatch between start and end times")

# Compute durations
durations = [end - start for start, end in zip(start_times, end_times)]

# Compute average per fraction
avg_durations = []
idx = 0

for frac in fractions:
    group = durations[idx:idx + iters_per_fraction]
    if len(group) != iters_per_fraction:
        break
    avg_duration = sum(group) / len(group)
    avg_durations.append((500/avg_duration)*50)
    idx += iters_per_fraction

# Plot
plt.figure()
plt.plot(fractions[:len(avg_durations)], avg_durations, marker='o')
plt.xlabel("Fraction", fontweight="bold", fontsize=20)
plt.ylabel("Throughput (toks/sec)", fontweight="bold", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, max(avg_durations) + 500)
plt.tight_layout()
plt.grid(True)
# plt.title("Average Scheduler Duration vs Fraction")
plt.savefig("../../../results/throughput_vs_fraction_two_fam_llama8b.png")