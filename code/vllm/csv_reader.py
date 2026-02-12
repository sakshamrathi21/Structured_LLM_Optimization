import csv

l2_fcfs_vals = []
with open(".out_fcfs") as f:
    reader = csv.reader(f)

    for row in reader:
        if row[-3] == "L2 Hit Rate":
            l2_fcfs_vals.append(float(row[-1]))

l2_fam_vals = []
with open(".out_fam") as f:
    reader = csv.reader(f)

    for row in reader:
        if row[-3] == "L2 Hit Rate":
            l2_fam_vals.append(float(row[-1]))

import statistics
import numpy as np  # convenient for percentiles


def print_stats(name, vals):
    if not vals:
        print(f"{name}: no data")
        return

    print(f"\n===== {name} =====")
    print(f"Count: {len(vals)}")
    print(f"Mean: {statistics.mean(vals):.6f}")
    print(f"Median: {statistics.median(vals):.6f}")
    print(f"Min: {min(vals):.6f}")
    print(f"Max: {max(vals):.6f}")

    if len(vals) > 1:
        print(f"Std Dev: {statistics.stdev(vals):.6f}")

    # Percentiles
    p = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc = np.percentile(vals, p)

    print("Percentiles:")
    for pct, val in zip(p, perc):
        print(f"  {pct:>2}th: {val:.6f}")


print_stats("FCFS L2 Hit Rate", l2_fcfs_vals)
print_stats("FAM  L2 Hit Rate", l2_fam_vals)
