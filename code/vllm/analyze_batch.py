#!/usr/bin/env python3
import re
import sys
import statistics


def extract_times(lines):
    times = []
    pattern = re.compile(r"current time:\s*([0-9]+\.[0-9]+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            times.append(float(match.group(1)))

    return times


def compute_differences(times):
    return [times[i] - times[i - 1] for i in range(1, len(times))]


def percentile(data, pct):
    """
    Return the requested percentile using statistics.quantiles.
    pct should be in range 0-100.
    """
    if not data:
        return None

    # statistics.quantiles splits into 100 groups to approximate percentiles
    # We ask for 100-quantiles and pick the desired one
    qs = statistics.quantiles(data, n=100)
    index = max(0, min(98, int(pct) - 1))  # clamp to valid range
    return qs[index]


def print_stats(diffs):
    if not diffs:
        print("No differences to analyze.")
        return

    print("Min:   ", min(diffs))
    print("Max:   ", max(diffs))
    print("Mean:  ", statistics.mean(diffs))

    if len(diffs) > 1:
        print("Stdev: ", statistics.stdev(diffs))

    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        try:
            val = percentile(diffs, p)
            print(f"  P{p}: {val}")
        except Exception:
            print(f"  P{p}: not enough data")


def main():
    # Read from a file if provided, otherwise from stdin
    with open(".out", "r") as f:
        lines = f.readlines()

    times = extract_times(lines)

    if len(times) < 2:
        print("Not enough time values found.")
        return

    diffs = compute_differences(times)
    print_stats(diffs[1:])


if __name__ == "__main__":
    main()
