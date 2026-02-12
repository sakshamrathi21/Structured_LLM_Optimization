#!/home/saksham/CS496/code/vllm/.venv/bin/python
import csv
import statistics


def analyze_l2_hit_rates(filename):
    l2_values = []

    with open(filename, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Defensive check for malformed rows
            if len(row) < 2:
                continue

            # Match metric name exactly
            whether_flash = False
            for row_entry in row:
                if "flash" in row_entry:
                    whether_flash = True
                    break
            if "L2 Hit Rate" in row and whether_flash:
                try:
                    value = float(row[-1])
                    l2_values.append(value)
                except ValueError:
                    pass  # skip non-numeric entries

    if not l2_values:
        print("No L2 Hit Rate entries found.")
        return

    print("L2 Hit Rate Statistics")
    print("-" * 30)
    print(f"Count: {len(l2_values)}")
    print(f"Min:   {min(l2_values):.2f}%")
    print(f"Max:   {max(l2_values):.2f}%")
    print(f"Mean:  {statistics.mean(l2_values):.2f}%")
    print(f"Median:{statistics.median(l2_values):.2f}%")

    if len(l2_values) > 1:
        print(f"Stddev:{statistics.stdev(l2_values):.2f}%")

    # Optional: simple histogram buckets
    buckets = {}
    for v in l2_values:
        bucket = int(v // 5) * 5
        buckets[bucket] = buckets.get(bucket, 0) + 1

    print("\nHistogram (5% buckets):")
    for b in sorted(buckets):
        print(f"{b:02d}–{b+4:02d}% : {buckets[b]}")


# Example usage
analyze_l2_hit_rates(".out_fam")
