import re
from collections import defaultdict

log_file = ".out"  # change this to your log file path

# Data structure: {function_name: {"calls": int, "total": int}}
stats = defaultdict(lambda: {"calls": 0, "total": 0})

# Regex to parse the line
pattern = re.compile(
    r"(?P<func>[^|]+)\s*\|\s*calls=(?P<calls>\d+)\s*\|\s*total=(?P<total>\d+)\s*ns"
)

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        match = pattern.search(line)
        if not match:
            continue

        func = match.group("func").strip()
        calls = int(match.group("calls"))
        total = int(match.group("total"))

        stats[func]["calls"] = calls
        stats[func]["total"] = total

# Print results
print(f"{'Function':50} {'Calls':>10} {'Total(ns)':>15} {'Avg(ns)':>15}")
print("-" * 95)

for func, data in sorted(stats.items()):
    calls = data["calls"]
    total = data["total"]
    avg = total / calls if calls else 0

    print(f"{func:50} {calls:10d} {total:15d} {avg:15.2f}")
