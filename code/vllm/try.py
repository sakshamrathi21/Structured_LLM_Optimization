import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CONFIG ---
filename = ".out"  # replace with your file name

# --- REGEX PATTERNS ---
request_pattern = re.compile(r"Total number of requests to simulate: (\d+) for max_num_seqs = (\d+)")
start_pattern = re.compile(r"Simulation started at time: ([\d.]+)")
end_pattern = re.compile(r"Simulation completed at time: ([\d.]+)")
family_pattern = re.compile(r"num_families\s*=\s*(\d+)")  # optional, if mentioned

# --- PARSE FILE ---
data = defaultdict(list)
current_family = 1  # default

with open(filename, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]

    # detect family number if specified in text
    family_match = family_pattern.search(line)
    if family_match:
        current_family = int(family_match.group(1))

    # detect start of a block
    req_match = request_pattern.search(line)
    if req_match:
        total_requests = int(req_match.group(1))
        max_num_seqs = int(req_match.group(2))

        # expect next two lines for start and end times
        start_match = start_pattern.search(lines[i + 1])
        end_match = end_pattern.search(lines[i + 2])

        if start_match and end_match:
            start_time = float(start_match.group(1))
            end_time = float(end_match.group(1))
            duration = end_time - start_time
            throughput = total_requests / duration
            data[current_family].append((max_num_seqs, throughput))

        i += 3
    else:
        i += 1

# --- SORT DATA ---
for fam in data:
    data[fam].sort(key=lambda x: x[0])

# --- PLOT ---
plt.figure(figsize=(8, 5))
for fam, values in sorted(data.items()):
    xs = [v[0] for v in values]
    ys = [v[1] for v in values]
    plt.plot(xs, ys, marker="o", label=f"num_families = {fam}")

plt.xlabel("Max batch size (number of requests)")
plt.ylabel("Throughput (requests/sec)")
plt.title("Throughput vs max batch size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("../../results/throughput_vs_max_batch_size_vllm_multiple_families.png")
