import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- CONFIG ----------
input_file = "/home/saksham/CS492/code/vllm/vllm_multiple_families_varying_workloads_output.txt"

# ---------- PARSING ----------
# policy -> {len_factor -> throughput}
data = defaultdict(dict)

line_regex = re.compile(
    r"\(len factor: ([0-9.]+)\).*?\(policy: ([a-zA-Z_]+)\).*?: ([0-9.]+)$"
)

with open(input_file, "r") as f:
    for line in f:
        match = line_regex.search(line)
        if match:
            len_factor = float(match.group(1))
            policy = match.group(2)
            throughput = float(match.group(3))
            data[policy][len_factor*1300] = throughput

# ---------- PLOTTING ----------
plt.figure()

for policy, values in data.items():
    # Sort by sequence length
    x = sorted(values.keys())
    y = [values[l] for l in x]
    plt.plot(x, y, marker='o', label=policy)

plt.xlabel("Sequence Length")
plt.ylabel("Throughput (req/sec)")
plt.title("Throughput vs Sequence Length")
plt.legend()
plt.savefig("/home/saksham/CS492/results/Throughput_vs_seq_length.png")
