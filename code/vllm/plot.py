import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
INPUT_FILE = "vllm_multiple_families_varying_workloads_output.txt"
# ----------------------------

# Regex to extract fields
pattern = re.compile(
    r"(?P<req>\d+)\s+req/sec.*?"
    r"\(len factor:\s+(?P<len_factor>[\d.]+)\).*?"
    r"\(policy:\s+(?P<policy>[^)]+)\).*?"
    r":\s+(?P<throughput>[\d.]+)"
)

# data[len_factor][policy] = list of (req/sec, throughput)
data = defaultdict(lambda: defaultdict(list))

# ---------- PARSE FILE ----------
with open(INPUT_FILE, "r") as f:
    for line in f:
        match = pattern.search(line)
        if not match:
            continue

        req = int(match.group("req"))
        len_factor = float(match.group("len_factor"))
        policy = match.group("policy")
        throughput = float(match.group("throughput"))

        data[len_factor][policy].append((req, throughput))

print(data)
# ---------- PLOTTING ----------
for len_factor, policies in sorted(data.items()):
    plt.figure()

    for policy, points in policies.items():
        # sort by req/sec for proper lines
        points.sort(key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        print(len_factor, policy, x, y)

        plt.plot(x, y, marker='o', label=policy)

    plt.xlabel("Requests per Second")
    plt.ylabel("Throughput")
    plt.title(f"Throughput vs Req/sec (len factor = {len_factor})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"../../results/final_{len_factor}.png")
