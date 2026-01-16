import re
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# ----------- CONFIG -----------
LOG_FILE = "/home/saksham/CS492/code/vllm/vllm_multiple_families_varying_workloads_output.txt"   # change to your file name
# ------------------------------

pattern = re.compile(
    r"Throughput Test with (\d+) families, (\d+) req/sec.*\(policy: ([^)]+)\).*: ([0-9.]+)"
)

# data[families][policy][req_sec] = throughput
data = defaultdict(lambda: defaultdict(dict))

with open(LOG_FILE, "r") as f:
    for line in f:
        line = line.strip()
        m = pattern.search(line)
        if m:
            families = int(m.group(1))
            req_sec = int(m.group(2))
            policy = m.group(3)
            throughput = float(m.group(4))

            data[families][policy][req_sec] = throughput


# Ensure consistent ordering
policies = ["radix_cost", "family_load", "fcfs"]

for families in sorted(data.keys()):
    plt.figure()

    req_rates = sorted({
        r for p in data[families].values() for r in p.keys()
    })

    for policy in policies:
        if policy not in data[families]:
            continue

        y = []
        x = []

        for r in req_rates:
            if r in data[families][policy]:
                x.append(r)
                y.append(data[families][policy][r])

        policy_label = policy
        if policy == "family_load":
            policy_label = "user provides family IDs"
        elif policy == "radix_cost":
            policy_label = "radix tree based"
        plt.plot(x, y, marker='o', label=policy)

    plt.title(f"Throughput vs Req/sec ({families} families)")
    plt.xlabel("Request Rate (req/sec)")
    plt.ylabel("Throughput")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../../results/throughput_vs_req_sec_families_{families}_poisson.png")

# plt.savefig("try.png")
