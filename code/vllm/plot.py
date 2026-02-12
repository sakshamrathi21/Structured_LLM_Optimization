import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
INPUT_FILE = "vllm_multiple_families_varying_workloads_output.txt"
# ----------------------------

# More specific regex that handles all the intermediate fields
pattern = re.compile(
    r"""
    Throughput\s+Test\s+with\s+
    (?P<num_families>\d+)\s+families,\s+
    (?P<req_per_sec>\d+)\s+req/sec\s+and\s+
    max_batch_size\s+of\s+(?P<max_batch>\d+)\s+
    \(randomness:\s*(?P<randomness>\d+)\)\s+
    \(len\s+factor:\s*(?P<len_factor>[\d.]+)\)\s+
    \(model_name:\s*(?P<model>[^)]+)\)\s+
    \(policy:\s*(?P<policy>[\w-]+)\)\s+
    \(max_new_tokens:\s*(?P<max_new_tokens>\d+)\)\s+
    \(num_prompts:\s*(?P<num_prompts>\d+)\)\s*
    :\s*(?P<throughput>[\d.]+)
    """,
    re.VERBOSE,
)

pattern2 = re.compile(
    r"""
    Throughput\s+Test\s*\|\s*
    families=(?P<num_families>\d+)\s*\|\s*
    req/s=(?P<req_per_sec>\d+)\s*\|\s*
    max_batch=(?P<max_batch>\d+)\s*\|\s*
    randomness=(?P<randomness>\d+)\s*\|\s*
    len_factor=(?P<len_factor>[\d.]+)\s*\|\s*
    model=(?P<model>[^|]+?)\s*\|\s*
    max_new_tokens=(?P<max_new_tokens>\d+)\s*\|\s*
    num_prompts=(?P<num_prompts>\d+)\s*\|\s*
    throughput=(?P<throughput>[\d.]+)\s*\|\s*
    policy=(?P<policy>[\w-]+)
    """,
    re.VERBOSE,
)

# data[len_factor][policy] = list of (req/sec, throughput)
data = defaultdict(lambda: defaultdict(list))

# ---------- PARSE FILE ----------
matched_lines = 0
total_lines = 0

with open(INPUT_FILE, "r") as f:
    for line in f:
        total_lines += 1
        second_match = False
        match = pattern.search(line)
        print(match)
        if not match:
            match = pattern2.search(line)
            second_match = True
            if not match:
                continue

        matched_lines += 1
        req = int(match.group("req_per_sec"))
        len_factor = float(match.group("len_factor"))
        policy = match.group("policy")
        throughput = float(match.group("throughput"))
        if policy == "fcfs" and not second_match:
            policy = "vllm fcfs"
        elif policy == "radix_cost":
            policy = "chunked hash tree"
        elif policy == "family_load":
            policy = "user gives out family IDs"
        elif policy == "dynamic_batching":
            policy = "dynamic batching"
        elif policy == "fcfs" and second_match:
            policy = "sglang fcfs"
        elif policy == "lpm":
            policy = "sglang longest prefix match"
        elif policy == "dfs-weight":
            policy = "sglang dfs weight"
        data[len_factor][policy].append((req, throughput))

        # Debug first few matches
        if matched_lines <= 3:
            print(f"\nMatch {matched_lines}:")
            print(f"  Req/sec: {req}")
            print(f"  Len factor: {len_factor}")
            print(f"  Policy: {policy}")
            print(f"  Throughput: {throughput}")

print(f"\n=== PARSING SUMMARY ===")
print(f"Total lines: {total_lines}")
print(f"Matched lines: {matched_lines}")
print(f"Length factors found: {sorted(data.keys())}")

for lf in sorted(data.keys()):
    print(f"\nLen Factor {lf}:")
    for policy, points in data[lf].items():
        print(f"  {policy}: {len(points)} points")
        if points:
            print(f"    Req/sec values: {sorted(set(p[0] for p in points))}")

import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
INPUT_FILE = "vllm_multiple_families_varying_workloads_output.txt"
# ----------------------------

# More specific regex that handles all the intermediate fields
pattern = re.compile(
    r"""
    Throughput\s+Test\s+with\s+
    (?P<num_families>\d+)\s+families,\s+
    (?P<req_per_sec>\d+)\s+req/sec\s+and\s+
    max_batch_size\s+of\s+(?P<max_batch>\d+)\s+
    \(randomness:\s*(?P<randomness>\d+)\)\s+
    \(len\s+factor:\s*(?P<len_factor>[\d.]+)\)\s+
    \(model_name:\s*(?P<model>[^)]+)\)\s+
    \(policy:\s*(?P<policy>[\w-]+)\)\s+
    \(max_new_tokens:\s*(?P<max_new_tokens>\d+)\)\s+
    \(num_prompts:\s*(?P<num_prompts>\d+)\)\s*
    :\s*(?P<throughput>[\d.]+)
    """,
    re.VERBOSE,
)

pattern2 = re.compile(
    r"""
    Throughput\s+Test\s*\|\s*
    families=(?P<num_families>\d+)\s*\|\s*
    req/s=(?P<req_per_sec>\d+)\s*\|\s*
    max_batch=(?P<max_batch>\d+)\s*\|\s*
    randomness=(?P<randomness>\d+)\s*\|\s*
    len_factor=(?P<len_factor>[\d.]+)\s*\|\s*
    model=(?P<model>[^|]+?)\s*\|\s*
    max_new_tokens=(?P<max_new_tokens>\d+)\s*\|\s*
    num_prompts=(?P<num_prompts>\d+)\s*\|\s*
    throughput=(?P<throughput>[\d.]+)\s*\|\s*
    policy=(?P<policy>[\w-]+)
    """,
    re.VERBOSE,
)

# data[len_factor][policy] = list of (req/sec, throughput)
data = defaultdict(lambda: defaultdict(list))

# ---------- PARSE FILE ----------
matched_lines = 0
total_lines = 0

with open(INPUT_FILE, "r") as f:
    for line in f:
        total_lines += 1
        second_match = False
        match = pattern.search(line)
        print(match)
        if not match:
            match = pattern2.search(line)
            second_match = True
            if not match:
                continue

        matched_lines += 1
        req = int(match.group("req_per_sec"))
        len_factor = float(match.group("len_factor"))
        policy = match.group("policy")
        throughput = float(match.group("throughput"))
        if policy == "fcfs" and not second_match:
            policy = "vllm fcfs"
        elif policy == "radix_cost":
            policy = "chunked hash tree + vllm"
        elif policy == "family_load":
            policy = "user gives out family IDs + vllm"
        elif policy == "dynamic_batching":
            policy = "dynamic batching"
        elif policy == "fcfs" and second_match:
            policy = "sglang fcfs"
        elif policy == "lpm":
            policy = "sglang longest prefix match"
        elif policy == "dfs-weight":
            policy = "sglang dfs weight"

        found_existing = False
        for i in range(len(data[len_factor][policy])):
            if data[len_factor][policy][i][0] == req:
                # data[len_factor][policy][i] = (req, throughput)
                found_existing = True
                break

        print(found_existing, policy)
        if not found_existing:
            data[len_factor][policy].append((req, throughput))
        elif found_existing and policy == "chunked hash tree + vllm":
            data[len_factor]["chunked hash tree + vllm + c++"].append((req, throughput))

        # Debug first few matches
        if matched_lines <= 3:
            print(f"\nMatch {matched_lines}:")
            print(f"  Req/sec: {req}")
            print(f"  Len factor: {len_factor}")
            print(f"  Policy: {policy}")
            print(f"  Throughput: {throughput}")

print(f"\n=== PARSING SUMMARY ===")
print(f"Total lines: {total_lines}")
print(f"Matched lines: {matched_lines}")
print(f"Length factors found: {sorted(data.keys())}")

for lf in sorted(data.keys()):
    print(f"\nLen Factor {lf}:")
    for policy, points in data[lf].items():
        print(f"  {policy}: {len(points)} points")
        if points:
            print(f"    Req/sec values: {sorted(set(p[0] for p in points))}")

# ---------- PLOTTING ----------
if not data:
    print("\n⚠️  ERROR: No data was parsed!")
else:
    for len_factor, policies in sorted(data.items()):
        plt.figure(figsize=(10, 6))

        all_x_values = set()  # Collect all unique x values

        for policy, points in policies.items():
            if not points:
                continue

            # sort by req/sec for proper lines
            points.sort(key=lambda x: x[0])
            x = [p[0] for p in points]
            y = [p[1] for p in points]

            all_x_values.update(x)  # Add all x values to our set

            print(
                f"\nPlotting {policy} (len_factor={len_factor}): {len(points)} points"
            )
            print(f"  X (req/sec): {x}")
            print(f"  Y (throughput): {y}")

            plt.plot(x, y, marker="o", label=policy, linewidth=2, markersize=6)

        # Set explicit x-axis ticks to show all data points
        if all_x_values:
            x_ticks = sorted(all_x_values)
            plt.xticks(x_ticks, rotation=45 if len(x_ticks) > 10 else 0)

        plt.xlabel("Requests per Second", fontsize=12)
        plt.ylabel("Throughput (req/sec)", fontsize=12)
        plt.title(
            f"Throughput vs Request Rate (average shared prefix length = {int(len_factor*1300)} tokens)",
            fontsize=14,
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = f"../../results/throughput_across_req_rate_{len_factor}.png"
        plt.savefig(output_file, dpi=150)
        print(f"\nSaved: {output_file}")
        plt.close()

print("\n✓ Done!")
