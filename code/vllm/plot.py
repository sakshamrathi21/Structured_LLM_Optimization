import re
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
INPUT_FILE = "vllm_multiple_families_varying_workloads_output.txt"
# ----------------------------

# More specific regex that handles all the intermediate fields
pattern = re.compile(
    r"Throughput Test.*?"
    r"(?P<req>\d+)\s+req/sec.*?"
    r"\(len factor:\s+(?P<len_factor>[\d.]+)\).*?"
    r"\(model_name:.*?\).*?"  # Match and skip model_name
    r"\(policy:\s+(?P<policy>[^)]+)\).*?"
    r"\(max_new_tokens:.*?\).*?"  # Match and skip max_new_tokens
    r"\(num_prompts:.*?\).*?"  # Match and skip num_prompts
    r":\s+(?P<throughput>[\d.]+)"
)

# data[len_factor][policy] = list of (req/sec, throughput)
data = defaultdict(lambda: defaultdict(list))

# ---------- PARSE FILE ----------
matched_lines = 0
total_lines = 0

with open(INPUT_FILE, "r") as f:
    for line in f:
        total_lines += 1
        match = pattern.search(line)
        if not match:
            if total_lines <= 3:
                print(f"Line {total_lines} didn't match: {line.strip()}")
            continue
        
        matched_lines += 1
        req = int(match.group("req"))
        len_factor = float(match.group("len_factor"))
        policy = match.group("policy")
        throughput = float(match.group("throughput"))
        if policy == "fcfs":
            policy = "vllm"
        elif policy == "radix_cost":
            policy = "chunked hash tree"
        elif policy == "family_load":
            policy = "user gives out family IDs"
        elif policy == "dynamic_batching":
            policy = "dynamic batching"
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

# ---------- PLOTTING ----------
if not data:
    print("\n⚠️  ERROR: No data was parsed!")
else:
    for len_factor, policies in sorted(data.items()):
        plt.figure(figsize=(10, 6))
        
        for policy, points in policies.items():
            if not points:
                continue
                
            # sort by req/sec for proper lines
            points.sort(key=lambda x: x[0])
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            
            print(f"\nPlotting {policy} (len_factor={len_factor}): {len(points)} points")
            print(f"  X (req/sec): {x}")
            print(f"  Y (throughput): {y}")
            
            plt.plot(x, y, marker='o', label=policy, linewidth=2, markersize=6)
        
        plt.xlabel("Requests per Second", fontsize=12)
        plt.ylabel("Throughput (req/sec)", fontsize=12)
        plt.title(f"Throughput vs Request Rate (average shared prefix length = {int(len_factor*1300)} tokens)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = f"../../results/throughput_across_req_rate_{len_factor}.png"
        plt.savefig(output_file, dpi=150)
        print(f"\nSaved: {output_file}")
        plt.close()

print("\n✓ Done!")