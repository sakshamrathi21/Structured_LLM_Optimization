# import pandas as pd
# import matplotlib.pyplot as plt
# import re

# # List of input files
# files = [".out_1000", ".out_1000_family"]

# # Prepare a 4x2 grid for 8 plots
# fig, axes = plt.subplots(nrows=len(files), ncols=2, figsize=(15, 10))
# fig.subplots_adjust(hspace=0.4, wspace=0.3)

# for i, file_path in enumerate(files):
#     # Extract batch size (last number in the filename)
#     match = re.search(r"(\d+)", file_path)
#     batch_size = match.group(1) if match else "?"
#     # --- Read data ---
#     data = []
#     with open(file_path, "r") as f:
#         for line in f:
#             if line.strip() and not line.startswith("#"):
#                 parts = line.split()
#                 if len(parts) == 4:
#                     _, _, smact, smocc = parts
#                     data.append((float(smact), float(smocc)))

#     df = pd.DataFrame(data, columns=["SMACT", "SMOCC"])

#     # --- Plot SMACT ---
#     ax1 = axes[i, 0]
#     ax1.plot(df["SMACT"], marker='o', linestyle='-', color='b')
#     ax1.set_title(f"Max Batch Size {batch_size}: SM Active (SMACT)")
#     ax1.set_xlabel("Time")
#     ax1.set_ylabel("SMACT")
#     ax1.grid(True)

#     # --- Plot SMOCC ---
#     ax2 = axes[i, 1]
#     ax2.plot(df["SMOCC"], marker='o', linestyle='-', color='r')
#     ax2.set_title(f"Max Batch Size {batch_size}: SM Occupancy (SMOCC)")
#     ax2.set_xlabel("Time")
#     ax2.set_ylabel("SMOCC")
#     ax2.grid(True)

# plt.tight_layout()
# plt.show()
# plt.savefig("try.png")


import re
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

log_file = ".out"

req_pattern = re.compile(r"Running req: (\d+) (\d+)")
time_pattern = re.compile(r"Scheduler called after:\s+([\d.]+)")

batches = []
current_batch = []

# -------------------------------
# Step 1: Parse log file
# -------------------------------
with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        
        m = req_pattern.search(line)
        if m:
            req_id = int(m.group(1))
            tokens = int(m.group(2))
            current_batch.append((req_id, tokens))
            continue
        
        t = time_pattern.search(line)
        if t:
            time_taken = float(t.group(1))
            batches.append({"reqs": current_batch, "time": time_taken})
            current_batch = []

# -------------------------------
# Step 2: Classify batches using next batch
# -------------------------------
batch_times = defaultdict(list)

for i in range(len(batches) - 1):
    curr = {req_id: tokens for req_id, tokens in batches[i]["reqs"]}
    nxt = {req_id: tokens for req_id, tokens in batches[i + 1]["reqs"]}
    time_taken = batches[i]["time"]

    prefix_prefill = prefill = decode = 0

    for req_id, curr_tokens in curr.items():
        if req_id in nxt:
            delta = nxt[req_id] - curr_tokens
            if delta == 1:
                decode += 1
            elif delta < 100:
                prefill += 1
            else:
                prefix_prefill += 1
        else:
            prefill += 1

    batch_type = (prefix_prefill, prefill, decode)
    if batch_type == (0, 0, 0):
        continue
    batch_times[batch_type].append(time_taken)

# -------------------------------
# Step 3: Compute averages
# -------------------------------
print("Average times per batch type:")
avg_data = []
for bt, times in batch_times.items():
    avg_time = sum(times) / len(times)
    avg_data.append((*bt, avg_time))
    print(f"{bt}: {avg_time:.6f} seconds (over {len(times)} batches)")

# -------------------------------
# Step 4: Helper for averaging along one dimension
# -------------------------------
def avg_over_dimension(index, filter_fn=None):
    """Compute average time over one dimension, optionally filtering points."""
    dim_data = defaultdict(list)
    for d in avg_data:
        if filter_fn and not filter_fn(d):
            continue
        dim_data[d[index]].append(d[3])
    xs = sorted(dim_data.keys())
    ys = [sum(dim_data[k]) / len(dim_data[k]) for k in xs]
    return xs, ys

# 1️⃣ Prefix prefill: use all data
prefix_x, prefix_y = avg_over_dimension(0)

# 2️⃣ Prefill: only include cases with prefix_prefill == 0
prefill_x, prefill_y = avg_over_dimension(1, filter_fn=lambda d: d[0] == 0)

# 3️⃣ Decode: only include cases with prefix_prefill == 0
decode_x, decode_y = avg_over_dimension(2, filter_fn=lambda d: d[0] == 0)

# -------------------------------
# Step 5: Create 2x2 grid for all plots
# -------------------------------
fig = plt.figure(figsize=(12, 10))

# (1,1): 3D Scatter
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
x = [d[0] for d in avg_data]  # prefix_prefill
y = [d[1] for d in avg_data]  # prefill
z = [d[2] for d in avg_data]  # decode
times = [d[3] for d in avg_data]  # average time
sc = ax1.scatter(x, y, z, c=times, cmap='viridis', s=80)
ax1.set_xlabel('Prefix Prefill Count')
ax1.set_ylabel('Prefill Count')
ax1.set_zlabel('Decode Count')
ax1.set_title("3D Scatter: Batch Composition vs Time")
fig.colorbar(sc, ax=ax1, shrink=0.6, pad=0.1, label='Avg Time (s)')

# (1,2): Prefix prefill
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(prefix_x, prefix_y, marker='o', color='tab:blue')
ax2.set_xlabel("Prefix Prefill Count")
ax2.set_ylabel("Average Time (s)")
ax2.set_title("Avg Time vs Prefix Prefill Count")

# (2,1): Prefill (only where prefix_prefill=0)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(prefill_x, prefill_y, marker='o', color='tab:orange')
ax3.set_xlabel("Prefill Count")
ax3.set_ylabel("Average Time (s)")
ax3.set_title("Avg Time vs Prefill Count (prefix_prefill=0)")

# (2,2): Decode (only where prefix_prefill=0)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(decode_x, decode_y, marker='o', color='tab:green')
ax4.set_xlabel("Decode Count")
ax4.set_ylabel("Average Time (s)")
ax4.set_title("Avg Time vs Decode Count (prefix_prefill=0)")

plt.tight_layout()
plt.show()
plt.savefig("try.png")
