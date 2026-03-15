# #!/usr/bin/env python3
# import sys
# sys.path.insert(0, "/usr/share/datacenter-gpu-manager-4/bindings/python3")

# import pydcgm
# import dcgm_fields

# # Initialize DCGM handle (embedded)
# handle = pydcgm.DcgmHandle()  # auto-connects to running DCGM daemon

# # Get all GPU IDs
# gpu_ids = [0]  # for simplicity, or list all GPUs if needed

# # Create a group for these GPUs
# group = pydcgm.DcgmGroup(handle, "l2_cache_group", gpu_ids)

# # Add field for L2 cache hit rate
# field_id = dcgm_fields.DCGM_FI_DEV_DEC_UTIL
# group.AddFields([field_id])

# # Update metrics
# group.UpdateAllLatestValues()

# # Fetch and print latest values
# latest_values = group.GetLatestValues()
# for val in latest_values:
#     print(f"GPU {val.entityId}: L2 Cache Hit Rate = {val.value:.2f}%")


import subprocess
import json

# Run dcgmi stats query for 1 GPU, 1 iteration
cmd = [
    "dcgmi",
    "dmon",
    "-e", "0",       # GPU 0
    "-s", "l2",      # L2 cache utilization
    "-i", "1",       # interval 1 second
    "-c", "1",       # 1 sample
    "--json"         # output JSON for parsing
]

proc = subprocess.run(cmd, capture_output=True, text=True)

if proc.returncode != 0:
    print("Error:", proc.stderr)
else:
    try:
        data = json.loads(proc.stdout)
        print(json.dumps(data, indent=4))
    except json.JSONDecodeError:
        print("Raw output:", proc.stdout)

