import pandas as pd
import matplotlib.pyplot as plt
import re

# List of input files
files = [".out_4096", ".out_8192", ".out_16384", ".out_32768"]

# Prepare a 4x2 grid for 8 plots
fig, axes = plt.subplots(nrows=len(files), ncols=2, figsize=(15, 20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, file_path in enumerate(files):
    # Extract batch size (last number in the filename)
    match = re.search(r"(\d+)", file_path)
    batch_size = match.group(1) if match else "?"

    # --- Read data ---
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) == 4:
                    _, _, smact, smocc = parts
                    data.append((float(smact), float(smocc)))

    df = pd.DataFrame(data, columns=["SMACT", "SMOCC"])

    # --- Plot SMACT ---
    ax1 = axes[i, 0]
    ax1.plot(df["SMACT"], marker='o', linestyle='-', color='b')
    ax1.set_title(f"Matrix Size {batch_size}: SM Active (SMACT)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SMACT")
    ax1.grid(True)

    # --- Plot SMOCC ---
    ax2 = axes[i, 1]
    ax2.plot(df["SMOCC"], marker='o', linestyle='-', color='r')
    ax2.set_title(f"Matrix Size {batch_size}: SM Occupancy (SMOCC)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("SMOCC")
    ax2.grid(True)

plt.tight_layout()
plt.show()
plt.savefig("matmul_compute.png")