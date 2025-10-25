import pandas as pd
import matplotlib.pyplot as plt
import re

files = [".out_1_100", ".out_1_500", ".out_1_700", ".out_10_100", ".out_10_500", ".out_10_700"]

fig, axes = plt.subplots(nrows=len(files), ncols=3, figsize=(20,20))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, file_path in enumerate(files):
    # Extract batch size (last number in the filename)
    match = re.search(r"(\d+)", file_path)
    family_size = match.group(1) if match else "?"
    batch_size = re.search(r"_(\d+)$", file_path).group(1) if re.search(r"_(\d+)$", file_path) else "?"

    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 5 and len(parts) <= 7:
                    smact = parts[2]
                    smocc = parts[3]
                    tenso = parts[4]
                    try:
                        data.append((float(smact), float(smocc), float(tenso)))
                    except ValueError:
                        continue

    df = pd.DataFrame(data, columns=["SMACT", "SMOCC", "TENSO"])

    ax1 = axes[i, 0]
    ax1.plot(df["SMACT"], marker='o', linestyle='-', color='b')
    ax1.set_title(f"Family Size {family_size}, Max Batch Size {batch_size}: SM Active (SMACT)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SMACT")
    ax1.grid(True)

    ax2 = axes[i, 1]
    ax2.plot(df["SMOCC"], marker='o', linestyle='-', color='r')
    ax2.set_title(f"Family Size {family_size}, Max Batch Size {batch_size}: SM Occupancy (SMOCC)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("SMOCC")
    ax2.grid(True)

    ax3 = axes[i, 2]
    ax3.plot(df["TENSO"], marker='o', linestyle='-', color='g')
    ax3.set_title(f"Family Size {family_size}, Max Batch Size {batch_size}: Tensor Utilization (TENSO)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("TENSO")
    ax3.grid(True)  
plt.tight_layout()
plt.show()
plt.savefig("../../results/smact_smocc_tenso_different_family_batches.png")