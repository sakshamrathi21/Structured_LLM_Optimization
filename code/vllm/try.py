import pandas as pd
import matplotlib.pyplot as plt
import re

files = [".out_good", ".out_bad"]

fig, axes = plt.subplots(nrows=len(files), ncols=4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for i, file_path in enumerate(files):
    match = re.search(r"(\d+)", file_path)
    batch_size = match.group(1) if match else "?"
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4 and len(parts) <= 7:
                    smact = parts[2]
                    smocc = parts[3]
                    tenso = parts[4]
                    drama = parts[5]
                    try:
                        data.append((float(smact), float(smocc), float(tenso), float(drama)))
                    except ValueError:
                        continue

    df = pd.DataFrame(data, columns=["SMACT", "SMOCC", "TENSO", "DRAMA"])

    ax1 = axes[i, 0]
    ax1.plot(df["SMACT"], marker='o', linestyle='-', color='b')
    ax1.set_title(f"SM Active (SMACT)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("SMACT")
    ax1.grid(True)

    ax2 = axes[i, 1]
    ax2.plot(df["SMOCC"], marker='o', linestyle='-', color='r')
    ax2.set_title(f"SM Occupancy (SMOCC)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("SMOCC")
    ax2.grid(True)

    ax3 = axes[i, 2]
    ax3.plot(df["TENSO"], marker='o', linestyle='-', color='g')
    ax3.set_title(f"Tensor Utilization (TENSO)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("TENSO")
    ax3.grid(True)  

    ax4 = axes[i, 3]
    ax4.plot(df["DRAMA"], marker='o', linestyle='-', color='m')
    ax4.set_title(f"DRAM Accesses (DRAMA)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("DRAMA")
    ax4.grid(True)
    print(f"Average DRAM accesses for {file_path}: {df['DRAMA'].mean()}")
plt.tight_layout()
plt.savefig("../../results/mult_fam_drama.png")