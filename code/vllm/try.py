import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- READ AND PARSE ----------
data = []
pattern = re.compile(
    r"Throughput Test with (\d+) families, (\d+) req/sec and max_batch_size of (\d+) \(randomness: (\d+)\): ([\d\.]+)"
)

tuple_to_throughput = {}

with open("vllm_multiple_families_varying_workloads_output.txt") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            families, req, batch, rand, value = m.groups()
            tuple_to_throughput[(int(families), int(req), int(batch), int(rand))] = float(value)
for (key, value) in tuple_to_throughput.items():
    families, req, batch, rand = key
    data.append({
        "families": families,
        "req_per_sec": req,
        "batch_size": batch,
        "randomness": rand,
        "throughput": value
    })

df = pd.DataFrame(data)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ---------- PLOT: Separate for each number of families ----------
for families, df_fam in df.groupby("families"):
    plt.figure(figsize=(15, 10))
    groups = list(df_fam.groupby(["batch_size", "randomness"]))
    colors = cm.get_cmap('tab20', len(groups))  # or try 'husl', 'viridis', etc.

    for idx, ((batch, rand), group) in enumerate(groups):
        group = group.sort_values("req_per_sec")
        plt.plot(
            group["req_per_sec"],
            group["throughput"],
            marker="o",
            color=colors(idx),
            label=f"Batch={batch}, Rand={rand}"
        )

    plt.xlabel("Requests per Second")
    plt.ylabel("Throughput")
    plt.title(f"Throughput vs Requests per Second (Families = {families})")
    plt.grid(True)

    # ---- Legend outside plot ----
    plt.legend(
        bbox_to_anchor=(1.05, 1),  # shift legend to right
        loc='upper left',
        borderaxespad=0.
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave room for legend
    plt.show()
    plt.savefig(f"../../results/throughput_vs_req_batch_rand_families_{families}.png")


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="batch_size", y="throughput", hue="randomness")
plt.title("Throughput Distribution Across Randomness Levels")
plt.show()
plt.savefig("../../results/throughput_boxplot_batch_size_randomness.png")
