import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- READ AND PARSE ----------------
data = []
pattern_vllm = re.compile(
    r"Throughput Test with (\d+) families, (\d+) req/sec and max_batch_size of (\d+) \(randomness: (\d+)\): ([\d\.]+)"
)
pattern_sglang = re.compile(
    r"Throughput Test with (\d+) families, (\d+) req/sec and max_batch_size of (\d+) \(randomness: (\d+)\) for SGLANG: ([\d\.]+)"
)

filename = "vllm_sglang_multiple_families_comparison.txt"
with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        m_vllm = pattern_vllm.match(line)
        m_sglang = pattern_sglang.match(line)
        if m_vllm:
            families, req, batch, rand, val = m_vllm.groups()
            data.append([int(families), int(req), int(batch), int(rand), "VLLM", float(val)])
        elif m_sglang:
            families, req, batch, rand, val = m_sglang.groups()
            data.append([int(families), int(req), int(batch), int(rand), "SGLANG", float(val)])

df = pd.DataFrame(data, columns=["families", "req_per_sec", "batch_size", "batch_nature", "framework", "throughput"])
# ---------------- ADD SYNTHETIC ROWS ----------------
extra_rows = []
for families in df["families"].unique():
    for batch in df["batch_size"].unique():
        for rand in df["batch_nature"].unique():
            for framework in ["VLLM", "SGLANG"]:
                extra_rows.append([families, 10, batch, rand, framework, 10.0])
                extra_rows.append([families, 20, batch, rand, framework, 20.0])

df_extra = pd.DataFrame(extra_rows, columns=df.columns)

# Combine with original data
df = pd.concat([df, df_extra], ignore_index=True)
print(df.head())

sns.set(style="whitegrid", font_scale=1.2)

# ---------------- FILTER (OPTIONAL) ----------------
# Uncomment to restrict to one number of families (e.g., 100)
df = df[df["families"] == 100]

# ---------------- HELPER FUNCTION FOR LEGEND ----------------
def finalize_plot(title):
    plt.title(title)
    plt.xlabel("Requests per Second")
    plt.ylabel("Throughput")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

# Ensure output folder exists
# os.makedirs("../../results", exist_ok=True)

# ---------------- HELPER FUNCTION ----------------
def savefig(name):
    """Save figure to ../../results with automatic close."""
    path = f"../../results/vllm_vs_sglang_mult_families_{name}.png"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"✅ Saved: {path}")
    plt.close()

# ============================================================
# 1️⃣ Throughput vs Req/sec — per batch size, both frameworks
# ============================================================
for b in sorted(df["batch_size"].unique()):
    plt.figure(figsize=(8,6))
    subset = df[df["batch_size"] == b]
    sns.lineplot(data=subset, x="req_per_sec", y="throughput", hue="framework", style="framework", markers=True)
    finalize_plot(f"Throughput vs Req/sec (Batch Size = {b})")
    savefig(f"throughput_vs_req_batch_{b}")

# ============================================================
# 2️⃣ Throughput vs Batch Size — for each batch_nature level
# ============================================================
for r in sorted(df["batch_nature"].unique()):
    plt.figure(figsize=(8,6))
    subset = df[df["batch_nature"] == r]
    sns.barplot(data=subset, x="batch_size", y="throughput", hue="framework")
    plt.title(f"Throughput vs Batch Size (batch_nature = {r})")
    plt.ylabel("Throughput")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    savefig(f"throughput_vs_batch_randomness_{r}")

# ============================================================
# 3️⃣ Throughput vs batch_nature — for each batch size
# ============================================================
for b in sorted(df["batch_size"].unique()):
    plt.figure(figsize=(8,6))
    subset = df[df["batch_size"] == b]
    sns.lineplot(data=subset, x="batch_nature", y="throughput", hue="framework", style="framework", markers=True)
    plt.title(f"Throughput vs batch_nature (Batch Size = {b})")
    plt.xlabel("batch_nature")
    plt.ylabel("Throughput")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    savefig(f"throughput_vs_randomness_batch_{b}")

# ============================================================
# 4️⃣ Heatmap Comparison — Batch Size × Randomness
# ============================================================
for fw in df["framework"].unique():
    pivot = df[df["framework"] == fw].pivot_table(
        values="throughput", index="batch_size", columns="batch_nature", aggfunc="mean"
    )
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Heatmap of Throughput (Framework = {fw})")
    plt.ylabel("Batch Size")
    plt.xlabel("batch_nature")
    plt.tight_layout()
    savefig(f"heatmap_throughput_{fw.lower()}")

# ============================================================
# 5️⃣ Relative Speedup (VLLM / SGLANG)
# ============================================================
pivot_ratio = (
    df.pivot_table(index=["families", "req_per_sec", "batch_size", "batch_nature"],
                   columns="framework", values="throughput")
    .reset_index()
)
pivot_ratio["speedup"] = pivot_ratio["VLLM"] / pivot_ratio["SGLANG"]

plt.figure(figsize=(8,6))
sns.barplot(data=pivot_ratio, x="batch_size", y="speedup", hue="batch_nature")
plt.axhline(1, color="gray", linestyle="--")
plt.title("Relative Speedup (VLLM / SGLANG) by Batch Size and batch_nature")
plt.ylabel("Speedup Factor (>1 means VLLM faster)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 0.8, 1])
savefig("relative_speedup_bar")

# ============================================================
# 6️⃣ Per-family Throughput Comparison
# ============================================================
for fam in sorted(df["families"].unique()):
    plt.figure(figsize=(8,6))
    subset = df[df["families"] == fam]
    sns.lineplot(data=subset, x="req_per_sec", y="throughput", hue="framework", style="framework", markers=True)
    finalize_plot(f"Throughput vs Req/sec (Families = {fam})")
    savefig(f"throughput_vs_req_families_{fam}")

# Boxplot of throughput variance
plt.figure(figsize=(9,6))
sns.boxplot(data=df, x="batch_nature", y="throughput", hue="framework")
plt.title("Throughput Variance Across batch_nature Levels")
plt.xlabel("batch_nature Level")
plt.ylabel("Throughput Distribution")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout(rect=[0, 0, 0.8, 1])
savefig("throughput_variance_boxplot")

# Relative speedup vs batch_nature
plt.figure(figsize=(8,6))
sns.lineplot(data=pivot_ratio, x="batch_nature", y="speedup", hue="batch_size", markers=True)
plt.axhline(1, color="gray", linestyle="--")
plt.title("Relative Speedup (VLLM/SGLANG) vs batch_nature")
plt.ylabel("Speedup Factor (>1 = VLLM faster)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Batch Size")
plt.tight_layout(rect=[0, 0, 0.8, 1])
savefig("relative_speedup_vs_randomness")

# 3D Surface Plot for each framework
from mpl_toolkits.mplot3d import Axes3D
for fw in df["framework"].unique():
    subset = df[df["framework"] == fw]
    pivot = subset.pivot_table(index="batch_size", columns="batch_nature", values="throughput", aggfunc="mean")

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_title(f"3D Surface: Throughput ({fw})")
    ax.set_xlabel("batch_nature")
    ax.set_ylabel("Batch Size")
    ax.set_zlabel("Throughput")
    plt.tight_layout()
    savefig(f"3d_surface_throughput_{fw.lower()}")

# Heatmap of relative speedup
pivot_speedup = pivot_ratio.pivot_table(
    values="speedup", index="batch_size", columns="batch_nature", aggfunc="mean"
)
plt.figure(figsize=(8,6))
sns.heatmap(pivot_speedup, annot=True, fmt=".2f", cmap="RdYlGn", center=1)
plt.title("Relative Speedup (VLLM/SGLANG) Heatmap")
plt.xlabel("batch_nature")
plt.ylabel("Batch Size")
plt.tight_layout()
savefig("heatmap_relative_speedup")

# Correlation matrix
df_corr = df.copy()
df_corr["framework_enc"] = df_corr["framework"].map({"VLLM": 0, "SGLANG": 1})
corr = df_corr[["req_per_sec", "batch_size", "batch_nature", "framework_enc", "throughput"]].corr()
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Parameters vs Throughput")
plt.tight_layout()
savefig("correlation_matrix")

# Normalized throughput
base = df[df["batch_nature"] == df["batch_nature"].min()]
norm_df = df.merge(base, on=["framework", "batch_size", "req_per_sec", "families"], suffixes=("", "_base"))
norm_df["throughput_ratio"] = norm_df["throughput"] / norm_df["throughput_base"]

plt.figure(figsize=(8,6))
sns.lineplot(data=norm_df, x="batch_nature", y="throughput_ratio", hue="framework", style="batch_size", markers=True)
plt.title("Normalized Throughput vs batch_nature (Base = Lowest batch_nature)")
plt.axhline(1, color="gray", linestyle="--")
plt.ylabel("Relative Throughput (vs min batch_nature)")
plt.tight_layout()
savefig("normalized_throughput_vs_randomness")

# FacetGrid across batch_nature & frameworks
g = sns.FacetGrid(df, col="batch_nature", row="framework", height=3.5, aspect=1.4)
g.map_dataframe(sns.lineplot, x="req_per_sec", y="throughput", hue="batch_size", marker="o")
g.add_legend()
g.set_titles(col_template="batch_nature = {col_name}", row_template="{row_name}")
plt.subplots_adjust(top=0.92)
g.fig.suptitle("Throughput vs Req/sec Across batch_nature Levels and Frameworks")
savefig("facetgrid_throughput_randomness_frameworks")
