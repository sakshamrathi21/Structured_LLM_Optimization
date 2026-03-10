"""
Throughput Plotting Script
==========================
- Filter the DataFrame by any column/value combinations
- Choose the X axis column
- Y axis is fixed: throughput
- Legend is fixed: policy

Usage example:
    python plot_throughput.py
    
Or import and call plot_throughput() directly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit these before running
# ─────────────────────────────────────────────

CSV_PATH = "/home/saksham/CS496/code/vllm/paper_eval_results.csv"          # path to your CSV file

X_AXIS   = "families"         # column to use as X axis
                               # e.g. "req/s", "max_batch", "max_new_tokens",
                               #      "num_requests", "families", ...

# Filters: list of (column, value) tuples — ALL are applied (AND logic).
# Leave empty [] to use the full DataFrame.
FILTERS  = [
    # ("engine",   "vllm"),
    ("sys_prompt_tokens", 5000),
    ("num_requests", 7500)
]

OUTPUT_PATH = "../../results/throughput_vs_families_length_5000.png"   # where to save the figure

# ─────────────────────────────────────────────


def load_and_filter(csv_path: str, filters: list[tuple]) -> pd.DataFrame:
    """Load CSV and apply column-value filters."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from '{csv_path}'.")
    for col, val in filters:
        if col not in df.columns:
            raise ValueError(f"Filter column '{col}' not found. Available: {list(df.columns)}")
        df = df[df[col] == val]
        print(f"  After filter ({col} == {val!r}): {len(df)} rows remain.")
    print(df.head())
    df["policy"] = df["engine"] + ":" + df["policy"]  # combine for legend
    return df


def plot_throughput(
    csv_path:    str        = CSV_PATH,
    x_axis:      str        = X_AXIS,
    filters:     list       = FILTERS,
    output_path: str        = OUTPUT_PATH,
) -> None:
    """
    Plot throughput vs. x_axis, one line per policy, and save to output_path.

    Parameters
    ----------
    csv_path    : path to the input CSV
    x_axis      : column name for the X axis
    filters     : list of (column, value) tuples applied as AND filters
    output_path : file path for the saved figure (PNG / PDF / SVG …)
    """
    df = load_and_filter(csv_path, filters)

    if df.empty:
        print("⚠  DataFrame is empty after filtering — nothing to plot.")
        return

    if x_axis not in df.columns:
        raise ValueError(f"X-axis column '{x_axis}' not found. Available: {list(df.columns)}")

    # ── aggregate: mean throughput per (policy, x_axis) ──────────────────
    grouped = (
        df.groupby(["policy", x_axis], sort=True)["throughput"]
        .mean()
        .reset_index()
    )

    policies = sorted(grouped["policy"].unique())

    # ── colour palette ────────────────────────────────────────────────────
    cmap   = plt.get_cmap("tab10")
    colors = {p: cmap(i) for i, p in enumerate(policies)}

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in policies:
        sub = grouped[grouped["policy"] == policy].sort_values(x_axis)
        ax.plot(
            sub[x_axis],
            sub["throughput"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors[policy],
            label=policy,
        )

    # ── labels & formatting ───────────────────────────────────────────────
    ax.set_xlabel(x_axis.replace("_", " ").title(), fontsize=15)
    ax.set_ylabel("Throughput (toks/s)", fontsize=15)
    # set fontsize of ticks
    ax.tick_params(axis="both", which="major", labelsize=15)
    # ax.set_title(f"Throughput vs. {x_axis.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    # position legend to be top-right, but inside the plot area
    # ax.legend(loc="upper left", fontsize=15)
    ax.legend(fontsize=15)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✓ Plot saved to '{output_path}'")
    plt.show()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    plot_throughput(
        csv_path    = CSV_PATH,
        x_axis      = X_AXIS,
        filters     = FILTERS,
        output_path = OUTPUT_PATH,
    )