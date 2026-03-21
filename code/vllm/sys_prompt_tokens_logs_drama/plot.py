#!/usr/bin/env python3
"""
Read results.json and generate a plot:
  Mean DRAM Activity vs Shared Prefix Tokens (sptokens)

results.json structure:
  { model -> engine -> policy -> sptokens (str) -> mean_drama (float | null) }

Lines are labelled  "engine:policy"  (matching the existing plotting convention).
"""

import os
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")
OUT_DIR      = os.path.expanduser("../../../results/sys_prompt_tokens_plots")

# ── Plot styling (colour-blind friendly, same palette as existing script) ───
PALETTE = [
    "#E63946", "#2A9D8F", "#E9C46A", "#8338EC", "#06D6A0", "#FB5607",
    "#3A86FF", "#FF006E",
]
MARKERS    = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":"]

# ── Policy / engine display-name normalisation ───────────────────────────────
def normalise_policy(policy: str) -> str:
    mapping = {
        "dfs-weight":                "DFS-W",
        "lpm":                       "LPM",
        "fcfs":                      "FCFS",
        "chunked_hash_tree_bandit":  "Feather",
        "dynamic_batching":          "DB",
        "family_load":               "FamilyLoad",
    }
    return mapping.get(policy, policy)

def normalise_engine(engine: str) -> str:
    return {"vllm": "vLLM", "sglang": "SGLang"}.get(engine, engine)

SKIP_POLICIES = {"chunked_hash_tree_heuristic", "fcfs", "family_load"}

# ── Load JSON ────────────────────────────────────────────────────────────────
with open(RESULTS_FILE, "r") as f:
    raw = json.load(f)

# ── Flatten into  label -> {tok_int -> mean_drama} ──────────────────────────
data = defaultdict(dict)   # label -> {int tok -> float|None}

for model, engines in raw.items():
    for engine, policies in engines.items():
        eng_label = normalise_engine(engine)
        for policy, sptokens_map in policies.items():
            if policy in SKIP_POLICIES:
                continue
            pol_label = normalise_policy(policy)
            label = f"{eng_label}:{pol_label}"
            for tok_str, mean_val in sptokens_map.items():
                tok = int(tok_str)
                # If the same (label, tok) appears in multiple models, average
                if tok in data[label]:
                    existing = data[label][tok]
                    if existing is None or mean_val is None:
                        data[label][tok] = None
                    else:
                        data[label][tok] = (existing + mean_val) / 2.0
                else:
                    data[label][tok] = mean_val

all_tokens = sorted({tok for series in data.values() for tok in series})
print(f"Labels  : {sorted(data.keys())}")
print(f"Tokens  : {all_tokens}")

# ── Plot ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
outpath = os.path.join(OUT_DIR, "mean_dram_vs_sys_prompt_tokens.png")

fig, ax = plt.subplots(figsize=(12, 6))

for i, label in enumerate(sorted(data.keys())):
    color     = PALETTE[i % len(PALETTE)]
    marker    = MARKERS[i % len(MARKERS)]
    linestyle = LINESTYLES[(i // len(MARKERS)) % len(LINESTYLES)]

    xs, ys = [], []
    for tok in all_tokens:
        v = data[label].get(tok)
        if isinstance(v, float):
            v = round(v, 4)
        xs.append(tok)
        ys.append(v if v is not None else np.nan)

    ax.plot(
        xs, ys,
        marker=marker,
        linestyle=linestyle,
        linewidth=2.2,
        markersize=7,
        markeredgecolor="black",
        markeredgewidth=0.7,
        color=color,
        label=label,
    )

ax.set_xscale("log")
ax.set_xlabel("Number of Shared Prefix Tokens", fontsize=19)
ax.set_ylabel("Mean DRAM Activity",             fontsize=20)

ax.set_xticks(all_tokens)
ax.set_xticklabels(all_tokens, rotation=30)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.tick_params(axis="both", labelsize=19)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

ax.legend(fontsize=18, framealpha=0.3, ncol=1)
ax.grid(True, linewidth=0.7, linestyle="--")

plt.tight_layout()
fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)

print(f"\nSaved: {outpath}")