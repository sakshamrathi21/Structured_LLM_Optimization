import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LOG_DIR = "." 
OUT_DIR = "../../../results/tiered_prefix_sharing_plots/"
os.makedirs(OUT_DIR, exist_ok=True)

import re

def parse_filename(fname):
    m = re.search(r'_f1_(\d+(?:\.\d+)?)_f2_(\d+(?:\.\d+)?)_2026', fname)
    if not m:
        print(f"Parsing failed: {fname}")
        return None
    return {"f1": float(m[1]), "f2": float(m[2])}

def parse_throughput(text):
    for line in reversed(text.splitlines()):
        pattern = re.compile(r"Average throughput\s*:\s*([\d.]+)\s*req/s", re.IGNORECASE)
        m = pattern.search(line)
        if m:
            return float(m.group(1))*200
    return None

# --- collect data ---
tp_data = {}
kv_ref = None

for fpath in glob.glob(os.path.join(LOG_DIR, "*.log")):
    meta = parse_filename(os.path.basename(fpath))
    if not meta:
        continue
    with open(fpath) as f:
        text = f.read()

    tp = parse_throughput(text)
    key = (meta["f1"], meta["f2"])

    if tp is not None:
        tp_data[key] = tp

print(f"Parsed {len(tp_data)} log files for throughput.")
# --- build axis labels ---
all_keys = set(tp_data)
f1s = sorted(set(k[0] for k in all_keys))   # y axis (length)
f2s = sorted(set(k[1] for k in all_keys))   # x axis (families)

def build_matrix(data_dict):
    mat = np.full((len(f1s), len(f2s)), np.nan)
    for i, f1 in enumerate(f1s):
        for j, f2 in enumerate(f2s):
            v = data_dict.get((f1, f2))
            if v is not None:
                mat[i, j] = v
    return mat

tp_mat = build_matrix(tp_data)

# --- plot helper ---
def draw_heatmap(mat, title, fmt, cmap, outfile):
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(f2s)))
    ax.set_xticklabels(f2s, fontsize=25)
    ax.set_yticks(range(len(f1s)))
    ax.set_yticklabels(f1s, fontsize=25)
    ax.set_xlabel("f2", fontsize=30)
    ax.set_ylabel("f1", fontsize=30)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                norm_v = (v - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat) + 1e-9)
                txt_color = "white" if norm_v > 0.55 else "black"
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=20, color=txt_color, fontweight="500")
    # change the fontsize in the colour bar
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02).ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved to {outfile}")
    plt.show()

draw_heatmap(tp_mat,
             "Throughput (toks/s)",
             lambda v: f"{v:.0f}",
             cmap="YlGn",
             outfile=os.path.join(OUT_DIR, "vllm_heatmap_throughput.png"))