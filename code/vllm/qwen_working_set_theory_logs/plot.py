import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LOG_DIR = "." 
OUT_DIR = "../../../results/qwen_working_set_theory_plots/"
os.makedirs(OUT_DIR, exist_ok=True)

def parse_filename(fname):
    m = re.search(r'_tok(\d+)_fam(\d+)_sys(\d+)_', fname)
    if not m:
        return None
    return {"tok": int(m[1]), "fam": int(m[2]), "sys": int(m[3])}

def parse_throughput(text):
    for line in reversed(text.splitlines()):
        m = re.search(r'Throughput Test.*?:\s*([\d.]+)\s*$', line)
        if m:
            return float(m.group(1))
    return None

def parse_kv(text):
    mem_m = re.search(r'Available KV cache memory:\s*([\d.]+)\s*GiB', text)
    tok_m = re.search(r'GPU KV cache size:\s*([\d,]+)\s*tokens', text)
    if mem_m and tok_m:
        return {
            "totalGiB": float(mem_m.group(1)),
            "totalTokens": int(tok_m.group(1).replace(",", ""))
        }
    return None

# --- collect data ---
tp_data = {}
kv_data = {}
kv_ref = None

for fpath in glob.glob(os.path.join(LOG_DIR, "*.log")):
    meta = parse_filename(os.path.basename(fpath))
    if not meta:
        continue
    with open(fpath) as f:
        text = f.read()

    tp = parse_throughput(text)
    kv = parse_kv(text)

    key = (meta["tok"], meta["fam"])

    if tp is not None:
        tp_data[key] = tp

    if kv is not None:
        if kv_ref is None:
            kv_ref = kv
        prefix_tokens = meta["tok"] * meta["fam"]
        frac = prefix_tokens / kv["totalTokens"]
        kv_data[key] = frac * kv["totalGiB"]

# --- build axis labels ---
all_keys = set(tp_data) | set(kv_data)
toks = sorted(set(k[0] for k in all_keys))   # y axis (length)
fams = sorted(set(k[1] for k in all_keys))   # x axis (families)

def build_matrix(data_dict):
    mat = np.full((len(toks), len(fams)), np.nan)
    for i, t in enumerate(toks):
        for j, f in enumerate(fams):
            v = data_dict.get((t, f))
            if v is not None:
                mat[i, j] = v
    return mat

tp_mat = build_matrix(tp_data)
kv_mat = build_matrix(kv_data)

# --- plot helper ---
def draw_heatmap(mat, title, fmt, cmap, outfile):
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(mat, aspect="auto", cmap=cmap,
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, fontsize=20)
    ax.set_yticks(range(len(toks)))
    ax.set_yticklabels(toks, fontsize=20)
    ax.set_xlabel("Prefix groups", fontsize=25)
    ax.set_ylabel("Shared prefix tokens", fontsize=25)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                norm_v = (v - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat) + 1e-9)
                txt_color = "white" if norm_v > 0.55 else "black"
                ax.text(j, i, fmt(v), ha="center", va="center",
                        fontsize=17, color=txt_color, fontweight="500")

    plt.colorbar(im, ax=ax, shrink=0.85).ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved to {outfile}")
    plt.show()

draw_heatmap(tp_mat,
             "Throughput (toks/s)",
             lambda v: f"{v/1000:.1f}k",
             cmap="YlGn",
             outfile=os.path.join(OUT_DIR, "vllm_heatmap_throughput.png"))

# kv cache memory, greater than 1000 switch to G
draw_heatmap(kv_mat,
             "KV cache memory (MB)",
             lambda v: f"{(v*1024):.0f}M" if v < 1 else f"{v:.1f}G",
             cmap="OrRd",
             outfile=os.path.join(OUT_DIR, "vllm_heatmap_kvcache.png"))