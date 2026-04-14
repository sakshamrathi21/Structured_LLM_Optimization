import os, re, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Config ───────────────────────────────────────────────────────────────────
LOG_DIR   = os.path.expanduser("~/CS496/code/vllm/pat_logs")
JSONL_PATH = os.path.join(LOG_DIR, "/home/saksham/CS496/code/vllm/pat_logs/pat_fcfs.json")
JSONL_PATH2 = os.path.join(LOG_DIR, "/home/saksham/CS496/code/vllm/pat_logs/pat_feather.json")
OUT_DIR   = os.path.expanduser("../../../results/req_per_sec_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Regexes ──────────────────────────────────────────────────────────────────
RE_FILENAME   = re.compile(r"^run_rate_(?P<rps>\d+)\.log$")
RE_VLLM_TPUT  = re.compile(r"Throughput Test.*?:\s*([\d.]+)\s*$")
RE_SGL_TPUT   = re.compile(r"Throughput:\s*([\d.]+)\s*toks/s")

# ── Parse a single log for throughput ────────────────────────────────────────
def parse_log_throughput(path):
    throughput = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_VLLM_TPUT.search(line)
            if m:
                throughput = float(m.group(1))
                continue
            m = RE_SGL_TPUT.search(line)
            if m:
                throughput = float(m.group(1))
    return throughput

# ── Collect Feather data from log files ──────────────────────────────────────
feather_data = {}   # rps -> throughput

for path in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
    m = RE_FILENAME.match(os.path.basename(path))
    if not m:
        continue
    rps = int(m.group("rps"))
    tput = parse_log_throughput(path)
    if tput is not None:
        # Keep highest throughput if multiple files for same rps
        if rps not in feather_data or tput > feather_data[rps]:
            feather_data[rps] = tput
        print(f"  Feather  rps={rps:4d}  tput={tput:.2f}")

# ── Collect FLASH_ATTN / PREFIX_ATTN from JSONL ──────────────────────────────
json_data = {}   # backend -> { rps -> total_token_throughput }

if os.path.exists(JSONL_PATH):
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") != "success":
                continue
            cfg     = obj.get("config", {})
            backend = cfg.get("ATTENTION_BACKEND", "UNKNOWN")
            if "flash" in backend.lower():
                backend = "FCFS"
            else:
                backend = "PAT"
            rps     = float(cfg.get("request_rate", 0))
            tput    = obj["metrics"].get("output_token_throughput")
            if tput is None:
                continue
            if backend not in json_data:
                json_data[backend] = {}
            if rps not in json_data[backend] or tput > json_data[backend][rps]:
                json_data[backend][rps] = tput
            print(f"  {backend:12s}  rps={rps:.1f}  tput={tput:.2f}")
else:
    print(f"  JSONL not found: {JSONL_PATH}")

if os.path.exists(JSONL_PATH2):
    with open(JSONL_PATH2, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("status") != "success":
                continue
            cfg     = obj.get("config", {})
            backend = cfg.get("ATTENTION_BACKEND", "UNKNOWN")
            if "flash" in backend.lower():
                backend = "FCFS"
            else:
                backend = "Feather+PAT"
            rps     = float(cfg.get("request_rate", 0))
            tput    = obj["metrics"].get("output_token_throughput")
            if tput is None:
                continue
            if backend not in json_data:
                json_data[backend] = {}
            if rps not in json_data[backend] or tput > json_data[backend][rps]:
                json_data[backend][rps] = tput
            print(f"  {backend:12s}  rps={rps:.1f}  tput={tput:.2f}")
else:
    print(f"  JSONL not found: {JSONL_PATH2}")

# ── Plot ─────────────────────────────────────────────────────────────────────
PALETTE = [
    "#8338EC", "#F4A261", "#FF006E", "#E9C46A", "#06D6A0", "#FB5607",
    "#3A86FF",
]

MARKERS = ["P","v", ">", "s", "D", "^" , "X", "*", "<"]
LINESTYLES = ["-", ":", ":"]

fig, ax = plt.subplots(figsize=(12, 6))

series = []

# Feather
if feather_data:
    xs = sorted(feather_data.keys())
    ys = [feather_data[r] for r in xs]
    series.append(("Feather", xs, ys))

# JSON backends
for backend in sorted(json_data.keys()):
    bd = json_data[backend]
    xs = sorted(bd.keys())
    ys = [bd[r] for r in xs]
    series.append((backend, xs, ys))

for i, (label, xs, ys) in enumerate(series):
    ax.plot(
        xs, ys,
        marker=MARKERS[i % len(MARKERS)],
        linestyle=LINESTYLES[i % len(LINESTYLES)],
        linewidth=5,
        markersize=15,
        markeredgecolor="black",
        markeredgewidth=0.7,
        color=PALETTE[i % len(PALETTE)],
        label=f"vLLM:{label}",
    )

ax.set_xlabel("Input Workload (req/s)", fontsize=35)
ax.set_ylabel("TP (toks/s)", fontsize=35)
ax.tick_params(axis="both", labelsize=30)
# set thousand to K
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}K" if x >= 1000 else f"{int(x)}"))
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
ax.legend(fontsize=29, framealpha=0.3, loc="lower right", ncol=2, columnspacing=0.1)
ax.grid(True, linewidth=0.7, linestyle="--")
ax.set_ylim(bottom=0)
plt.tight_layout()

out = os.path.join(OUT_DIR, "throughput_vs_rps_pat_fcfs_feather.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")