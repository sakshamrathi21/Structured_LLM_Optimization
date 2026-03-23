import os
import re
import json
import numpy as np

LOG_DIR = "."
OUTPUT_FILE = "results.json"

# -------------------------------
# Engine → Policy mapping
# -------------------------------
VLLM_POLICIES = {"fcfs", "dynamic_batching", "chunked_hash_tree_bandit", "family_load"}
SGLANG_POLICIES = {"lpm", "dfs-weight", "fcfs"}

def infer_engine(policy: str) -> str:
    """
    Infer the engine from the policy name.
    'fcfs' is ambiguous — per spec, treat it as sglang.
    Policies in VLLM_POLICIES (excluding fcfs) → vllm.
    Policies in SGLANG_POLICIES → sglang.
    """
    if policy == "fcfs":
        return "sglang"
    if policy in VLLM_POLICIES:
        return "vllm"
    if policy in SGLANG_POLICIES:
        return "sglang"
    return "unknown"

# -------------------------------
# DRAMA reader
# -------------------------------
def read_drama(filename):
    drama = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("ID"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                drama.append(float(parts[-3]))
            except ValueError:
                continue

    drama = np.array(drama)
    nz = np.nonzero(drama)[0]
    if len(nz) == 0:
        raise ValueError(f"{filename}: DRAMA column is all zeros")
    return drama

# -------------------------------
# Metadata extraction
# -------------------------------
# Match the model name between "model-" and the next known fixed field
MODEL_PATTERN  = r"model-(.+?)_families-"
POLICY_PATTERN = r"policy-([a-zA-Z0-9\-_]+?)_maxnew"
SPTOKENS_PATTERN = r"sptokens-(\d+)"

def extract_metadata(filename):
    model_match    = re.search(MODEL_PATTERN,    filename)
    policy_match   = re.search(POLICY_PATTERN,   filename)
    sptokens_match = re.search(SPTOKENS_PATTERN, filename)

    if not (model_match and policy_match and sptokens_match):
        return None, None, None, None

    model    = model_match.group(1)
    policy   = policy_match.group(1)
    sptokens = sptokens_match.group(1)
    engine   = infer_engine(policy)

    return model, engine, policy, sptokens

# -------------------------------
# Main processing
# -------------------------------
def main():
    results = {}

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue

        model, engine, policy, sptokens = extract_metadata(filename)
        if model is None:
            print(f"Skipping (no metadata): {filename}")
            continue

        filepath = os.path.join(LOG_DIR, filename)
        print(f"Processing : {filename}")
        print(f"  model={model}, engine={engine}, policy={policy}, sptokens={sptokens}")

        try:
            drama    = read_drama(filepath)
            mean_val = float(np.sum(drama))
            print(f"  mean_drama={mean_val:.4f}  (n={len(drama)})")
        except Exception as e:
            print(f"  ERROR: {e}")
            mean_val = None

        # Nested structure: model → engine → policy → sptokens → mean
        results \
            .setdefault(model, {}) \
            .setdefault(engine, {}) \
            .setdefault(policy, {})[sptokens] = mean_val

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()