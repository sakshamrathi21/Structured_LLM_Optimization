import os
import re
import json
import numpy as np

LOG_DIR = "."
OUTPUT_FILE = "results.json"

# -------------------------------
# Your DRAMA reader (slightly trimmed)
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
ENGINE_PATTERN = r"model-([^_]+(?:_[^_]+)*)"
POLICY_PATTERN = r"policy-([a-zA-Z0-9\-_]+)"
SPTOKENS_PATTERN = r"sptokens-(\d+)"


def extract_metadata(filename):
    engine_match = re.search(ENGINE_PATTERN, filename)
    policy_match = re.search(POLICY_PATTERN, filename)
    sptokens_match = re.search(SPTOKENS_PATTERN, filename)

    if not (engine_match and policy_match and sptokens_match):
        return None, None, None

    return (
        engine_match.group(1),
        policy_match.group(1),
        sptokens_match.group(1),
    )


# -------------------------------
# Main processing
# -------------------------------
def main():
    results = {}

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue

        engine, policy, sptokens = extract_metadata(filename)

        if engine is None:
            print(f"Skipping: {filename}")
            continue

        filepath = os.path.join(LOG_DIR, filename)

        print(f"Processing: {filename}")
        print(f"  engine={engine}, policy={policy}, sptokens={sptokens}")

        try:
            drama = read_drama(filepath)
            mean_val = float(np.mean(drama))
        except Exception as e:
            print(f"  ERROR: {e}")
            mean_val = None

        # Store result
        results.setdefault(engine, {})
        results[engine].setdefault(policy, {})
        results[engine][policy][sptokens] = mean_val

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()