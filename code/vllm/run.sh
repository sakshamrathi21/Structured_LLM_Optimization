#!/usr/bin/env bash
set -e

SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
# VLLM_SCRIPT="try.py" 
PY_BIN="python"                        


NUM_FAMILIES=(10)
REQ_PER_SEC=(500)
MAX_BATCH_SIZE=(500)
RANDOMNESS=(0 1 5 10 25 50 100 200 250 400 499 500 1000)

for nf in "${NUM_FAMILIES[@]}"; do
  for rps in "${REQ_PER_SEC[@]}"; do
    for mbs in "${MAX_BATCH_SIZE[@]}"; do
      for rnd in "${RANDOMNESS[@]}"; do
        echo "-----------------------------------------------"
        echo "Running: num_families=$nf, req_per_sec=$rps, max_batch_size=$mbs, randomness=$rnd"
        $PY_BIN "$VLLM_SCRIPT" "$nf" "$rps" "$mbs" "$rnd"
        # $PY_BIN "$SGLANG_SCRIPT" "$nf" "$rps" "$mbs" "$rnd"
        echo "-----------------------------------------------"
        echo
      done
    done
  done
done

