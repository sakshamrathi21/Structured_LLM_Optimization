#!/usr/bin/env bash
set -e

SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
# VLLM_SCRIPT="try.py" 
PY_BIN="python"                        


NUM_FAMILIES=(10)
REQ_PER_SEC=(500)
MAX_BATCH_SIZE=(500)
RANDOMNESS=(250 500)
LEN_FACTOR=(0.01 0.1 0.2 0.5 0.8 1.0 2.0 3.0 4.0 5.0 10.0)

for nf in "${NUM_FAMILIES[@]}"; do
  for rps in "${REQ_PER_SEC[@]}"; do
    for mbs in "${MAX_BATCH_SIZE[@]}"; do
      for rnd in "${RANDOMNESS[@]}"; do
        for lf in "${LEN_FACTOR[@]}"; do
          echo "-----------------------------------------------"
          echo "Running: num_families=$nf, req_per_sec=$rps, max_batch_size=$mbs, randomness=$rnd, len_factor=$lf"
          $PY_BIN "$VLLM_SCRIPT" "$nf" "$rps" "$mbs" "$rnd" "$lf"
          # $PY_BIN "$SGLANG_SCRIPT" "$nf" "$rps" "$mbs" "$rnd" "$lf"
          echo "-----------------------------------------------"
          echo
        done
      done
    done
  done  
done

