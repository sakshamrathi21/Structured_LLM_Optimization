#!/usr/bin/env bash

VLLM_SCRIPT="/home/saksham/CS496/code/vllm/vllm_tree.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="radix_tree_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(5000)


VLLM_POLICIES=(sglang:lpm sglang:dfs-weight sglang:fcfs vllm:fcfs vllm:chunked_hash_tree_heuristic vllm:dynamic_batching)
# VLLM_POLICIES=(vllm:fcfs vllm:chunked_hash_tree_heuristic vllm:dynamic_batching)
MODES=(1 5 9 10 11)
for L in "${LEN_FACTOR[@]}"; do
    for m in "${MODES[@]}"; do
    
    NUM_REQUESTS=8000

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    for policy in "${VLLM_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/${policy}_req8000_tok200_sys${L}_mode${m}_${TIMESTAMP}.log"

        $PY_BIN "$VLLM_SCRIPT" \
            --L "$L" \
            --total_requests "$NUM_REQUESTS" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --total_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --mode "$m" \
            > "$LOG_FILE" 2>&1
    done
    done
done