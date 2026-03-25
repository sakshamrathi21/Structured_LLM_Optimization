#!/usr/bin/env bash

VLLM_SCRIPT="/home/saksham/CS496/code/vllm/vllm_tree.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="rl_comparison_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(1000)

VLLM_POLICIES=(vllm:chunked_hash_tree_bandit vllm:chunked_hash_tree_heuristic vllm:chunked_hash_tree_qlearning)
MODES=(9 10 11)
for L in "${LEN_FACTOR[@]}"; do
    for m in "${MODES[@]}"; do
    
    NUM_REQUESTS=20000

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