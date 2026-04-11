#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="no_pref_sharing_logs"
mkdir -p "$LOG_DIR"

REQ_PER_SEC_S=(15 20)

VLLM_POLICIES=(chunked_hash_tree_heuristic)
SGLANG_POLICIES=()

MIN_REQUESTS=10
MAX_REQUESTS=5000

for req in "${REQ_PER_SEC_S[@]}"; do
    NUM_REQUESTS=500

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Run VLLM policies
    for policy in "${VLLM_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/vllm_${policy}_rps${req}_req${NUM_REQUESTS}_tok200_fam5_sys1000_${TIMESTAMP}.log"

        $PY_BIN "$VLLM_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --num_families "$NUM_REQUESTS" \
            --sys_prompt_tokens 1000 \
            --req_per_sec "$req" \
            2>&1 | tee "$LOG_FILE"
    done

    # Run SGLang policies
    for policy in "${SGLANG_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/sglang_${policy}_rps${req}_req${NUM_REQUESTS}_tok200_fam5_sys1000_${TIMESTAMP}.log"

        $PY_BIN "$SGLANG_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --num_families "$NUM_REQUESTS" \
            --sys_prompt_tokens 1000 \
            --req_per_sec "$req" \
            2>&1 | tee "$LOG_FILE"
    done
done