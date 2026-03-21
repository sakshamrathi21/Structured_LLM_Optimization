#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="families_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100)
NUM_FAMILIES=(1 2 5 10 20 50 100)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
MODELS=(qwen/qwen2.5-0.5b-instruct meta-llama/Llama-3.1-8B-Instruct lmsys/longchat-13b-16k)

# VLLM_POLICIES=(fcfs chunked_hash_tree_heuristic chunked_hash_tree_bandit)
VLLM_POLICIES=(chunked_hash_tree_bandit)
# SGLANG_POLICIES=(fcfs lpm dfs-weight)
# SGLANG_POLICIES=(fcfs)

MIN_REQUESTS=10000
MAX_REQUESTS=20000

for fam in "${NUM_FAMILIES[@]}"; do
    NUM_REQUESTS=$((fam * 500))

    if [ $NUM_REQUESTS -lt $MIN_REQUESTS ]; then
        NUM_REQUESTS=$MIN_REQUESTS
    elif [ $NUM_REQUESTS -gt $MAX_REQUESTS ]; then
        NUM_REQUESTS=$MAX_REQUESTS
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Run VLLM policies
    for policy in "${VLLM_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/vllm_${policy}_req${NUM_REQUESTS}_tok200_fam${fam}_sys5000_${TIMESTAMP}.log"

        $PY_BIN "$VLLM_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --num_families "$fam" \
            --sys_prompt_tokens 5000 \
            > "$LOG_FILE" 2>&1
    done

    # # Run SGLang policies
    for policy in "${SGLANG_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/sglang_${policy}_req${NUM_REQUESTS}_tok200_fam${fam}_sys5000_${TIMESTAMP}.log"

        $PY_BIN "$SGLANG_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --num_families "$fam" \
            --sys_prompt_tokens 5000 \
            > "$LOG_FILE" 2>&1
    done
done