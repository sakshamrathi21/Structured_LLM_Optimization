#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="decode_tokens_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100)
NUM_FAMILIES=(50 100)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
MODELS=(qwen/qwen2.5-0.5b-instruct meta-llama/Llama-3.1-8B-Instruct lmsys/longchat-13b-16k)

VLLM_POLICIES=(dynamic_batching)
# SGLANG_POLICIES=(fcfs lpm dfs-weight)
SGLANG_POLICIES=()

MIN_REQUESTS=1000
MAX_REQUESTS=10000

for tok in "${MAX_NEW_TOKENS[@]}"; do

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Run VLLM policies
    for policy in "${VLLM_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/vllm_${policy}_req5000_tok${tok}_fam5_sys5000_${TIMESTAMP}.log"

        $PY_BIN "$VLLM_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests 5000 \
            --max_new_tokens "$tok" \
            --num_families 5 \
            --sys_prompt_tokens 5000 \
            > "$LOG_FILE" 2>&1
    done

    # Run SGLang policies
    for policy in "${SGLANG_POLICIES[@]}"; do
        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/sglang_${policy}_req5000_tok${tok}_fam5_sys5000_${TIMESTAMP}.log"

        $PY_BIN "$SGLANG_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests 5000 \
            --max_new_tokens "$tok" \
            --num_families 5 \
            --sys_prompt_tokens 5000 \
            > "$LOG_FILE" 2>&1
    done
done