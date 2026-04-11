#!/usr/bin/env bash
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"
LOG_DIR="longchat_families_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100)
NUM_FAMILIES=(50 20 10)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
MODELS=(qwen/qwen2.5-0.5b-instruct meta-llama/Llama-3.1-8B-Instruct lmsys/longchat-13b-16k)
VLLM_POLICIES=(family_load)
SGLANG_POLICIES=()
MIN_REQUESTS=5000
MAX_REQUESTS=10000

for fam in "${NUM_FAMILIES[@]}"; do

    # Base NUM_REQUESTS derived from family count (used for non-FCFS/dyn policies)
    BASE_REQUESTS=$((fam * 500))
    if [ $BASE_REQUESTS -lt $MIN_REQUESTS ]; then
        BASE_REQUESTS=$MIN_REQUESTS
    elif [ $BASE_REQUESTS -gt $MAX_REQUESTS ]; then
        BASE_REQUESTS=$MAX_REQUESTS
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    # Run VLLM policies
    for policy in "${VLLM_POLICIES[@]}"; do

        if [ "$policy" == "fcfs" ] || [ "$policy" == "dynamic_batching" ]; then
            NUM_REQUESTS=1000
        else
            NUM_REQUESTS=$BASE_REQUESTS
        fi

        source ~/.bashrc
        LOG_FILE="${LOG_DIR}/vllm_${policy}_req${NUM_REQUESTS}_tok200_fam${fam}_sys5000_${TIMESTAMP}.log"
        $PY_BIN "$VLLM_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 100 \
            --write_output \
            --num_requests "$NUM_REQUESTS" \
            --max_new_tokens 200 \
            --num_families "$fam" \
            --sys_prompt_tokens 5000 \
            --model_name "lmsys/longchat-13b-16k" \
            2>&1 | tee "$LOG_FILE"
    done

    # # Run SGLang policies
    # for policy in "${SGLANG_POLICIES[@]}"; do
    #     source ~/.bashrc
    #     LOG_FILE="${LOG_DIR}/sglang_${policy}_req${NUM_REQUESTS}_tok200_fam${fam}_sys5000_${TIMESTAMP}.log"
    #     $PY_BIN "$SGLANG_SCRIPT" \
    #         --policy "$policy" \
    #         --max_batch_size 100 \
    #         --write_output \
    #         --num_requests "$NUM_REQUESTS" \
    #         --max_new_tokens 200 \
    #         --num_families "$fam" \
    #         --sys_prompt_tokens 5000 \
    #         --model_name "lmsys/longchat-13b-16k" \
    #         > "$LOG_FILE" 2>&1
    # done

done