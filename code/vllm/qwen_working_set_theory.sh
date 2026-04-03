#!/usr/bin/env bash
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"
LOG_DIR="qwen_working_set_theory_logs"
mkdir -p "$LOG_DIR"
LEN_FACTOR=(100 200 500 1000 2000 5000 10000)
NUM_FAMILIES=(1 2 5 10 20 50 100)
for len_factor in "${LEN_FACTOR[@]}"; do
    for num_families in "${NUM_FAMILIES[@]}"; do
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="${LOG_DIR}/vllm_fcfs_qwen2.5-0.5b-instruct_req5000_tok${len_factor}_fam${num_families}_sys${len_factor}_${TIMESTAMP}.log"
        echo "Running with num_families=${num_families}, sys_prompt_tokens=${len_factor}..."
        $PY_BIN "$VLLM_SCRIPT" \
        --policy "fcfs" \
        --max_batch_size 500 \
        --write_output \
        --num_requests 5000 \
        --max_new_tokens 200 \
        --num_families "$num_families" \
        --sys_prompt_tokens "$len_factor" \
        --model_name "qwen/qwen2.5-0.5b-instruct" \
        > "$LOG_FILE" 2>&1
    done
done