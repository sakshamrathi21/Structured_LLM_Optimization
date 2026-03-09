#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100)
NUM_FAMILIES=(5)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
# MODELS=(qwen/qwen2.5-0.5b-instruct meta-llama/Llama-3.1-8B-Instruct lmsys/longchat-13b-16k)
MODELS=(lmsys/longchat-13b-16k)

VLLM_POLICIES=(fcfs chunked_hash_tree_heuristic family_load)
SGLANG_POLICIES=(fcfs lpm dfs-weight)

MIN_REQUESTS=1000
MAX_REQUESTS=5000

for mode in "${MODELS[@]}"; do

    # Run VLLM policies
    for policy in "${VLLM_POLICIES[@]}"; do
        $PY_BIN "$VLLM_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests 2500 \
            --max_new_tokens 200 \
            --num_families 5 \
            --sys_prompt_tokens 5000 \
            --model_name "$mode"
    done

    # Run SGLang policies
    for policy in "${SGLANG_POLICIES[@]}"; do
        $PY_BIN "$SGLANG_SCRIPT" \
            --policy "$policy" \
            --max_batch_size 500 \
            --write_output \
            --num_requests 2500 \
            --max_new_tokens 200 \
            --num_families 5 \
            --sys_prompt_tokens 5000 \
            --model_name "$mode"
    done

done