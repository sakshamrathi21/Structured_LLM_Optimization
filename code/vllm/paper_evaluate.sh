#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"                        

LEN_FACTOR=(10000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100 200 300 400 500)
NUM_FAMILIES=(50)
MIN_REQUESTS=1000 
for lf in "${LEN_FACTOR[@]}"; do
    for fam in "${NUM_FAMILIES[@]}"; do
        for req_per_sec in "${REQ_PER_SEC_S[@]}"; do
            NUM_REQUESTS=$((req_per_sec * 100))
            $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
            $PY_BIN "$VLLM_SCRIPT" --policy chunked_hash_tree --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
            $PY_BIN "$VLLM_SCRIPT" --policy family_load --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
            $PY_BIN "$SGLANG_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
            $PY_BIN "$SGLANG_SCRIPT" --policy dfs-weight --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
            $PY_BIN "$SGLANG_SCRIPT" --policy lpm --max_batch_size 500 --write_output --num_requests "$NUM_REQUESTS" --max_new_tokens 200 --req_per_sec "$req_per_sec" --num_families "$fam" --sys_prompt_tokens "$lf"
        done
    done
done