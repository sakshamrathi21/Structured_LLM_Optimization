#!/usr/bin/env bash

VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"                        

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 5 10 20 50 75 100)
NUM_FAMILIES=(5)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
MIN_REQUESTS=1000 
MAX_REQUESTS=5000
for tk in "${MAX_NEW_TOKENS[@]}"; do
    $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
    $PY_BIN "$VLLM_SCRIPT" --policy chunked_hash_tree_heuristic --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
    $PY_BIN "$VLLM_SCRIPT" --policy family_load --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
    $PY_BIN "$SGLANG_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
    $PY_BIN "$SGLANG_SCRIPT" --policy lpm --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
    $PY_BIN "$SGLANG_SCRIPT" --policy dfs-weight --max_batch_size 500 --write_output --num_requests 5000 --max_new_tokens "$tk" --num_families 5 --sys_prompt_tokens 5000
done