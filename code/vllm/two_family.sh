#!/usr/bin/env bash

VLLM_SCRIPT="/home/saksham/CS496/code/vllm/two_family.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

LOG_DIR="two_families_logs"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(1000 2000)
REQ_PER_SEC_S=(1 2 3 4 5 7 10 15 20)
NUM_FAMILIES=(20 50 100)
MAX_NEW_TOKENS=(1 2 5 10 20 50 100 200 300 400 500)
MODELS=(qwen/qwen2.5-0.5b-instruct meta-llama/Llama-3.1-8B-Instruct lmsys/longchat-13b-16k)
MODELS=(lmsys/longchat-13b-16k)
FRACTIONS=(0 0.002 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.998 1)

VLLM_POLICIES=(chunked_hash_tree chunked_hash_tree_bandit)
SGLANG_POLICIES=()

MIN_REQUESTS=10
MAX_REQUESTS=5000

for f in "${FRACTIONS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/vllm_two_fam_frac${f}_tok200_${TIMESTAMP}.log"
    $PY_BIN "$VLLM_SCRIPT" --fraction "$f" --iterations 20 --write_dcgmi > "$LOG_FILE" 2>&1
done