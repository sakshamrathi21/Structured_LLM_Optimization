#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/tiered_prefix_experiment.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"
LOG_DIR="tiered_prefix_sharing_logs"
mkdir -p "$LOG_DIR"
F1=(0 0.2 0.4 0.6 0.8 1)
F2=(0 0.2 0.4 0.6 0.8 1)
for f1 in "${F1[@]}"; do
    for f2 in "${F2[@]}"; do
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="${LOG_DIR}/vllm_tiered_prefix_sharing_req5000_tok10000_f1_${f1}_f2_${f2}_${TIMESTAMP}.log"
        echo "Running with f1=${f1}, f2=${f2}..."
        $PY_BIN "$VLLM_SCRIPT" \
        --requests 1000 \
        --max_new_tokens 200 \
        --num_families 51 \
        --L 10000 \
        --f1 "$f1" \
        --f2 "$f2" \
        > "$LOG_FILE" 2>&1
    done
done