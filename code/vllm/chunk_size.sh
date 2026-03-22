#!/usr/bin/env bash
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"
LOG_DIR="chunk_size_logs"
CHUNK_SCRIPT="/home/saksham/CS496/code/vllm/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/request_queue.py"
mkdir -p "$LOG_DIR"

LEN_FACTOR=(100 1000 2000 5000 10000)
CHUNK_SIZES=(1 2 5 10 20 50 100 200 500 1000)
MIN_REQUESTS=1000
MAX_REQUESTS=10000

# Line number in the Python script where self.chunk_size = <N> lives
CHUNK_LINE=564   # <-- UPDATE THIS to the actual line number

for ch in "${CHUNK_SIZES[@]}"; do

    # --- Patch chunk_size in the Python script ---
    sed -i "${CHUNK_LINE}s/.*/        self.chunk_size = ${ch}/" "$CHUNK_SCRIPT"

    for len in "${LEN_FACTOR[@]}"; do
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        LOG_FILE="${LOG_DIR}/sys${len}_chunk${ch}_${TIMESTAMP}.log"

        echo "[$(date)] Running: chunk_size=${ch}, sys_prompt_tokens=${len} -> ${LOG_FILE}"

        $PY_BIN "$VLLM_SCRIPT" \
            --policy chunked_hash_tree_heuristic --write_dcgmi \
            --max_batch_size 500 \
            --write_output \
            --num_requests 5000 \
            --max_new_tokens 200 \
            --num_families 5 \
            --sys_prompt_tokens "$len" \
            > "$LOG_FILE" 2>&1
    done

    # --- Restore chunk_size to 1 after all len runs for this chunk ---
    sed -i "${CHUNK_LINE}s/.*/        self.chunk_size = 1/" "$CHUNK_SCRIPT"

done