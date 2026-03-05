#!/usr/bin/env bash

VLLM_SCRIPT="/home/saksham/CS496/code/vllm/vllm_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"

B=(2 5 10 20 50 75 100 125 150 175 200 250 300 400 500 600 700 800 900)

MAX_REQUESTS=20000

# for l in "${B[@]}"; do
#     total_requests=$((100 * l))

#     if [ "$total_requests" -gt "$MAX_REQUESTS" ]; then
#         total_requests=$MAX_REQUESTS
#     fi

#     multiplier=$((total_requests / 100))

#     $PY_BIN "$VLLM_SCRIPT" \
#         --num_families 1 \
#         --max_batch_size "$l" \
#         --len_factor 0 \
#         --multiplier "$multiplier" \
#         --write_output \
#         --write_dcgmi
# done

for f in batch_size_dcgmi_logs/small_requests/*.log; do
    echo "$f"
    $PY_BIN read_drama.py "$f"
done
