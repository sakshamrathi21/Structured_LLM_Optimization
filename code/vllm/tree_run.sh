#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/vllm_tree.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
B=(100 200 500 1000 2000 5000)

for l in "${B[@]}"; do
    $PY_BIN "$VLLM_SCRIPT" --mode 1 --L "$l" --total_requests 2000 --write_output 
    $PY_BIN "$VLLM_SCRIPT" --mode 2 --L "$l" --total_requests 2000 --write_output 
    $PY_BIN "$VLLM_SCRIPT" --mode 9 --L "$l" --total_requests 2000 --write_output 
    $PY_BIN "$VLLM_SCRIPT" --mode 10 --L "$l" --total_requests 2000 --write_output 
    $PY_BIN "$VLLM_SCRIPT" --mode 11 --L "$l" --total_requests 2000 --write_output 
done

# for f in batch_size_dcgmi_logs/*.log; do
#     echo "$f"
#     $PY_BIN read_drama.py "$f"
# done
