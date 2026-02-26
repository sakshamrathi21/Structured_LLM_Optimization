#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/fraction_sharing_expt.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
B=(5 10 20 30 50 75 100 150 200 250 300 500 1000)

# for b in "${B[@]}"; do
#     REQUESTS=$((50 * b))
#     $PY_BIN "$VLLM_SCRIPT" --L 2000 --fraction 1 --write_dcgmi --iterations 10 --batch_size "$b" --requests "$REQUESTS"
# done

for f in batch_size_dcgmi_logs/*.log; do
    echo "$f"
    $PY_BIN read_drama.py "$f"
done
