#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/tiered_prefix_experiment.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
B=(5 10 20 25 30 50 75 100 150 200 250 300 400 500 600 700 800 900)

for b in "${B[@]}"; do
    REQUESTS=$((5 * b))
    $PY_BIN "$VLLM_SCRIPT" --L 2000 --f1 0 --f2 0.2 --num_families 5 --write_dcgmi --iterations 1 --batch_size "$b" --requests "$REQUESTS"
done

# for f in batch_size_dcgmi_logs/*.log; do
#     echo "$f"
#     $PY_BIN read_drama.py "$f"
# done
