#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/two_family.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
F=(0 0.002 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99 0.998 1.0)

for f in "${F[@]}"; do
    $PY_BIN "$VLLM_SCRIPT" --L 10000 --fraction "$f" --write_dcgmi --iterations 10 --requests 500 --model Qwen/Qwen2.5-14B
done

# for f in *.log; do
#     echo "$f"
#     $PY_BIN read_drama.py "$f"
# done
