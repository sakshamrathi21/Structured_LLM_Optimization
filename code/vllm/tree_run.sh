#!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/fraction_sharing_expt.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
F=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for f in "${F[@]}"; do
    $PY_BIN "$VLLM_SCRIPT" --L 2000 --fraction "$f" --write_dcgmi --iterations 10 --batch_size 100 --requests 100
done

# for f in two_family_dcgmi_longchat_logs/*.log; do
#     echo "$f"
#     $PY_BIN read_drama.py "$f"
# done
