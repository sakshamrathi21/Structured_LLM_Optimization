# #!/usr/bin/env bash
VLLM_SCRIPT="/home/saksham/CS496/code/vllm/tiered_prefix_experiment.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  

F1=(0 0.2 0.4 0.6 0.8 1.0)
F2=(0 0.2 0.4 0.6 0.8 1.0)
LS=(2000 10000)

for fam in "${LS[@]}"; do
    for f1 in "${F1[@]}"; do
        for f2 in "${F2[@]}"; do
            $PY_BIN "$VLLM_SCRIPT" --f1 "$f1" --f2 "$f2" --L "$fam" --num_families 51
        done
    done
done


# # #!/usr/bin/env bash
# VLLM_SCRIPT="/home/saksham/CS496/code/vllm/fraction_sharing_expt.py"
# PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"  
# # --fraction 0 --requests 100 --policy radix_cost
# F=(0 0.2 0.4 0.6 0.8 1.0)
# POLICIES=(fcfs radix_cost chunked_hash_tree_heuristic chunked_hash_tree_bandit chunked_hash_tree_qlearning)

# for f in "${F[@]}"; do
#     for p in "${POLICIES[@]}"; do
#         $PY_BIN "$VLLM_SCRIPT" --fraction "$f" --policy "$p" --requests 100
#     done
# done
