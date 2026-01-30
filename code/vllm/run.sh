# #!/usr/bin/env bash
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="/home/saksham/CS496/code/vllm/.venv/bin/python"                        

LEN_FACTOR=(1.0 3.0 5.0 10.0)
REQ_PER_SEC_S=(20 50 100 200 500)
NUM_FAMILIES=(5)

# for lf in "${LEN_FACTOR[@]}"; do
# # for fam in "${NUM_FAMILIES[@]}"; do
#     for req_per_sec in "${REQ_PER_SEC_S[@]}"; do
#         # $PY_BIN "$VLLM_SCRIPT" --policy radix_cost --max_batch_size 1000 --write_output --multiplier 20 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#         # $PY_BIN "$VLLM_SCRIPT" --policy family_load --max_batch_size 1000 --write_output --multiplier 20 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#         # $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --multiplier 20 --max_new_tokens 100 --req_per_sec "$req_per_sec"  --num_families 5 --len_factor "$lf"
#         # $PY_BIN "$VLLM_SCRIPT" --policy dynamic_batching --max_batch_size 500 --write_output --multiplier 20 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#         $PY_BIN "$SGLANG_SCRIPT" --policy dfs-weight --max_batch_size 500 --write_output --multiplier 10 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#         $PY_BIN "$SGLANG_SCRIPT" --policy lpm --max_batch_size 500 --write_output --multiplier 10 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#         $PY_BIN "$SGLANG_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --multiplier 10 --max_new_tokens 100 --req_per_sec "$req_per_sec" --num_families 5 --len_factor "$lf"
#     done
# done

# nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --kernel-name regex:flash --csv ./vllm_multiple_families_varying_workloads.py --multiplier 5 --policy family_load > .out_fam

# nv-nsight-cu-cli --target-processes all --section MemoryWorkloadAnalysis --kernel-name regex:flash --csv ./vllm_multiple_families_varying_workloads.py --multiplier 5 > .out_fcfs


# python vllm_multiple_families_varying_workloads.py --policy family_load --len_factor 10 --write_dcgmi --max_batch_size 200 --multiplier 20 --num_families 5 --req_per_sec 100
# python vllm_multiple_families_varying_workloads.py --len_factor 10 --write_dcgmi --max_batch_size 500 --multiplier 20 --num_families 5 --req_per_sec 100

$PY_BIN "$VLLM_SCRIPT" --policy radix_cost --max_batch_size 1000 --write_output --multiplier 4 --max_new_tokens 100 --req_per_sec 500 --num_families 5 --len_factor 5 > .out
# $PY_BIN "$VLLM_SCRIPT" --policy family_load --max_batch_size 1000 --write_output --multiplier 4 --max_new_tokens 100 --req_per_sec 500 --num_families 5 --len_factor 5 > .out2
# $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --multiplier 4 --max_new_tokens 100 --req_per_sec 500 --num_families 5 --len_factor 5  --model_name qwen/qwen2.5-0.5b-instruct  >> .out

# $PY_BIN "$VLLM_SCRIPT" --policy radix_cost --max_batch_size 1000 --write_output --multiplier 4 --max_new_tokens 100 --req_per_sec 500 --num_families 5 --len_factor 5 --model_name lmsys/longchat-13b-16k >> .out
# $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --multiplier 4 --max_new_tokens 100 --req_per_sec 500 --num_families 5 --len_factor 5  --model_name lmsys/longchat-13b-16k  >> .out




