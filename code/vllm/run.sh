# #!/usr/bin/env bash
VLLM_SCRIPT="vllm_multiple_families_varying_workloads.py"
PY_BIN="python"                        

LEN_FACTOR=(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
REQ_PER_SEC_S=(10 20 50)

# for lf in "${LEN_FACTOR[@]}"; do
for req_per_sec in "${REQ_PER_SEC_S[@]}"; do
    $PY_BIN "$VLLM_SCRIPT" --policy radix_cost --max_batch_size 1000 --write_output --multiplier 50 --max_new_tokens 100 --req_per_sec "$req_per_sec"
    $PY_BIN "$VLLM_SCRIPT" --policy family_load --max_batch_size 1000 --write_output --multiplier 50 --max_new_tokens 100 --req_per_sec "$req_per_sec"
    $PY_BIN "$VLLM_SCRIPT" --policy fcfs --max_batch_size 500 --write_output --multiplier 50 --max_new_tokens 100 --req_per_sec "$req_per_sec"
done