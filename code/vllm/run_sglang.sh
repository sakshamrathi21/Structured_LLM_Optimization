# #!/usr/bin/env bash
SGLANG_SCRIPT="sglang_multiple_families_varying_workloads.py"
PY_BIN="python"                        

LEN_FACTOR=(1.0 2.0 5.0 10.0)
REQ_PER_SEC_S=(10 20 50)

for lf in "${LEN_FACTOR[@]}"; do
# for req_per_sec in "${REQ_PER_SEC_S[@]}"; do
    # $PY_BIN "$SGLANG_SCRIPT" --policy lpm --max_batch_size 500 --write_output --multiplier 50 --max_new_tokens 100 --len_factor "$lf"
    $PY_BIN "$SGLANG_SCRIPT" --policy dfs-weight --max_batch_size 500 --write_output --multiplier 25 --max_new_tokens 100 --len_factor "$lf"
done