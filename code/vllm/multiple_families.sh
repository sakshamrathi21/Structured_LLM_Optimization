#!/bin/bash

# Experiment: vary number of families with equal distribution
# Equal distribution means f2 = 1/num_families (family 1 gets same share as others)
# Common prefix f1 = 0 (fixed)

MODEL="meta-llama/Llama-3.1-8B-Instruct"
L=10000
F1=0
ITERATIONS=1
MAX_NEW_TOKENS=50
BATCH_SIZE=500
SEED=42

FAMILIES=(5 10 20 50 100)

LOG_DIR="./logs_families_experiment"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Tiered Prefix Sharing - Family Sweep"
echo "========================================"
echo "Model       : $MODEL"
echo "L           : $L tokens"
echo "f1          : $F1"
echo "Iterations  : $ITERATIONS"
echo "Max tokens  : $MAX_NEW_TOKENS"
echo "Families    : ${FAMILIES[*]}"
echo "========================================"
echo ""

for NUM_FAM in "${FAMILIES[@]}"; do

    if [ "$NUM_FAM" -eq 1 ]; then
        ACTUAL_FAM=2
        F2="1.0"
    else
        ACTUAL_FAM=$NUM_FAM
        F2=$(python3 -c "print(f'{1.0/$NUM_FAM:.6f}')")
    fi

    REQUESTS=$(( 50 * NUM_FAM ))
    REQUESTS=$(( REQUESTS < 2000 ? 2000 : REQUESTS ))

    LOG_FILE_DEFAULT="$LOG_DIR/families_${NUM_FAM}_default.log"
    LOG_FILE_FAMILY_LOAD="$LOG_DIR/families_${NUM_FAM}_family_load.log"

    echo "----------------------------------------"
    echo "Running: num_families=$NUM_FAM  (actual=$ACTUAL_FAM)  f2=$F2"
    echo "Policy: default"
    echo "Log: $LOG_FILE_DEFAULT"
    echo "----------------------------------------"

    python3 tiered_prefix_experiment.py \
        --model "$MODEL" \
        --L "$L" \
        --f1 "$F1" \
        --f2 "$F2" \
        --num_families "$ACTUAL_FAM" \
        --requests "$REQUESTS" \
        --iterations "$ITERATIONS" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        2>&1 | tee "$LOG_FILE_DEFAULT"

    # echo ""
    # echo "----------------------------------------"
    # echo "Running: num_families=$NUM_FAM  (actual=$ACTUAL_FAM)  f2=$F2"
    # echo "Policy: family_load"
    # echo "Log: $LOG_FILE_FAMILY_LOAD"
    # echo "----------------------------------------"

    # python3 tiered_prefix_experiment.py \
    #     --model "$MODEL" \
    #     --L "$L" \
    #     --f1 "$F1" \
    #     --f2 "$F2" \
    #     --num_families "$ACTUAL_FAM" \
    #     --requests "$REQUESTS" \
    #     --iterations "$ITERATIONS" \
    #     --max_new_tokens "$MAX_NEW_TOKENS" \
    #     --batch_size "$BATCH_SIZE" \
    #     --seed "$SEED" \
    #     --policy radix_cost \
    #     2>&1 | tee "$LOG_FILE_FAMILY_LOAD"

    echo ""
    echo "Done with num_families=$NUM_FAM"
    echo ""
done

echo "========================================"
echo "All experiments complete."
echo "Logs saved to: $LOG_DIR"
echo "========================================"

# Summary: extract average throughput from each log
echo ""
echo "=== THROUGHPUT SUMMARY ==="
printf "%-15s %-10s %-15s %-20s\n" "num_families" "f2" "policy" "avg_throughput (req/s)"
printf "%-15s %-10s %-15s %-20s\n" "------------" "--" "------" "---------------------"

for NUM_FAM in "${FAMILIES[@]}"; do
    if [ "$NUM_FAM" -eq 1 ]; then
        F2_DISP="1.000000"
    else
        F2_DISP=$(python3 -c "print(f'{1.0/$NUM_FAM:.6f}')")
    fi

    LOG_FILE_DEFAULT="$LOG_DIR/families_${NUM_FAM}_default.log"
    LOG_FILE_FAMILY_LOAD="$LOG_DIR/families_${NUM_FAM}_family_load.log"

    if [ -f "$LOG_FILE_DEFAULT" ]; then
        THROUGHPUT=$(grep "Average throughput" "$LOG_FILE_DEFAULT" | awk '{print $NF}')
        printf "%-15s %-10s %-15s %-20s\n" "$NUM_FAM" "$F2_DISP" "default" "$THROUGHPUT"
    fi

    if [ -f "$LOG_FILE_FAMILY_LOAD" ]; then
        THROUGHPUT=$(grep "Average throughput" "$LOG_FILE_FAMILY_LOAD" | awk '{print $NF}')
        printf "%-15s %-10s %-15s %-20s\n" "$NUM_FAM" "$F2_DISP" "family_load" "$THROUGHPUT"
    fi
done