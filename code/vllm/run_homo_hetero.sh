#!/usr/bin/env bash
# run_all.sh — Sweep all modes x decode token counts, collect results into a CSV
# Usage: bash run_all.sh [optional extra args passed to bench.py]
# Example: bash run_all.sh --model meta-llama/Llama-3.1-8B-Instruct --L 10000

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODES=(1 2 3 4 5 6)
DECODE_TOKENS=(500 1000)
RESULTS_DIR="homo_hetero"
CSV="$RESULTS_DIR/summary.csv"
LOG_DIR="$RESULTS_DIR/logs"
PYTHON="${PYTHON:-python}"          # override with: PYTHON=python3 bash run_all.sh
SCRIPT="/home/saksham/CS496/code/vllm/homo_hetero_without_warmup"
EXTRA_ARGS="$@"                     # forward any extra CLI args (--model, --L, etc.)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

# Write CSV header
echo "mode,max_new_tokens,avg_time_s,avg_throughput_req_s" > "$CSV"

echo "============================================================"
echo " Starting sweep"
echo "  Modes        : ${MODES[*]}"
echo "  Decode tokens: ${DECODE_TOKENS[*]}"
echo "  Results CSV  : $CSV"
echo "  Extra args   : ${EXTRA_ARGS:-<none>}"
echo "============================================================"

total=$(( ${#MODES[@]} * ${#DECODE_TOKENS[@]} ))
run=0

for tok in "${DECODE_TOKENS[@]}"; do
    for mode in "${MODES[@]}"; do
        run=$(( run + 1 ))
        log="$LOG_DIR/mode${mode}_tok${tok}.log"

        echo ""
        echo "──────────────────────────────────────────────────────────"
        echo " Run $run / $total  |  mode=$mode  max_new_tokens=$tok"
        echo " Log → $log"
        echo "──────────────────────────────────────────────────────────"

        # Run the benchmark, tee output to log file
        $PYTHON "$SCRIPT" \
            --mode "$mode" \
            --max_new_tokens "$tok" \
            $EXTRA_ARGS \
            2>&1 | tee "$log"

        # Parse the result line from the RESULT block:
        # Format: <setting_name>   <avg_time>   <avg_throughput>
        # The line looks like: "1x800  (8 fam/batch)         12.345              65.43"
        result_line=$(grep -A4 "^RESULT" "$log" | grep "fam/batch" || true)

        if [[ -z "$result_line" ]]; then
            echo "[WARN] Could not parse result from $log — writing N/A"
            echo "${mode},${tok},N/A,N/A" >> "$CSV"
        else
            # Extract the last two numeric fields (avg_time, avg_throughput)
            avg_time=$(echo "$result_line" | awk '{print $(NF-1)}')
            avg_tp=$(echo "$result_line" | awk '{print $NF}')
            echo "${mode},${tok},${avg_time},${avg_tp}" >> "$CSV"
            echo "[OK] mode=$mode tok=$tok → time=${avg_time}s  tp=${avg_tp} req/s"
        fi

    done
done

echo ""
echo "============================================================"
echo " SWEEP COMPLETE"
echo " Results saved to: $CSV"
echo "============================================================"
echo ""
cat "$CSV"