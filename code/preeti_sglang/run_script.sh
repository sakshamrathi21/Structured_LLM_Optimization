#!/bin/bash
set -e  # stop on error
set -u  # treat unset vars as error

# ==================== GLOBAL CONFIG ====================
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
MEM_FRACTION=0.9
ATTN_BACKEND="flashinfer"
ENABLE_METRICS="--enable-metrics"
DISABLE_CUDA_GRAPH="--disable-cuda-graph"
HOST="0.0.0.0"
SCHEDULE_POLICY="dynamic"
# "fcfs", "lpm", "random", "dfs-weight", "lof", "dynamic"

NUM_PROMPTS=2000
# REQUEST_RATE=50
RANDOM_OUTPUT_LEN=30
DATASET_NAME="custom_shared_prefix_dataset"
DATASET_PATH="/home/preeti/sglang/custom_shared_prefix_dataset.json"

BASE_DIR="/home/preeti/sglang"
LOG_DIR="$BASE_DIR/logs"
RESULT_BASE="$BASE_DIR/diff_scheduler_results_new"

mkdir -p "$RESULT_BASE"

# ==================== LOOP PARAMETERS ====================
NUM_FAMILIES_LIST=(1 2 4)
# NUM_FAMILIES_LIST=(20 100)
DATASET_TYPES=("order" "random")   # order → 0, random → 1

# ==================== FUNCTION TO GENERATE FACTORS ====================

generate_factors() {
  local num=$1
  local factors=()
  for ((i=10; i<=num; i+=10)); do
    if (( num % i == 0 )); then
      factors+=($i)
    fi
  done
  echo "${factors[@]}"
}

# MAX_NUM_REQUESTS_LIST=($(generate_factors $REQUESTS_PER_FAMILY"))

# ==================== LOOP EXECUTION ====================
for DATASET_TYPE in "${DATASET_TYPES[@]}"; do
  # Assign integer flag for benchmark argument
  if [ "$DATASET_TYPE" == "order" ]; then
    TYPE_INT=0
  else
    TYPE_INT=1
  fi

  for NUM_FAMILIES in "${NUM_FAMILIES_LIST[@]}"; do
    REQUESTS_PER_FAMILY=$((NUM_PROMPTS / NUM_FAMILIES))
    echo "------------------------------------------"
    echo "NUM_FAMILIES=$NUM_FAMILIES → Requests per family=$REQUESTS_PER_FAMILY"
    # MAX_NUM_REQUESTS_LIST=(10 20 50 100 200 300 400)
    MAX_NUM_REQUESTS_LIST=(50 100 200 500)
    echo "Generated MAX_NUM_REQUESTS_LIST: ${MAX_NUM_REQUESTS_LIST[*]}"
    echo "------------------------------------------"

    for MAX_NUM_REQUESTS in "${MAX_NUM_REQUESTS_LIST[@]}"; do
      echo "=========================================="
      echo " Running Benchmark:"
      echo "   DATASET_TYPE     = $DATASET_TYPE ($TYPE_INT)"
      echo "   NUM_FAMILIES     = $NUM_FAMILIES"
      echo "   MAX_NUM_REQUESTS = $MAX_NUM_REQUESTS"
      echo "   SCHEDULE_POLICY  = $SCHEDULE_POLICY"
      echo "=========================================="

      # ---- Step 1: Start SGLang Server ----
      echo "[1] Starting SGLang server..."
      python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --mem-fraction-static "$MEM_FRACTION" \
        $DISABLE_CUDA_GRAPH \
        --host "$HOST" \
        --attention-backend "$ATTN_BACKEND" \
        $ENABLE_METRICS \
        --max-running-requests "$MAX_NUM_REQUESTS" \
        --schedule-policy "$SCHEDULE_POLICY" \
        > server_output.log 2>&1 &
      SERVER_PID=$!
      # --schedule-policy "$SCHEDULE_POLICY" \

      echo "Server started with PID: $SERVER_PID"
      echo "Waiting for server to initialize..."
      sleep 60  # wait for server startup

      # ---- Step 2: Run Client for Prefix Sharing ----
      echo "[2] Running prefix sharing client (NUM_FAMILIES=$NUM_FAMILIES)..."
      python3 client_prefix.py "$NUM_FAMILIES"

      # ---- Step 3: Prepare Output Paths ----
      OUT_DIR="$RESULT_BASE/${DATASET_TYPE}_${SCHEDULE_POLICY}"
      mkdir -p "$OUT_DIR"

      BENCH_RESULT="$OUT_DIR/result_${DATASET_TYPE}_fam${NUM_FAMILIES}_req${NUM_PROMPTS}_bs${MAX_NUM_REQUESTS}_${SCHEDULE_POLICY}.csv"
      COPY_TO="$OUT_DIR/log_${DATASET_TYPE}_fam${NUM_FAMILIES}_req${NUM_PROMPTS}_bs${MAX_NUM_REQUESTS}_${SCHEDULE_POLICY}.csv"
      DCGMI_LOG="$OUT_DIR/dcgmi_${DATASET_TYPE}_fam${NUM_FAMILIES}_req${NUM_PROMPTS}_bs${MAX_NUM_REQUESTS}_${SCHEDULE_POLICY}.log"
      DRAM_LOG="$OUT_DIR/dram_${DATASET_TYPE}_fam${NUM_FAMILIES}_req${NUM_PROMPTS}_bs${MAX_NUM_REQUESTS}_${SCHEDULE_POLICY}.log"

      # ---- Step 4: Start DCGMI Monitoring ----
      echo "[3] Starting DCGMI monitoring..."
      dcgmi dmon -e 1001,1002,1003,1004,1005,1007,1008,1009,1010 -d 10 -c 0 > "$DCGMI_LOG" 2>&1 &
      DCGMI_PID=$!
      echo "DCGMI started with PID: $DCGMI_PID"

      # ---- Step 4b: Start DRAM Monitoring ----
      echo "[3b] Starting DRAM monitoring..."
      python3 /home/preeti/sglang/monitor_dram_in_background.py --interval 1.0 > "$DRAM_LOG" 2>&1 &
      DRAM_PID=$!
      echo "DRAM monitoring started with PID: $DRAM_PID"

      # ---- Step 5: Run Benchmark ----
      echo "[4] Running SGLang benchmark..."
      python3 -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 \
        --port 30000 \
        --num-prompts "$NUM_PROMPTS" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --model "$MODEL_PATH" \
        --dataset-name "$DATASET_NAME" \
        --dataset-path "$DATASET_PATH" \
        --num-families "$NUM_FAMILIES" \
        --type-of-dataset "$TYPE_INT" \
        --fam-size "$MAX_NUM_REQUESTS" \
        | tee "$BENCH_RESULT"
        # --request-rate "$REQUEST_RATE" \

      # ---- Step 6: Stop DCGMI Monitoring ----
      echo "[5] Stopping DCGMI monitoring..."
      kill -2 "$DCGMI_PID" 2>/dev/null || true
      sleep 2

      echo "[5b] Stopping DRAM monitoring..."
      kill -2 "$DRAM_PID" 2>/dev/null || true
      sleep 2

      echo "[✔] DCGMI log saved to: $DCGMI_LOG"
      echo "[✔] DRAM log saved to: $DRAM_LOG"

      # ---- Step 7: Copy Logs ----
      if [ -f "$LOG_DIR/log.csv" ]; then
        cp "$LOG_DIR/log.csv" "$COPY_TO"
        rm -f "$LOG_DIR/log.csv"
        echo "[6] Copied server log to: $COPY_TO"
      else
        echo "[6] Warning: No log.csv found in $LOG_DIR"
      fi

      INPUT=$DCGMI_LOG
      OUTPUT=$BENCH_RESULT
      
      awk '
        BEGIN {
            # Sums
            sum_gract = sum_smact = sum_smocc = 0
            sum_tenso = sum_drama = sum_fp32 = 0
            sum_fp16 = sum_pcitx = sum_pcirx = 0

            # Non-zero counts (per column)
            cnt_gract = cnt_smact = cnt_smocc = 0
            cnt_tenso = cnt_drama = cnt_fp32 = 0
            cnt_fp16 = cnt_pcitx = cnt_pcirx = 0
        }

        # Skip headers and empty lines
        /^#Entity/ || /^ID/ || NF == 0 {
            next
        }

        {
            # Expected format:
            # GPU 0 GRACT SMACT SMOCC TENSO DRAMA FP32A FP16A PCITX PCIRX
            if (NF < 11) next

            if ($3 != 0.0) { sum_gract += $3; cnt_gract++ }
            if ($4 != 0.0) { sum_smact += $4; cnt_smact++ }
            if ($5 != 0.0) { sum_smocc += $5; cnt_smocc++ }
            if ($6 != 0.0) { sum_tenso += $6; cnt_tenso++ }
            if ($7 != 0.0) { sum_drama += $7; cnt_drama++ }
            if ($8 != 0.0) { sum_fp32  += $8; cnt_fp32++ }
            if ($9 != 0.0) { sum_fp16  += $9; cnt_fp16++ }
            if ($10 != 0.0) { sum_pcitx += $10; cnt_pcitx++ }
            if ($11 != 0.0) { sum_pcirx += $11; cnt_pcirx++ }
        }

        END {
            print ""
            print "==== DCGMI Averages (Non-zero only) ===="

            if (cnt_gract > 0)
                printf "Avg GRACT: %.3f, Non-zero count: %d\n", sum_gract / cnt_gract, cnt_gract
            else
                print "GRACT: No non-zero samples"

            if (cnt_smact > 0)
                printf "Avg SMACT: %.3f, Non-zero count: %d\n", sum_smact / cnt_smact, cnt_smact
            else
                print "SMACT: No non-zero samples"

            if (cnt_smocc > 0)
                printf "Avg SMOCC: %.3f, Non-zero count: %d\n", sum_smocc / cnt_smocc, cnt_smocc
            else
                print "SMOCC: No non-zero samples"

            if (cnt_tenso > 0)
                printf "Avg TENSO: %.3f, Non-zero count: %d\n", sum_tenso / cnt_tenso, cnt_tenso
            else
                print "TENSO: No non-zero samples"

            if (cnt_drama > 0)
                printf "Avg DRAMA: %.3f, Non-zero count: %d\n", sum_drama / cnt_drama, cnt_drama
            else
                print "DRAMA: No non-zero samples"

            if (cnt_fp32 > 0)
                printf "Avg FP32A: %.3f, Non-zero count: %d\n", sum_fp32 / cnt_fp32, cnt_fp32
            else
                print "FP32A: No non-zero samples"

            if (cnt_fp16 > 0)
                printf "Avg FP16A: %.3f, Non-zero count: %d\n", sum_fp16 / cnt_fp16, cnt_fp16
            else
                print "FP16A: No non-zero samples"

            if (cnt_pcitx > 0)
                printf "Avg PCITX: %.3f, Non-zero count: %d\n", sum_pcitx / cnt_pcitx, cnt_pcitx
            else
                print "PCITX: No non-zero samples"

            if (cnt_pcirx > 0)
                printf "Avg PCIRX: %.3f, Non-zero count: %d\n", sum_pcirx / cnt_pcirx, cnt_pcirx
            else
                print "PCIRX: No non-zero samples"
        }
        ' "$INPUT" >> "$OUTPUT"

      echo "Averages and non-zero counts appended to $OUTPUT"

      # Extract Summary and append to OUTPUT
      echo "" >> "$OUTPUT"
      grep -A 8 "Summary:" "$DRAM_LOG" >> "$OUTPUT" 2>/dev/null || true
      
      # Extract just the Avg Throughput GB/s and print it clearly
      TOTAL_SAMPLES=$(awk '/Total Samples/ {print $4}' "$DRAM_LOG")
      NON_ZERO_SAMPLES=$(awk '/Non-Zero Samples/ {print $4}' "$DRAM_LOG")
      TOTAL_THROUGHPUT=$(awk '/Total Throughput/ {print $4}' "$DRAM_LOG")
      AVG_THROUGHPUT=$(awk '/Avg Throughput/ {print $4}' "$DRAM_LOG")
      AVG_NZ_THROUGHPUT=$(awk -v total="${TOTAL_THROUGHPUT:-0}" -v nz="${NON_ZERO_SAMPLES:-0}" 'BEGIN {if (nz > 0) print total / nz; else print 0}')
      
      if [ -n "$TOTAL_SAMPLES" ]; then
          # echo "DRAM_Total_Samples: $TOTAL_SAMPLES" >> "$OUTPUT"
          # echo "DRAM_Non_Zero_Samples: $NON_ZERO_SAMPLES" >> "$OUTPUT"
          # echo "DRAM_Total_Throughput_GB_per_s_sum: $TOTAL_THROUGHPUT" >> "$OUTPUT"
          echo "DRAM_Avg_Throughput_GBps: $AVG_THROUGHPUT" >> "$OUTPUT"
          echo "DRAM_Avg_NZ_Throughput_GBps: $AVG_NZ_THROUGHPUT" >> "$OUTPUT"
      fi

      # ---- Step 8: Cleanup Server ----
      echo "[7] Killing SGLang server and all subprocesses..."
      pkill -f "sglang::scheduler" || true
      pkill -f "sglang::http_server/tokenizer_manager" || true
      pkill -f "sglang::detokenizer" || true
      pkill -f "python3 -m sglang.launch_server" || true
      kill -9 "$SERVER_PID" 2>/dev/null || true
      sleep 5

      echo "[✔] Completed benchmark for DATASET_TYPE=$DATASET_TYPE, NUM_FAMILIES=$NUM_FAMILIES, MAX_NUM_REQUESTS=$MAX_NUM_REQUESTS"
      echo ""
    done
  done
done

echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="
