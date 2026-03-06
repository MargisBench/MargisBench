#!/bin/bash

# --- Configuration ---
DEFAULT_MODEL="efficientnet_distilled_full_integer_quant_edgetpu.tflite"
DEFAULT_RESIZE_DIM=256
BENCHMARK_BIN="./benchmark_model_arm8_final"
LIBEDGETPU="libedgetpu.so.1"
CSV_FILE="benchmark_results.csv"
TEMP_ACC_LOG="acc_output.log"
PROFILE_LOG="op_profiling_breakdown.txt"

# --- Arguments ---
MODEL=${1:-$DEFAULT_MODEL}

RESIZE_DIM=${2:-$DEFAULT_RESIZE_DIM}

shift 2
CLASSES_ARGS="$@"

if [ -z "$CLASSES_ARGS" ]; then
    echo "Error: No classes provided."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' not found."
    exit 1
fi


# Fixed: Added $ to CSV_FILE in the rm command
if [ -f "$CSV_FILE" ]; then
    rm "$CSV_FILE"
fi

#if [ -f "$PROFILE_LOG" ]; then 
    #rm "$PROFILE_LOG"; 
#fi

# We initialize the header immediately
echo "Accuracy(%),InferenceAvg(ms),InitTime(ms),PeakMemory(MB)" > "$CSV_FILE"


# Run python and save output to log while showing it on screen
python3 accuracy.py --model "$MODEL" --resize-dim "$RESIZE_DIM" --classes "$CLASSES_ARGS"| tee "$TEMP_ACC_LOG"

# Parse the Accuracy
ACCURACY_RAW=$(grep "ACCURACY_RESULT" "$TEMP_ACC_LOG" | awk -F': ' '{print $2}' | tr -d '%')

if [ -z "$ACCURACY_RAW" ]; then
    ACCURACY_RAW="0.0"
fi


# RUN TRACING (Native Benchmark)
echo ""


echo "[INFO] CLEARING SYSTEM CACHE FOR CORAL..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
echo "[INFO] CORAL SYSTEM CACHE CLEANED!"

# Run Benchmark & Capture Output
OUTPUT=$($BENCHMARK_BIN \
    --graph="$MODEL" \
    --external_delegate_path="$LIBEDGETPU" \
    --num_runs=665 \
    --warmup_runs=10 \
    --enable_op_profiling=true 2>&1)

# --- PARSING LOGIC ---

SUMMARY_LINE=$(echo "$OUTPUT" | grep "Inference timings in us")
INIT_RAW=$(echo "$SUMMARY_LINE" | awk '{print $6}' | tr -d ',')
AVG_RAW=$(echo "$SUMMARY_LINE" | awk '{print $NF}')
MEM_OVERALL=$(echo "$OUTPUT" | grep "Peak memory footprint" | grep -o "overall=[0-9.]*" | cut -d= -f2)

# Math: Convert us -> ms
INIT_MS=$(awk -v val="$INIT_RAW" 'BEGIN {printf "%.2f", val / 1000}')
AVG_MS=$(awk -v val="$AVG_RAW" 'BEGIN {printf "%.2f", val / 1000}')

# Extract operator profiling 
echo "$OUTPUT" > "$PROFILE_LOG"
TPU_NODE_TIME=$(echo "$OUTPUT" | grep -E "TfLiteGpuDelegateV2|edgetpu|Custom" | head -n 1)

echo "======================== CORAL STATS ======================"
echo "    Accuracy:       ${ACCURACY_RAW} %"
echo "    Inference Avg:  ${AVG_MS} ms"
echo "    Init Time:      ${INIT_MS} ms"
echo "    Peak Memory:    ${MEM_OVERALL} MB"
echo "==========================================================="

# 4. WRITE TO CSV
# Append the data line to the file created in step 1
echo "$ACCURACY_RAW,$AVG_MS,$INIT_MS,$MEM_OVERALL" >> "$CSV_FILE"

# Cleanup
rm "$TEMP_ACC_LOG"

echo ""
