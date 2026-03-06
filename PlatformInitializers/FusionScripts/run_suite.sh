#!/bin/bash

# This script allows to run the inference of a specific model over all the images present in DatasetFiles/${IMAGE_SIZE}
# on Fusion Target. The output will be showd in the terminal and stored in a csv.

if [ "$#" -ne 4 ]; then
    echo "Incorrect usage."
    echo "Usage: ./run_suite <model-name> <model-name-path> <batch-size> <image-size>"
    exit 1
fi


MODEL_NAME=$1
MODEL_NAME_PATH=$2
BATCH_SIZE=$3
IMAGE_SIZE=$4
CSV_FILE="benchmark_results.csv"

if [ -z "$BATCH_SIZE" ]; then
    echo "Error: No batch size provided."
    exit 1
fi
if [ -z "$MODEL_NAME" ]; then
    echo "Error: No model name provided."
    exit 1
fi

if [ ! -f "$MODEL_NAME_PATH" ]; then
    echo "Error: Model file '$MODEL_NAME_PATH' not found."
    exit 1
fi

echo ""
echo "=========================================="
echo "    FULL EVALUATION SUITE"
echo "    Model: $MODEL_NAME_PATH"
echo "=========================================="
echo ""
# 1. INITIALIZE CSV
# Fixed: Added $ to CSV_FILE in the rm command
if [ -f "$CSV_FILE" ]; then
    rm "$CSV_FILE"
fi

echo "Accuracy(%),InferenceAvg(ms),PeakMemory(MB),PeakMemoryAccelerator(MB),WeightsMemory(MB),AvgOverallLatencyPerImage(ms),AvgOverallLatencyPerBatch(ms),AvgHwLatencyPerImage(ms),AvgHwLatencyPerBatch(ms)" > "$CSV_FILE"

#DROP CACHE AND INFER
echo "[INFO] CLEARING SYSTEM CACHE FOR FUSION..."
sync
echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
echo "[INFO] FUSION SYSTEM CACHE CLEANED!"
./inference $MODEL_NAME_PATH DatasetFiles/${IMAGE_SIZE}/ memory_profiling_files/${MODEL_NAME}.csv $BATCH_SIZE | tee output.log


ACCURACY=$(cat output.log | grep "FINAL_ACCURACY" | cut -d ":" -f 2 | cut -d " " -f 2 | cut -d "%" -f 1)
AVG_OVERALL_LATENCY_PER_IMAGE=$(cat output.log | grep "AVG_OVERALL_LATENCY_PER_IMAGE" | cut -d ":" -f 2 | cut -d " " -f 2)
AVG_OVERALL_LATENCY_PER_BATCH=$(cat output.log | grep "AVG_OVERALL_LATENCY_PER_BATCH" | cut -d ":" -f 2 | cut -d " " -f 2)
AVG_HW_LATENCY_PER_IMAGE=$(cat output.log | grep "AVG_HW_LATENCY_PER_IMAGE" | cut -d ":" -f 2 | cut -d " " -f 2)
AVG_HW_LATENCY_PER_BATCH=$(cat output.log | grep "AVG_HW_LATENCY_PER_BATCH" | cut -d ":" -f 2 | cut -d " " -f 2)
OVERALL_LATENCY=$(cat output.log | grep "TOTAL_OVERALL_LATENCY" | cut -d ":" -f 2 | cut -d " " -f 2)
MEMORY_PEAK_HOST=$(cat output.log | grep "MEMORY_PEAK_HOST" | cut -d ":" -f 2 | cut -d " " -f 2)
MEMORY_PEAK_ACCELERATOR=$(cat output.log | grep "MEMORY_PEAK_ACCELERATOR" | cut -d ":" -f 2 | cut -d " " -f 2)
MEMORY_WEIGHTS=$(du -b $MODEL_NAME_PATH | awk '{print $1}')
echo "memory weights: $MEMORY_WEIGHTS"

echo "$ACCURACY,$OVERALL_LATENCY,$MEMORY_PEAK_HOST,$MEMORY_PEAK_ACCELERATOR,$MEMORY_WEIGHTS,$AVG_OVERALL_LATENCY_PER_IMAGE,$AVG_OVERALL_LATENCY_PER_BATCH,$AVG_HW_LATENCY_PER_IMAGE,$AVG_HW_LATENCY_PER_BATCH" >> "$CSV_FILE"
echo ""
echo "[INFO] Suite Complete. Data saved to $CSV_FILE"
rm *.log
