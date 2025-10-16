#!/bin/bash
set -e

trap 'echo "Caught signal, exiting..."; exit 130' INT TERM


export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

OUTPUT_DIR="$PROJECT_ROOT/atlas_results"
MODEL_LIST="$SCRIPT_DIR/examples/atlas_models.csv"
GPUS=(1 2 3)


echo "Running with max 3 GPUs total, 1 GPU per model"
python "$PROJECT_ROOT/run_experiment.py" \
    --model_list "$MODEL_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --gpus ${GPUS[@]} \
    --num_gpus_per_job 1 \
    --fix_fingers DKS \
    --filter_zeros \
    --gpu_memory_threshold 500 \
    --max_check 1


if [ $? -eq 130 ]; then
    echo "Script was interrupted"
    exit 130
fi