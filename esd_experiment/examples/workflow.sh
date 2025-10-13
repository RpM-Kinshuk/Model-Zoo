#!/bin/bash
# Example workflow for running large-scale ESD analysis

# This script demonstrates a complete workflow:
# 1. Create a model list
# 2. Run the experiment
# 3. Analyze results

set -e  # Exit on error

export MKL_THREADING_LAYER=GNU

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================="
echo "ESD Experiment Example Workflow"
echo "========================================="

# Configuration
OUTPUT_DIR="$PROJECT_ROOT/example_results"
MODEL_LIST="$SCRIPT_DIR/example_models.csv"
GPUS=(0 1 2 3 4 5 6 7)  # Adjust to your available GPUs (e.g., 0 1 2 3 for multiple GPUs)

# Step 1: Create model list
echo ""
echo "Step 1: Creating model list..."
echo "========================================="

# Option A: Create from scratch
python "$PROJECT_ROOT/create_model_list.py" create \
    --output "$MODEL_LIST" \
    --models \
        "meta-llama/Llama-2-7b-hf" \
        "microsoft/phi-2" \
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Option B: Convert from existing text file
# python create_model_list.py from-text existing_models.txt --output "$MODEL_LIST"

# Option C: Convert from existing CSV
# python create_model_list.py from-csv existing_data.csv \
#     --output "$MODEL_LIST" \
#     --model_col "full_name" \
#     --relation_col "base_model_relation" \
#     --source_col "source_model"

echo "Created model list: $MODEL_LIST"
cat "$MODEL_LIST"

# Step 2: Run experiment
echo ""
echo "Step 2: Running ESD analysis..."
echo "========================================="

python "$PROJECT_ROOT/run_experiment.py" \
    --model_list "$MODEL_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --gpus ${GPUS[@]} \
    --num_gpus_per_job 3 \
    --fix_fingers DKS \
    --filter_zeros \
    --gpu_memory_threshold 500 \
    --max_check 1

# To run with different ESD parameters:
# python run_esd_experiment.py \
#     --model_list "$MODEL_LIST" \
#     --output_dir "$OUTPUT_DIR" \
#     --gpus ${GPUS[@]} \
#     --fix_fingers xmin_peak \
#     --evals_thresh 1e-6 \
#     --bins 200 \
#     --parallel_esd

# Step 3: Analyze results
echo ""
echo "Step 3: Analyzing results..."
echo "========================================="

python "$PROJECT_ROOT/analyze_results.py" \
    --results_dir "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/summary.csv" \
    --verbose

# Step 4: Display summary
echo ""
echo "Step 4: Summary"
echo "========================================="

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Summary saved to: $OUTPUT_DIR/summary.csv"

# Count files
NUM_RESULTS=$(find "$OUTPUT_DIR" -name "*.csv" -not -name "summary.csv" -not -name "failed_models.txt" | wc -l)
echo "Number of analyzed models: $NUM_RESULTS"

if [ -f "$OUTPUT_DIR/failed_models.txt" ]; then
    NUM_FAILED=$(wc -l < "$OUTPUT_DIR/failed_models.txt")
    echo "Number of failed models: $NUM_FAILED"
    echo ""
    echo "Failed models:"
    cat "$OUTPUT_DIR/failed_models.txt"
fi

if [ $? -eq 130 ]; then
    echo "Script was interrupted"
    exit 130
fi

echo ""
echo "========================================="
echo "Workflow completed successfully!"
echo "========================================="

# Optional: Generate visualizations with Python
# python -c "
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# df = pd.read_csv('$OUTPUT_DIR/summary.csv')
# 
# # Plot alpha distribution
# plt.figure(figsize=(10, 6))
# plt.hist(df['alpha_mean'], bins=30, edgecolor='black')
# plt.xlabel('Mean Alpha')
# plt.ylabel('Count')
# plt.title('Distribution of Mean Alpha Across Models')
# plt.savefig('$OUTPUT_DIR/alpha_distribution.png')
# print('Saved plot to: $OUTPUT_DIR/alpha_distribution.png')
# "
