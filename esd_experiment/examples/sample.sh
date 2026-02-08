#!/bin/bash
# Example script for running ESD analysis with GPU scheduling

# Example 1: Use max 4 GPUs from a pool of 8, with 1 GPU per model
echo "Example 1: Running with max 4 GPUs total, 1 GPU per model"
python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results_scheduled \
    --use-gpu-scheduling \
    --gpu-pool 0 1 2 3 4 5 6 7 \
    --max-total-gpus 4 \
    --gpus-per-model 1

# Example 2: Use max 4 GPUs, with 2 GPUs per model (2 models in parallel)
echo "Example 2: Running with max 4 GPUs total, 2 GPUs per model"
python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results_scheduled_parallel \
    --use-gpu-scheduling \
    --gpu-pool 0 1 2 3 4 5 6 7 \
    --max-total-gpus 4 \
    --gpus-per-model 2 \
    --num-workers 2

# Example 3: Limited GPU pool (only GPUs 4-7 available)
echo "Example 3: Using GPUs 4-7 only, max 2 at a time"
python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results_limited_pool \
    --use-gpu-scheduling \
    --gpu-pool 4 5 6 7 \
    --max-total-gpus 2 \
    --gpus-per-model 1

# Example 4: Without GPU scheduling (old behavior)
echo "Example 4: Traditional mode without GPU scheduling"
python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results_traditional \
    --device-ids 0 1 2 3 \
    --num-workers 2
