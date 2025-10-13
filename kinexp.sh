#!/bin/bash
set -e

trap 'echo "Caught signal, exiting..."; exit 130' INT TERM

echo "Running with max 4 GPUs total, 2 GPUs per model"

python esd_analysis/run_analysis.py \
    --model-list atlas_metadata.csv \
    --output-dir ./results \
    --device-ids 0 1 2 3 4 5 6 7 \
    --num-workers 2

if [ $? -eq 130 ]; then
    echo "Script was interrupted"
    exit 130
fi