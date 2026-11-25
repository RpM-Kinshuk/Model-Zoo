#!/bin/bash
set -uo pipefail

# Load private credentials relative to this script, regardless of working directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/.env"
fi
: "${HF_TOKEN:?Set HF_TOKEN in Model-Zoo/.env or export it before running}"
export HF_TOKEN

# -------------------------------- Environment ---------------------------------
CACHE_DIR="/scratch/kinshuk/.cache"
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"
export WORKER_CACHE_ROOT="$CACHE_DIR/hf_worker_cache"

# ------------------------------------ Paths -----------------------------------
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
echo "Project Directory: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/results"
mkdir -p "$OUTPUT_DIR" || true
MODEL_LIST="$SCRIPT_DIR/data/curated/model_zoo_phase2.csv"
# MODEL_LIST="$SCRIPT_DIR/data/ablation/ablation_selected.csv"
GPUS=(0 1 2 3 4 5 6 7)

# Durable packages follow ZOO_DATA when set; shared parent blocks stay on scratch.
PACKAGES_ROOT="${ZOO_DATA:-$OUTPUT_DIR}/packages"
PARENT_CACHE_DIR="$CACHE_DIR/parent-blocks"


echo "Running with GPUs: ${GPUS[*]}"
echo "Model packages: $PACKAGES_ROOT"
echo "Parent block cache: $PARENT_CACHE_DIR"

# The runner owns worker cleanup and prints progress to the launch log.
# Auto budgets use reviewed weight sizes; reprepare older lists missing download_bytes.
# CPU evidence overlaps ESD; oversized models get supervised row-block evidence.
# Download-ahead is bounded separately from each worker's private cache.
exec python -u "$PROJECT_ROOT/Model-Zoo/esd_experiment/run_experiment.py" \
  --model_list "$MODEL_LIST" \
  --output_dir "$OUTPUT_DIR" \
  --gpus "${GPUS[@]}" \
  --num_gpus_per_job auto \
  --max_weight_gib auto \
  --fix_fingers DKS \
  --filter_zeros \
  --save_eigs \
  --gpu_memory_threshold 500 \
  --max_check 1 \
  --max_concurrent_jobs 3 \
  --worker_cache_root "$WORKER_CACHE_ROOT" \
  --max_worker_ram_gib auto \
  --max_worker_cache_gib auto \
  --prefetch_models auto \
  --prefetch_max_gib 64 \
  --evidence \
  --evidence_device cpu \
  --evidence_max_inflight_gib 2 \
  --with_parent \
  --parent_cache "$PARENT_CACHE_DIR" \
  --parent_cache_max_gib 64 \
  --publish \
  --packages_root "$PACKAGES_ROOT" \
  --oversized_models_evidence \
  --oversized_models_device auto \
  --stale_process_action terminate \
  --heartbeat_timeout_seconds 7200 \
  --stage_timeout_seconds auto \
  --termination_grace_seconds 30
