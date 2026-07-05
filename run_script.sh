#!/bin/bash
set -uo pipefail

# --- Signal-safe background cleaner management ---
cleanup() {
    echo "Caught signal or exit. Cleaning up process group..."
    
    # Kill the background cleaner
    if [[ -n "${CLEANER_PID:-}" ]]; then
        kill "${CLEANER_PID}" 2>/dev/null || true
    fi

    # The Nuclear Option: Kill the entire process group (PGID).
    # This ensures python, children, and grandchildren die.
    # We use a trap on EXIT, so this runs on success, failure, or Ctrl+C.
    
    # 'kill 0' sends signal to every process in the process group of the calling process.
    # We trap SIGINT/TERM specifically to exit, which triggers the EXIT trap.
    trap '' EXIT INT TERM # Avoid recursion
    kill -TERM 0 2>/dev/null || true
    wait
}

trap cleanup EXIT INT TERM

# --- Environment ---
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export HF_HOME="/scratch/kinshuk/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/kinshuk/.cache/huggingface"
export MODEL_ZOO_WORKER_CACHE_ROOT="/scratch/kinshuk/.cache/hf_worker_cache"

# --- Paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
echo "Project Directory: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/results"
mkdir -p "$OUTPUT_DIR" || true
# MODEL_LIST="$SCRIPT_DIR/data/ablation/ablation_selected.csv"
MODEL_LIST="$SCRIPT_DIR/data/curated/model_zoo_phase2.csv"
GPUS=(0 1 2 3 4 5 6 7)

# --- Periodic cache cleaner  ---
clean_cache() {
  echo "[$(date '+%F %T')] Clearing HF cache at: $HF_HOME"
  rm -rf "$HF_HOME" || true
  # Recreate directories so the running process can continue using the paths
  mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" || true
}

start_cleaner() {
  (
    while true; do
      sleep 1800  # ~30 minutes
      clean_cache
    done
  ) &
  CLEANER_PID=$!
  echo "Started background cache cleaner (PID: $CLEANER_PID)"
}

# Worker caches are per-worker and cleaned by the dispatcher after each worker exits.
# Do not delete the shared HF cache while workers are active.
# start_cleaner

echo "Running with GPUs: ${GPUS[*]}"

# --- Run Python, capture exit code explicitly ---
# set +e
python "$PROJECT_ROOT/Model-Zoo/esd_experiment/run_experiment.py" \
  --model_list "$MODEL_LIST" \
  --output_dir "$OUTPUT_DIR" \
  --gpus ${GPUS[@]} \
  --num_gpus_per_job 1 \
  --fix_fingers DKS \
  --filter_zeros \
  --save_eigs \
  --gpu_memory_threshold 500 \
  --max_check 1 \
  --max_concurrent_jobs 1 \
  --worker_cache_root "$MODEL_ZOO_WORKER_CACHE_ROOT" \
  --stale_process_action terminate \
  --heartbeat_timeout_seconds 7200 \
  --stage_timeout_seconds load=4000,analyze=28800,save=1800,default=14400 \
  --termination_grace_seconds 30 \
  --skip_failed
PY_EXIT_CODE=$?
# set -e

# Propagate exit status with helpful messages
if [[ $PY_EXIT_CODE -eq 130 ]]; then
  echo "Script was interrupted"
  exit 130
elif [[ $PY_EXIT_CODE -ne 0 ]]; then
  echo "Python exited with code $PY_EXIT_CODE"
  exit "$PY_EXIT_CODE"
fi

echo "Python completed successfully."
