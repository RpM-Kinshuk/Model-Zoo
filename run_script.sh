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

# --- Paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
echo "Project Directory: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/Model-Zoo/svd_results"
# MODEL_LIST="$SCRIPT_DIR/esd_experiment/examples/atlas_models.csv"
MODEL_LIST="$PROJECT_ROOT/data/sampled_metadata.csv"
GPUS=(0 1 2)

# --- Periodic cache cleaner  ---
clean_cache() {
  echo "[$(date '+%F %T')] Clearing HF cache at: $HF_HOME"
  rm -rf "$HF_HOME" || true
  rm -rf "/scratch/kinshuk/.cache/huggingface" || true
  # Recreate directories so the running process can continue using the paths
  mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" || true
  mkdir -p "/scratch/kinshuk/.cache/huggingface" || true
}

start_cleaner() {
  (
    while true; do
      sleep 1500  # ~25 minutes
      clean_cache
    done
  ) &
  CLEANER_PID=$!
  echo "Started background cache cleaner (PID: $CLEANER_PID)"
}

start_cleaner

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
  --gpu_memory_threshold 500 \
  --max_check 1
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