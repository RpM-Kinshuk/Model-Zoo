#!/bin/bash
set -euo pipefail

# --- Signal-safe background cleaner management ---
cleanup_bg() {
  if [[ -n "${CLEANER_PID:-}" ]]; then
    # Kill the cleaner subshell
    kill "${CLEANER_PID}" 2>/dev/null || true

    # Also kill any child processes (like `sleep`) of the cleaner
    if command -v pkill >/dev/null 2>&1; then
      pkill -P "${CLEANER_PID}" 2>/dev/null || true
    else
      for cp in $(ps -o pid= --ppid "${CLEANER_PID}" 2>/dev/null || true); do
        kill "$cp" 2>/dev/null || true
      done
    fi

    # Reap the subshell
    wait "${CLEANER_PID}" 2>/dev/null || true
  fi
}

on_signal() {
  echo "Caught signal, exiting..."
  cleanup_bg
  exit 130
}

trap on_signal INT TERM
trap cleanup_bg EXIT

# --- Environment ---
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

# --- Paths ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
echo "Project Directory: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/Model-Zoo/sample_results"
MODEL_LIST="$SCRIPT_DIR/esd_experiment/examples/atlas_models.csv"
GPUS=(2 3 4 5 6 7)

# --- Periodic cache cleaner (every 10 minutes) ---
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
    # Exit promptly if this subshell gets a TERM/INT
    trap 'exit 0' INT TERM
    while true; do
      sleep 1000  # ~15 minutes
      clean_cache
    done
  ) &
  CLEANER_PID=$!
  echo "Started background cache cleaner (PID: $CLEANER_PID)"
}

echo "Running with GPUs: ${GPUS[*]}"

# Start periodic cleaner
start_cleaner

# --- Run Python, capture exit code explicitly ---
set +e
python "$PROJECT_ROOT/Model-Zoo/esd_experiment/run_experiment.py" \
  --model_list "$MODEL_LIST" \
  --output_dir "$OUTPUT_DIR" \
  --gpus ${GPUS[@]} \
  --num_gpus_per_job 3 \
  --fix_fingers DKS \
  --filter_zeros \
  --gpu_memory_threshold 500 \
  --max_check 1
PY_EXIT_CODE=$?
set -e

# Stop background cleaner before exiting
cleanup_bg

# Propagate exit status with helpful messages
if [[ $PY_EXIT_CODE -eq 130 ]]; then
  echo "Script was interrupted"
  exit 130
elif [[ $PY_EXIT_CODE -ne 0 ]]; then
  echo "Python exited with code $PY_EXIT_CODE"
  exit "$PY_EXIT_CODE"
fi

echo "Python completed successfully."