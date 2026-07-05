#!/bin/bash
set -uo pipefail

# ------------------ Signal-safe background cleaner management -----------------
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

# -------------------------------- Environment ---------------------------------
CACHE_DIR="/scratch/kinshuk/.cache"
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
: "${HF_TOKEN:?Please export HF_TOKEN in your shell before running (do not hardcode it in run_script.sh)}"; export HF_TOKEN
export HF_HOME="$CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface"
export WORKER_CACHE_ROOT="$CACHE_DIR/hf_worker_cache"

# ------------------------------------ Paths -----------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
echo "Project Directory: $PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/results"
mkdir -p "$OUTPUT_DIR" || true
MODEL_LIST="$SCRIPT_DIR/data/curated/model_zoo_phase2.csv"
# MODEL_LIST="$SCRIPT_DIR/data/ablation/ablation_selected.csv"
GPUS=(0 1 2 3 4 5 6 7)


echo "Running with GPUs: ${GPUS[*]}"

# ------------------ Run Python, capture exit code explicitly ------------------
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
  --worker_cache_root "$WORKER_CACHE_ROOT" \
  --stale_process_action terminate \
  --heartbeat_timeout_seconds 7200 \
  --stage_timeout_seconds load=1800,analyze=28800,save=1800,default=14400 \
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
