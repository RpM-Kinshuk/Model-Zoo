# Quickstart

From the repository root:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 1
```

From `esd_experiment/`:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

Check after launch:

- scheduler log: `<output_dir>/logs/esd_experiment.log`
- live worker state: `<output_dir>/logs/current_state.json`
- failures: `<output_dir>/logs/failure_records.jsonl`
- successes: both `stats/*.csv` and matching `metrics/*.h5`

For HPC runs, use the repository-root `run_script.sh` as the reference wrapper.
