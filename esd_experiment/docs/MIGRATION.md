# Migration Notes

Older docs and scripts used flatter names. Current paths are:

| Legacy name | Current path |
| --- | --- |
| `run_esd_experiment.py` | `run_experiment.py` wrapper or `src/run_experiment.py` implementation |
| `esd_worker.py` | `src/worker.py` |
| `model_loader.py` at root | `src/model_loader.py` |
| `analyze_results.py` implementation | `utils/analyze_results.py` |
| `test_gpu.py` | `tests/test_gpu.py` |
| `example_workflow.sh` | `examples/workflow.sh` |

Current command from the repository root:

```bash
python esd_experiment/run_experiment.py \
  --model_list data/curated/model_zoo_phase2.csv \
  --output_dir analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```

Current command from `esd_experiment/`:

```bash
python run_experiment.py \
  --model_list ../data/curated/model_zoo_phase2.csv \
  --output_dir ../analysis_runs/phase2/example_run \
  --gpus 0 1 2 3
```
