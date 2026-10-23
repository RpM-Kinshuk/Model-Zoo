# Model-Zoo

Build a dataset of Hugging Face checkpoint spectra: per-weight ESD measurements,
power-law fit diagnostics, eigenvalues, coverage and pinned provenance.

The operating workflow is **prepare → review → run → index → query**.
Collection is GPU-aware and supervised, with private disposable caches and one
model worker by default. It is ready for bounded research collection, not yet
proven at 50,000-model scale. Broad architecture coverage is not universal support,
and spectral descriptors are not validated model-quality scores.

## 1 · Create the environment

```bash
git clone https://github.com/RpM-Kinshuk/Model-Zoo.git
cd Model-Zoo
conda env create -f environment.yml
conda activate model_zoo
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python esd_experiment/tests/test_setup.py
```

Use a new environment; do not update the shared `esd_ind` environment in place.
Conda supplies Python and pip supplies the analysis packages. These are direct
dependency pins, not a full transitive lock; the CUDA wheel pairing follows the
[official PyTorch recipe](https://pytorch.org/get-started/previous-versions/#v2110).
For CPU-only work, use its `cpu` wheel index. The setup check is offline and CPU-only.

Optional quantization backends and the
[clustering dashboard](clustering/unsupervised/README.md) are separate from the
basic collection workflow. Install only what you actually need; a healthy
backend alone does not guarantee a checkpoint can be measured.

For gated/private repositories, supply an authorized token externally:

```bash
export HF_TOKEN="<your-token>"
```

## 2 · Prepare and review a small list

Create `models.csv`:

```csv
model_id
openai-community/gpt2
google/flan-t5-small
```

Adapters may also supply `base_model_relation=adapter` and
`source_model=org/base-model`. Revisions can be given as `org/model@revision`
or in `revision_norm`.

```bash
python esd_experiment/run_experiment.py \
  --model_list models.csv \
  --output_dir analysis_runs/phase2/my_run --limit 5 --prepare_only
```

Preparation downloads metadata and small config/index JSON, not model weights.
Review the generated `models.csv`: pins, config labels, selected weight sizes,
`preparation_status` and `preparation_reason`. Ready rows can launch; held rows
remain visible and are not dispatched.

For the larger metadata collection, start from
`data/curated/model_zoo_phase2.csv`. Its old labels/ready subsets have known
[limitations](docs/operations/curation.md#architecture-label-caveat); use pinned
config labels rather than guessing model families from names.

## 3 · Run, stop and resume

Run the prepared manifest on GPUs you are allowed to use:

```bash
python esd_experiment/run_experiment.py \
  --model_list analysis_runs/phase2/my_run/models.csv \
  --output_dir analysis_runs/phase2/my_run \
  --gpus 0 1 2 3 --num_gpus_per_job 1 \
  --worker_cache_root /scratch/$USER/model_zoo_worker_cache
```

The runner waits for repeated idle checks with no visible GPU processes. GPU
IDs are physical; the assigned GPU becomes local `cuda:0` inside a worker.
Idle checks do not reserve a device against other users.

Defaults: one worker, 6 GiB selected weights including adapter bases, 16 GiB
sampled worker-tree RAM and 12 GiB sampled private cache. Change the corresponding
`--max_*` options when actual collection costs justify it; sampled limits can
overshoot and do not predict GPU memory. Private caches are removed after
confirmed worker termination. Shared caches are not deleted.

Watch progress:

```bash
watch -n 5 'python -m json.tool analysis_runs/phase2/my_run/logs/current_state.json'
```

The runner PID is in that snapshot. Send `SIGUSR1` to drain (finish active
workers, launch no more); use Ctrl-C or `SIGTERM` to stop and clean up owned
workers. Allow cleanup to finish; avoid killing the runner with `SIGKILL`.
Stage/heartbeat timeouts stop stuck workers by default. Edit `gpu_config.json`
and send `SIGHUP` for live GPU/limit changes; details are in the
[supervision guide](docs/operations/analysis.md#scheduling-and-monitoring).

Resume with the same launch command, pinned manifest and measurement settings.
Compatible completed pairs are skipped; interrupted models restart, not individual
layers. Failures are retried on a later launch unless `--skip_failed` is set.
Incompatible/incomplete outputs require inspection; use a fresh directory for
changed measurement conventions. `--overwrite` deliberately replaces results.

Ordinary model failures do not stop the queue after successful cleanup.
Status, supervision or unconfirmed cleanup failures stop new dispatch and exit
nonzero. Inspect retained diagnostics/owned processes before resuming in that case.
The local HPC wrapper is `run_script.sh`; its site-specific settings are separate.

## 4 · Index and read results

```bash
python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/my_run

python esd_experiment/analyze_results.py \
  --results_dir analysis_runs/phase2/*/ \
  --output analysis_runs/phase2/catalog.csv
```

Each successful model has `stats/*.csv` and `metrics/*.h5`. Full eigenvalues
are saved by default, including values filtered out of fit/retained metrics.
`summary.csv` and the collection catalog are rebuildable per-model indexes;
the original pairs and terminal outcomes are authoritative. Repeated IDs,
different revisions/settings, failures and held rows remain separate.

Inspect `matrix_coverage` as well as `outcome`: a saved result can still be
partial. Missing fits are explicit, not missing layers. Use the
[named-spectrum query examples](docs/operations/analysis.md#query-models-and-read-a-named-spectrum)
to select a model and read one layer without loading all spectra.

## Measurement scope

Strict loading preserves checkpoint weights. The default `--analysis_source auto`
falls back to stored checkpoint matrices only for narrowly identified architecture
support failures—not damaged weights, network errors or OOM.

Declared Linear/Embedding/Conv layouts and ordinary floating dense 2D registered
parameters are measured, with canonical names and recorded layouts. Unknown
higher-dimensional tensors and unsupported packed/quantized representations
are not guessed or flattened. Checkpoint-only results describe stored matrices,
not verified model layers. Convolution kernel spectra are not full convolution
operator spectra. See [coverage and loader rules](docs/operations/analysis.md#loading-and-coverage).

Default fitting uses `xmin_mid`, SVD and an absolute retained-eigenvalue threshold
of `1e-5`. Precision, filtering, cutoff, versions and actual loading source are
recorded. A finite alpha or small KS distance does not establish power-law
behavior or generalization. See the [metric definitions](docs/operations/analysis.md#interpreting-the-metrics).

## For maintainers

Keep commands, names and helpers simple. Add support for recurring consequential
collection failures, not speculative universal machinery. Keep regression tests
for integrity, numerics, persistence and process ownership; general probes are
not a prerequisite for every collection batch.

```bash
python esd_experiment/run_experiment.py --help
# Optional development dependency; not needed by workers:
python -m pip install 'pytest>=8,<9'
python -m pytest esd_experiment/tests -q
```

Code: `esd_experiment/src/` loads and runs one model; `gputracker/` supervises
workers; `net_esd/` discovers weights and computes spectra;
`esd_experiment/utils/analyze_results.py` reads/indexes saved results.

[Analysis reference](docs/operations/analysis.md) ·
[Runner code map](esd_experiment/README.md#code-map) ·
[Operations and curation](docs/operations/README.md) · [License](LICENSE)
