<p align="center">
  <img src="docs/assets/model-zoo-hero.png" width="100%" alt="Models flowing through a GPU cluster into neural-network layers and spectral curves">
</p>

<h1 align="center">Model-Zoo</h1>

<p align="center">
  <strong>GPU-aware orchestration for large-scale neural-network spectral analysis</strong>
</p>

<p align="center">
  Turn a list of Hugging Face models into isolated, supervised ESD jobs—without manually assigning GPUs, babysitting workers, or cleaning caches.
</p>

<p align="center">
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-003262?style=for-the-badge&logo=python&logoColor=FDB515">
  <img alt="PyTorch and CUDA" src="https://img.shields.io/badge/PyTorch-CUDA-003B95?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-Models-006CE4?style=for-the-badge&logo=huggingface&logoColor=white">
  <img alt="Tests: 146 passing" src="https://img.shields.io/badge/Tests-146_passing-00693E?style=for-the-badge&logo=pytest&logoColor=white">
</p>

---

## At a glance

| 🎯 GPU-aware dispatch | ⚡ Two-level parallelism | 🛡️ Supervised workers |
| :--- | :--- | :--- |
| Waits for genuinely free GPUs and reserves them per model. | Runs models concurrently, then distributes layers within each model. | Tracks heartbeats, stages, PIDs, process groups, and failures. |
| **♻️ Ephemeral caches** | **🧩 Format-aware loading** | **🎛️ Live control** |
| Isolates every worker cache and removes it after the job. | Routes standard, adapter, multimodal, and quantized repositories. | Reloads GPUs and limits at runtime; supports drain and hard stop. |

## Start in three steps

### 1 · Create the environment

```bash
git clone https://github.com/RpM-Kinshuk/Model-Zoo.git
cd Model-Zoo
conda env create -f environment.yml
conda activate esd_ind
```

For gated or private repositories:

```bash
export HF_TOKEN="<your-token>"
```

### 2 · List the models

The smallest valid CSV has one column:

```csv
model_id
openai-community/gpt2
google/flan-t5-small
```

Adapters can name their base explicitly:

```csv
model_id,base_model_relation,source_model
org/my-lora,adapter,org/base-model
```

Model revisions work as `org/model@revision` or through the optional `revision_norm` column. Curated tables may also carry loader, architecture, file, pipeline, and hub-availability hints for better preflight decisions.

### 3 · Run

```bash
python esd_experiment/run_experiment.py \
  --model_list models.csv \
  --output_dir analysis_runs/my_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 1 \
  --max_concurrent_jobs 3
```

Use `--limit 5` for a smoke run. For the curated workflow, use `data/curated/model_zoo_phase2.csv`.

> [!TIP]
> `run_script.sh` is the HPC-oriented launch template. Adapt its paths, GPU list, and scheduler policy for your cluster.

## The system, visually

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","primaryColor":"#EAF3FF","primaryTextColor":"#002F6C","primaryBorderColor":"#006CE4","lineColor":"#006CE4","secondaryColor":"#FFF4D6","tertiaryColor":"#E8F5EF","clusterBkg":"#F8FAFC","clusterBorder":"#8DB9E8"}}}%%
flowchart LR
    CSV[(Model CSV)] --> RUN

    subgraph ORCH[Orchestration]
        direction TB
        RUN[Validate + resume] --> PREFLIGHT{Preflight}
        PREFLIGHT -->|eligible| QUEUE[Worker queue]
        PREFLIGHT -->|blocked| DIAG[Diagnostic]
    end

    subgraph SCHED[GPU scheduler]
        direction TB
        QUEUE --> POLL[Poll GPU memory]
        POLL --> RESERVE[Reserve GPU set]
        RESERVE --> SUP[Launch + supervise]
    end

    subgraph JOB[Isolated model worker]
        direction TB
        SUP --> LOAD[Load / merge model]
        LOAD --> ESD[Parallel layer ESD]
        ESD --> FINAL[Validate + finalize]
    end

    CONFIG[[gpu_config.json]] -. SIGHUP .-> SCHED
    SIGNALS[[Runtime signals]] -. drain / stop .-> SCHED
    CACHE[(Worker cache)] <--> LOAD

    classDef gold fill:#FFF4D6,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef blue fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef green fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:2px;
    classDef dark fill:#003B95,stroke:#003262,color:#FFFFFF,stroke-width:2px;
    class CSV,CONFIG,SIGNALS,CACHE gold;
    class RUN,QUEUE,POLL,RESERVE blue;
    class PREFLIGHT,DIAG green;
    class SUP,LOAD,ESD,FINAL dark;
```

| Component | Owns |
| --- | --- |
| `run_experiment.py` | Input normalization, resume filtering, preflight, job creation |
| `gputracker/` | GPU polling, reservations, concurrency, signals, supervision |
| `worker.py` | One model's load → analyze → finalize lifecycle |
| `model_loader.py` | Hugging Face class selection, revisions, adapters, format fallbacks |
| `net_esd/` | Layer selection, GPU work queue, spectral computation |

## Two levels of parallelism

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","lineColor":"#006CE4","clusterBkg":"#F8FAFC","clusterBorder":"#8DB9E8"}}}%%
flowchart TB
    Q[Model queue] --> A[Model A]
    Q --> B[Model B]

    subgraph OUTER[Level 1 · model parallelism]
        A --> GA[Reserved GPUs 0 + 1]
        B --> GB[Reserved GPUs 2 + 3]
    end

    subgraph INNERA[Level 2 · layers inside Model A]
        GA --> A0[GPU 0<br/>large layers first]
        GA --> A1[GPU 1<br/>shared task queue]
    end

    subgraph INNERB[Level 2 · layers inside Model B]
        GB --> B0[GPU 2<br/>large layers first]
        GB --> B1[GPU 3<br/>shared task queue]
    end

    classDef model fill:#FFF4D6,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef group fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef gpu fill:#003B95,stroke:#003262,color:#FFFFFF,stroke-width:2px;
    class Q,A,B model;
    class GA,GB group;
    class A0,A1,B0,B1 gpu;
```

### Level 1 · models across GPUs

The dispatcher considers a physical GPU free when:

```text
used memory < --gpu_memory_threshold
AND it is not reserved by this run
AND it passes --max_check consecutive polls
```

It atomically reserves `--num_gpus_per_job` devices, exposes only those devices through `CUDA_VISIBLE_DEVICES`, and releases them in a `finally` block on every exit path.

`--max_concurrent_jobs` adds a separate cap for host RAM, network, filesystem, or API-rate constraints. Without it, free GPUs determine concurrency.

> [!NOTE]
> GPU IDs on the CLI are physical IDs. CUDA remaps them inside a worker: a worker assigned physical GPU 6 will usually see it as local `cuda:0`.

### Level 2 · layers across each worker's GPUs

`net_esd` estimates per-layer compute cost, schedules the largest layers first, and feeds them to one fixed-device thread per visible GPU. A shared queue keeps faster GPUs busy while result ordering remains deterministic.

```bash
# Two concurrent models, two GPUs available to each model
python esd_experiment/run_experiment.py \
  --model_list models.csv \
  --output_dir analysis_runs/my_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 2 \
  --max_concurrent_jobs 2
```

The library API also offers a spawned-process backend. Batch workers use the thread backend, avoiding per-layer tensor serialization.

## A worker's lifecycle

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","lineColor":"#006CE4"}}}%%
flowchart LR
    START([Dispatched]) --> PREP[Prepare]
    PREP --> LOAD[Load]
    LOAD --> ANALYZE[Analyze]
    ANALYZE --> SAVE[Finalize]
    SAVE --> OK([Complete])

    LOAD -->|retryable error| RETRY{Retries left?}
    ANALYZE -->|retryable error| RETRY
    SAVE -->|retryable error| RETRY
    RETRY -->|yes| CLEAN[Free model + CUDA cache]
    CLEAN --> LOAD
    RETRY -->|no| FAIL([Terminal failure])

    HEARTBEAT[[Heartbeat thread<br/>every 30 s]] -. stage + state .-> PREP
    HEARTBEAT -.-> LOAD
    HEARTBEAT -.-> ANALYZE
    HEARTBEAT -.-> SAVE
    WATCH[[Supervisor]] -. timeout .-> KILL[TERM → grace → KILL]
    KILL --> FAIL

    classDef stage fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef success fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:2px;
    classDef warning fill:#FFF4D6,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef failure fill:#FDECEC,stroke:#B42318,color:#7A271A,stroke-width:2px;
    class PREP,LOAD,ANALYZE,SAVE stage;
    class START,OK success;
    class RETRY,CLEAN,HEARTBEAT,WATCH warning;
    class KILL,FAIL failure;
```

One dispatch thread walks the queue. Each active model gets a lightweight controller thread, a separate subprocess, and its own Unix process group. This isolates model-specific crashes and gives the supervisor a precise termination boundary.

### Retries and resume

| Mechanism | What it handles |
| --- | --- |
| Loader fallback | Meta-tensor reload, corrected model-class routing, spectral-only `AutoModel` fallback |
| Worker retry | CUDA OOM, load errors, analysis exceptions, finalization errors |
| Terminal failure | Unsupported formats, unresolved adapter bases, gated/missing repos, empty analysis |
| Resume | Skips complete work; clears and regenerates partial work |

> [!IMPORTANT]
> The direct worker supports `--max_retries` and defaults to zero. The batch runner does not currently forward a whole-job retry count, so batch jobs make one attempt while still using loader-level corrective fallbacks. On the next batch run, failures are retried unless `--skip_failed` is set.

Use `--overwrite` to regenerate all selected models, or `--skip_failed` to move past the recorded failure set.

## Smart model loading

Preflight probes optional backends in GPU-hidden child processes, preventing imports from polluting scheduler CUDA state. Eligible jobs then follow the most specific known route.

| Route | Behavior |
| --- | --- |
| Causal / seq2seq / classification | Selects the matching Transformers auto class |
| Multimodal | Uses the image-text auto model path when metadata indicates it |
| PEFT / LoRA | Resolves the base, loads both, then merges and unloads the adapter |
| GPTQ | Applies compatibility handling and loads when the backend is healthy |
| GGUF | Resolves a repository GGUF file for the Transformers loader |
| Compressed tensors | Runs only when its optional backend imports successfully |
| Alternate quantization | Rejects EXL2 explicitly; reports unsupported merge paths clearly |

If an adapter's base cannot be inferred confidently, add `source_model` to the CSV. Explicit metadata wins over guesswork.

## Cache: isolated by design

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","lineColor":"#006CE4"}}}%%
flowchart LR
    ROOT[(worker_cache_root)] --> RUN[run_id]
    RUN --> A[worker A]
    RUN --> B[worker B]
    A --> ENV1[HF_HOME<br/>HF_HUB_CACHE<br/>TRANSFORMERS_CACHE]
    B --> ENV2[HF_HOME<br/>HF_HUB_CACHE<br/>TRANSFORMERS_CACHE]
    A -->|finish / fail / kill| CLEAN1([remove])
    B -->|finish / fail / kill| CLEAN2([remove])

    classDef root fill:#FFF4D6,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef worker fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef clean fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:2px;
    class ROOT,RUN root;
    class A,B,ENV1,ENV2 worker;
    class CLEAN1,CLEAN2 clean;
```

Every job gets `<worker_cache_root>/<run_id>/<worker_id>/`. Isolation prevents partial-download collisions; automatic removal prevents long runs from filling scratch storage.

```bash
--worker_cache_root /scratch/$USER/model_zoo_worker_cache
```

The default is `MODEL_ZOO_WORKER_CACHE_ROOT` or `/scratch/kinshuk/hf_worker_cache`. Pass an empty value to inherit a shared Hugging Face cache instead.

## Live controls

At startup, scheduler options become `<output_dir>/gpu_config.json`. Edit the file, then reload it without interrupting active workers:

```bash
kill -HUP <runner-pid>
```

```json
{
  "available_gpus": [0, 1, 2, 3],
  "max_checks": 5,
  "memory_threshold_mb": 500,
  "max_concurrent_jobs": 2,
  "stale_process_action": "log",
  "heartbeat_timeout_seconds": 3600,
  "stage_timeout_seconds": {
    "load": 7200,
    "analyze": 28800,
    "save": 1800,
    "default": 14400
  },
  "termination_grace_seconds": 30
}
```

| Signal | Effect |
| --- | --- |
| 🔄 `SIGHUP` | Reload GPU pool, concurrency, and timeout policy |
| 🟡 `SIGUSR1` | Drain: stop dispatching, let active workers finish |
| 🔴 `SIGINT` / `SIGTERM` | Hard stop all active worker process groups |

Removing a GPU from the live pool affects future jobs; it does not evict a worker already using that GPU.

### Stale-worker policy

Two clocks catch different failure modes:

```text
heartbeat timeout  → the heartbeat writer stopped
stage timeout      → heartbeat is alive, but load/analyze/save is stuck
```

Start with `stale_process_action: "log"` while tuning. Once the limits fit your model sizes, switch to `"terminate"`; stale process groups receive `SIGTERM`, a grace period, then `SIGKILL` if required.

## ESD in one picture

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","lineColor":"#006CE4"}}}%%
flowchart LR
    MODEL[Model] --> LAYERS[Linear · Conv1d · Conv2d]
    LAYERS --> SPLIT[Split eligible attention Q / K / V]
    SPLIT --> MATRIX[2D matrix / batched conv matrices]
    MATRIX --> SPECTRUM{Spectrum}
    SPECTRUM -->|default| GRAM[Smaller Gram matrix]
    SPECTRUM -->|--use_svd| SVD[Direct SVD]
    GRAM --> FIT[Power-law fit + KS distance]
    SVD --> FIT
    FIT --> METRICS[α · spectral norm · stable rank<br/>entropy · matrix rank · norm metrics]

    classDef input fill:#FFF4D6,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef stage fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef result fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:2px;
    class MODEL input;
    class LAYERS,SPLIT,MATRIX,SPECTRUM,GRAM,SVD,FIT stage;
    class METRICS result;
```

The estimator avoids copying the model, skips linear matrices with aspect ratio ≥ 8, batches convolution kernels, and uses pinned host memory for CPU→GPU transfers. The Gram path symmetrizes its matrix, retries with diagonal jitter, then falls back to SVD if eigendecomposition remains unstable.

### Tune the analysis

| Option | Default | Choice |
| --- | ---: | --- |
| `--fix_fingers` | `xmin_mid` | `xmin_mid` · `xmin_peak` · `DKS` |
| `--evals_thresh` | `1e-5` | Near-zero eigenvalue cutoff |
| `--bins` | `100` | Histogram resolution for `xmin_peak` |
| `--use_svd` | off | Direct SVD instead of the Gram path |
| `--filter_zeros` | on | Filter values below the threshold |
| `--parallel_esd` | on | Distribute layers across visible GPUs |

`xmin_mid` is fastest. `xmin_peak` focuses cutoff candidates near the histogram peak. `DKS` scans valid cutoffs and minimizes the Kolmogorov–Smirnov distance.

## Scheduler cheat sheet

| Goal | Use |
| --- | --- |
| Choose physical devices | `--gpus 0 1 2 3` |
| Reserve N GPUs per model | `--num_gpus_per_job N` |
| Limit active models | `--max_concurrent_jobs N` |
| Require nearly empty GPUs | `--gpu_memory_threshold 500` |
| Confirm a GPU stays free | `--max_check 5` |
| Change timeout behavior | `--stale_process_action log\|terminate` |
| Resume but ignore old failures | `--skip_failed` |
| Recompute selected models | `--overwrite` |

```bash
python esd_experiment/run_experiment.py --help
```

## Observe a live run

```bash
watch -n 5 'python -m json.tool analysis_runs/my_run/logs/current_state.json'
```

The live snapshot surfaces the runner, active workers, current stages, physical GPUs, PIDs, process groups, heartbeat data, log paths, and cache paths. The scheduler log is `logs/esd_experiment.log`; per-worker active metadata and logs are cleaned after each job, while terminal diagnostics remain available for post-mortem debugging.

<details>
<summary><strong>Nothing is starting</strong></summary>

- Check `gpustat` or `nvidia-smi`; selected GPUs may exceed the threshold.
- Confirm every ID passed to `--gpus` exists on the node.
- Reduce `--max_check` for a faster allocation decision.
- Read the scheduler log for preflight blocks or already-complete rows.

</details>

<details>
<summary><strong>A worker is alive but stuck</strong></summary>

Inspect its stage and log path in `current_state.json`. Keep the stale action at `log` while calibrating; switch it to `terminate` and send `SIGHUP` once the timeout windows are trustworthy.

</details>

<details>
<summary><strong>CUDA out of memory</strong></summary>

- Give `device_map=auto` more GPUs per job when a model needs sharding.
- Reduce concurrent jobs when simultaneous loads exhaust host resources.
- Lower the memory threshold to make allocation more conservative.
- Use the worker log to distinguish loading OOM from layer-analysis OOM.

</details>

<details>
<summary><strong>Private model or unresolved adapter</strong></summary>

Export an authorized `HF_TOKEN` for private/gated repositories. For adapters, add `source_model` explicitly when hub metadata is incomplete.

</details>

## Code map

```text
esd_experiment/
├── run_experiment.py          public entrypoint
├── src/
│   ├── run_experiment.py      preflight, resume, jobs
│   ├── model_preflight.py     eligibility + routing
│   ├── worker.py              single-model lifecycle
│   └── model_loader.py        Hugging Face loading
├── gputracker/
│   ├── gputracker.py          dispatch + supervision
│   └── supervision.py         live state + cache cleanup
└── tests/

net_esd/
├── __init__.py                estimator + layer scheduler
├── core.py                    spectral computation
└── utils.py                   layer filtering + cost model
```

## Verify the installation

```bash
python -m pytest esd_experiment/tests -q
python esd_experiment/tests/test_setup.py
python esd_experiment/tests/test_gpu.py
```

Current suite: **146 passing tests**.

## Read next

| Guide | Best for |
| --- | --- |
| [Quick start](esd_experiment/docs/QUICKSTART.md) | The shortest launch path |
| [Infrastructure overview](esd_experiment/docs/OVERVIEW.md) | Component boundaries |
| [GPU supervision](esd_experiment/docs/GPU_FIX.md) | Assignment and stale-worker details |
| [Operations](docs/operations/README.md) | End-to-end workflow context |

---

<p align="center">
  Built for experiments that should keep moving—even when individual models do not.
  <br><br>
  <a href="LICENSE">License</a>
</p>
