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

The smallest input CSV for preparation has one column:

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

### 3 · Pin, review, run

```bash
python esd_experiment/run_experiment.py \
  --model_list models.csv \
  --output_dir analysis_runs/my_run --limit 5 --prepare_only
```

Preparation fetches metadata and adapter config JSON, not weights. Review the
generated `models.csv`; fix or remove rows marked `pin_status=error`, then launch:

```bash
python esd_experiment/run_experiment.py \
  --model_list analysis_runs/my_run/models.csv \
  --output_dir analysis_runs/my_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 1 \
  --max_concurrent_jobs 3
```

The prepared list pins full model and adapter-base commit SHAs. Keep it for
resume; preparation never overwrites it. Remote code is off by default—only use
`--trust_remote_code` for repositories you reviewed. For the curated workflow,
prepare from `data/curated/model_zoo_phase2.csv`; see the [analysis guide](docs/operations/analysis.md).

> [!TIP]
> `run_script.sh` is the HPC-oriented launch template. Adapt its paths, GPU list, and scheduler policy for your cluster.

## The system, visually

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":32,"rankSpacing":48},"themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","fontSize":"15px","lineColor":"#3978C5","clusterBkg":"#F7FAFF","clusterBorder":"#8DB9E8","edgeLabelBackground":"#FFFFFF"}}}%%
flowchart TB
    INPUT[("📋  MODEL MANIFEST")]

    subgraph CONTROL["CONTROL PLANE  ·  plan once, adapt live"]
        direction LR
        PREP["①  PREPARE<br/>validate · normalize · resume"]
        PREFLIGHT{"②  PREFLIGHT<br/>can this model run?"}
        QUEUE["③  QUEUE<br/>one job per model"]
        PREP --> PREFLIGHT
        PREFLIGHT -->|eligible| QUEUE
    end

    subgraph EXECUTION["EXECUTION PLANE  ·  isolated per model"]
        direction LR
        ALLOC["④  ALLOCATE<br/>poll · confirm · reserve"]
        LOAD["⑤  LOAD<br/>route · download · merge"]
        ESD["⑥  ANALYZE<br/>cost-sort · parallel ESD"]
        FINAL["⑦  FINALIZE<br/>validate · clean up"]
        ALLOC --> LOAD --> ESD --> FINAL
    end

    INPUT --> PREP
    QUEUE --> ALLOC
    PREFLIGHT -->|blocked| EXPLAIN["Known incompatibility<br/>explained before allocation"]
    FINAL --> DONE(["✓  MODEL COMPLETE"])

    CONFIG[["⚙  gpu_config.json"]] -. SIGHUP reload .-> ALLOC
    SIGNALS[["↯  runtime signals"]] -. drain / stop .-> ALLOC
    CACHE[("♻  isolated cache")] <--> LOAD
    WATCH[["♥  heartbeat supervisor"]] -. observe .-> LOAD
    WATCH -. observe .-> ESD

    classDef source fill:#FFF7E3,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef control fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef decision fill:#FFF4D6,stroke:#D69E00,color:#3B2A00,stroke-width:3px;
    classDef execute fill:#003B95,stroke:#002F6C,color:#FFFFFF,stroke-width:2px;
    classDef success fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:3px;
    classDef muted fill:#F2F4F7,stroke:#98A2B3,color:#344054,stroke-width:2px;
    class INPUT,CONFIG,SIGNALS,CACHE,WATCH source;
    class PREP,QUEUE control;
    class PREFLIGHT decision;
    class ALLOC,LOAD,ESD,FINAL execute;
    class DONE success;
    class EXPLAIN muted;

    style CONTROL fill:#F7FAFF,stroke:#8DB9E8,stroke-width:2px,color:#002F6C
    style EXECUTION fill:#F3F8FF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    linkStyle default stroke:#3978C5,stroke-width:2px
```

<p align="center"><sub>Gold = inputs & controls&nbsp;&nbsp;·&nbsp;&nbsp;Light blue = orchestration&nbsp;&nbsp;·&nbsp;&nbsp;Navy = GPU execution&nbsp;&nbsp;·&nbsp;&nbsp;Green = successful completion</sub></p>

| Component | Owns |
| --- | --- |
| `run_experiment.py` | Input normalization, resume filtering, preflight, job creation |
| `gputracker/` | GPU polling, reservations, concurrency, signals, supervision |
| `worker.py` | One model's load → analyze → finalize lifecycle |
| `model_loader.py` | Hugging Face class selection, revisions, adapters, format fallbacks |
| `net_esd/` | Layer selection, GPU work queue, spectral computation |

## Two levels of parallelism

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":36,"rankSpacing":52},"themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","fontSize":"15px","lineColor":"#3978C5","clusterBkg":"#F7FAFF","clusterBorder":"#8DB9E8","edgeLabelBackground":"#FFFFFF"}}}%%
flowchart TB
    POOL["PHYSICAL GPU POOL<br/>GPU 0 · GPU 1 · GPU 2 · GPU 3"]
    LIMIT[["max_concurrent_jobs = 2"]]

    POOL -->|reserve 2| QA
    POOL -->|reserve 2| QB
    LIMIT -. caps model processes .-> POOL

    subgraph A["MODEL WORKER A  ·  CUDA_VISIBLE_DEVICES=0,1"]
        direction TB
        QA[("cost-sorted layer queue")]
        QA --> A0["▦  local cuda:0<br/>physical GPU 0"]
        QA --> A1["▦  local cuda:1<br/>physical GPU 1"]
        A0 --> AO["ordered layer results"]
        A1 --> AO
    end

    subgraph B["MODEL WORKER B  ·  CUDA_VISIBLE_DEVICES=2,3"]
        direction TB
        QB[("cost-sorted layer queue")]
        QB --> B0["▦  local cuda:0<br/>physical GPU 2"]
        QB --> B1["▦  local cuda:1<br/>physical GPU 3"]
        B0 --> BO["ordered layer results"]
        B1 --> BO
    end

    OUTER["LEVEL 1<br/>independent model subprocesses"] -.-> POOL
    INNER["LEVEL 2<br/>one layer thread per visible GPU"] -.-> QA
    INNER -.-> QB

    classDef annotation fill:#FFF7E3,stroke:#FDB515,color:#3B2A00,stroke-width:2px;
    classDef scheduler fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:3px;
    classDef queue fill:#FFF4D6,stroke:#D69E00,color:#3B2A00,stroke-width:2px;
    classDef gpu fill:#003B95,stroke:#002F6C,color:#FFFFFF,stroke-width:2px;
    classDef result fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:2px;
    class POOL scheduler;
    class LIMIT,OUTER,INNER annotation;
    class QA,QB queue;
    class A0,A1,B0,B1 gpu;
    class AO,BO result;

    style A fill:#F7FAFF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    style B fill:#F7FAFF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    linkStyle default stroke:#3978C5,stroke-width:2px
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
  --model_list analysis_runs/my_run/models.csv \
  --output_dir analysis_runs/my_run \
  --gpus 0 1 2 3 \
  --num_gpus_per_job 2 \
  --max_concurrent_jobs 2
```

The library API also offers a spawned-process backend. Batch workers use the thread backend, avoiding per-layer tensor serialization.

## A worker's lifecycle

```mermaid
%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":34,"rankSpacing":48},"themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","fontSize":"15px","lineColor":"#3978C5","clusterBkg":"#F7FAFF","clusterBorder":"#8DB9E8","edgeLabelBackground":"#FFFFFF"}}}%%
flowchart TB
    subgraph WORKER["WORKER PROCESS  ·  one model, one process group"]
        direction LR
        START(["DISPATCHED"]) --> PREP["① PREPARE"]
        PREP --> LOAD["② LOAD"]
        LOAD --> ANALYZE["③ ANALYZE"]
        ANALYZE --> FINAL["④ FINALIZE"]
        FINAL --> OK(["✓ COMPLETE"])
    end

    LOAD -->|error| CLASSIFY
    ANALYZE -->|error| CLASSIFY
    FINAL -->|error| CLASSIFY

    subgraph RECOVERY["RECOVERY PATH"]
        direction LR
        CLASSIFY{"retryable +<br/>attempts left?"}
        CLEAN["release model<br/>empty CUDA cache"]
        FAIL(["TERMINAL FAILURE"])
        CLASSIFY -->|yes| CLEAN
        CLASSIFY -->|no| FAIL
    end
    CLEAN -->|next attempt| LOAD

    subgraph SUPERVISION["SUPERVISOR  ·  outside the worker process"]
        direction LR
        HEART["♥ heartbeat<br/>stage + state · every 30 s"]
        CLOCKS{"heartbeat stale?<br/>stage too long?"}
        POLICY{"policy"}
        LOG["mark + log"]
        STOP["SIGTERM<br/>grace period<br/>SIGKILL"]
        HEART --> CLOCKS
        CLOCKS -->|yes| POLICY
        POLICY -->|log| LOG
        POLICY -->|terminate| STOP
    end

    PREP -. updates .-> HEART
    LOAD -. updates .-> HEART
    ANALYZE -. updates .-> HEART
    FINAL -. updates .-> HEART
    STOP -. process group .-> FAIL

    classDef stage fill:#003B95,stroke:#002F6C,color:#FFFFFF,stroke-width:2px;
    classDef success fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:3px;
    classDef control fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef decision fill:#FFF4D6,stroke:#D69E00,color:#3B2A00,stroke-width:3px;
    classDef failure fill:#FDECEC,stroke:#B42318,color:#7A271A,stroke-width:3px;
    class START,OK success;
    class PREP,LOAD,ANALYZE,FINAL stage;
    class CLEAN,HEART,LOG control;
    class CLASSIFY,CLOCKS,POLICY decision;
    class STOP,FAIL failure;

    style WORKER fill:#F3F8FF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    style RECOVERY fill:#FFFBF0,stroke:#FDB515,stroke-width:2px,color:#3B2A00
    style SUPERVISION fill:#F7FAFF,stroke:#8DB9E8,stroke-width:2px,color:#002F6C
    linkStyle default stroke:#3978C5,stroke-width:2px
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
%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":38,"rankSpacing":48},"themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","fontSize":"15px","lineColor":"#3978C5","clusterBkg":"#F7FAFF","clusterBorder":"#8DB9E8","edgeLabelBackground":"#FFFFFF"}}}%%
flowchart TB
    ROOT[("♻  WORKER CACHE ROOT")]
    ROOT --> RUN["run_id  ·  isolates separate launches"]

    subgraph SCOPE["ONE DIRECTORY PER ACTIVE WORKER"]
        direction LR
        A["worker A<br/>model-000"]
        B["worker B<br/>model-001"]
        C["worker C<br/>model-002"]
    end

    RUN --> A
    RUN --> B
    RUN --> C
    A --> ENV["same variable names · different paths<br/>HF_HOME · HF_HUB_CACHE<br/>TRANSFORMERS_CACHE · HF_DATASETS_CACHE"]
    B --> ENV
    C --> ENV
    ENV --> EXIT{"job exits"}
    EXIT -->|success| CLEAN(["✓ remove cache"])
    EXIT -->|failure| CLEAN
    EXIT -->|terminated| CLEAN

    classDef source fill:#FFF7E3,stroke:#FDB515,color:#3B2A00,stroke-width:3px;
    classDef worker fill:#003B95,stroke:#002F6C,color:#FFFFFF,stroke-width:2px;
    classDef control fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef decision fill:#FFF4D6,stroke:#D69E00,color:#3B2A00,stroke-width:3px;
    classDef success fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:3px;
    class ROOT source;
    class A,B,C worker;
    class RUN,ENV control;
    class EXIT decision;
    class CLEAN success;

    style SCOPE fill:#F3F8FF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    linkStyle default stroke:#3978C5,stroke-width:2px
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
%%{init: {"theme":"base","flowchart":{"curve":"basis","nodeSpacing":34,"rankSpacing":50},"themeVariables":{"fontFamily":"Inter, ui-sans-serif, system-ui","fontSize":"15px","lineColor":"#3978C5","clusterBkg":"#F7FAFF","clusterBorder":"#8DB9E8","edgeLabelBackground":"#FFFFFF"}}}%%
flowchart TB
    MODEL[("NEURAL NETWORK")]

    subgraph PREPARE["①  SELECT + PREPARE"]
        direction LR
        LAYERS["Linear · Embedding<br/>Conv1d/2d/3d · HF Conv1D"]
        FILTER["skip unsupported weights<br/>and Linear aspect ratio ≥ 8"]
        MATRIX["whole 2D weight matrix<br/>or batched conv matrices"]
        LAYERS --> FILTER --> MATRIX
    end

    subgraph SOLVE["②  COMPUTE THE SPECTRUM"]
        direction LR
        CHOICE{"solver"}
        GRAM["--no-use_svd<br/>smaller symmetric<br/>Gram matrix"]
        SVD["DEFAULT<br/>direct singular<br/>values"]
        STABLE["sort · threshold<br/>numerical safeguards"]
        CHOICE -->|fast path| GRAM
        CHOICE -->|robust path| SVD
        GRAM --> STABLE
        SVD --> STABLE
    end

    subgraph INTERPRET["③  FIT + INTERPRET"]
        direction LR
        XMIN{"choose x-min"}
        FIT["power-law α<br/>+ KS distance"]
        METRICS["spectral norm · stable rank<br/>entropy · matrix rank<br/>norm + α-weighted measures"]
        XMIN --> FIT --> METRICS
    end

    MODEL --> LAYERS
    MATRIX --> CHOICE
    STABLE --> XMIN
    METRICS --> DONE(["✓  LAYER PROFILE"])

    MID[["xmin_mid"]] -. fast .-> XMIN
    PEAK[["xmin_peak"]] -. histogram .-> XMIN
    DKS[["DKS"]] -. full scan .-> XMIN

    classDef source fill:#FFF7E3,stroke:#FDB515,color:#3B2A00,stroke-width:3px;
    classDef stage fill:#EAF3FF,stroke:#006CE4,color:#002F6C,stroke-width:2px;
    classDef execute fill:#003B95,stroke:#002F6C,color:#FFFFFF,stroke-width:2px;
    classDef decision fill:#FFF4D6,stroke:#D69E00,color:#3B2A00,stroke-width:3px;
    classDef success fill:#E8F5EF,stroke:#00693E,color:#00452A,stroke-width:3px;
    class MODEL,MID,PEAK,DKS source;
    class LAYERS,FILTER,SPLIT,MATRIX,STABLE stage;
    class GRAM,SVD,FIT,METRICS execute;
    class CHOICE,XMIN decision;
    class DONE success;

    style PREPARE fill:#F7FAFF,stroke:#8DB9E8,stroke-width:2px,color:#002F6C
    style SOLVE fill:#F3F8FF,stroke:#006CE4,stroke-width:2px,color:#002F6C
    style INTERPRET fill:#F7FAFF,stroke:#8DB9E8,stroke-width:2px,color:#002F6C
    linkStyle default stroke:#3978C5,stroke-width:2px
```

The estimator avoids copying the model, skips linear matrices with aspect ratio ≥ 8, batches convolution kernels, and uses pinned host memory for CPU→GPU transfers. The Gram path symmetrizes its matrix, retries with diagonal jitter, then falls back to SVD if eigendecomposition remains unstable.

### Tune the analysis

| Option | Default | Choice |
| --- | ---: | --- |
| `--fix_fingers` | `xmin_mid` | `xmin_mid` · `xmin_peak` · `DKS` |
| `--evals_thresh` | `1e-5` | Absolute cutoff for fitting and retained metrics; saved spectra stay full |
| `--bins` | `100` | Histogram resolution for `xmin_peak` |
| `--use_svd` | on | Direct SVD; `--no-use_svd` selects Gram eigenvalues |
| `--save_eigs` | on | Save full spectra; `--no-save_eigs` stores scalars only |
| `--filter_zeros` | on | Retain values strictly above the threshold |
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
python -m pytest esd_experiment/tests -q \
  --ignore=esd_experiment/tests/test_setup.py \
  --ignore=esd_experiment/tests/test_gpu.py
python esd_experiment/tests/test_setup.py
python esd_experiment/tests/test_gpu.py
```

## Read next

| Guide | Best for |
| --- | --- |
| [Experiment runner](esd_experiment/README.md) | Quick start and code map |
| [Analysis guide](docs/operations/analysis.md) | Measurements, storage, resume and GPU supervision |
| [Operations](docs/operations/README.md) | End-to-end workflow context |

---

<p align="center">
  Built for experiments that should keep moving—even when individual models do not.
  <br><br>
  <a href="LICENSE">License</a>
</p>
