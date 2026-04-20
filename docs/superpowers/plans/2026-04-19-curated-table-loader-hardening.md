# Curated Table Loader Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Model-Zoo read the curated phase-2 table directly while hardening model loading, retries, and failure recording without changing the ESD core.

**Architecture:** Keep the current `run_experiment.py -> worker.py -> model_loader.py` flow, but add a small normalized job-record layer, scenario-aware loading, and structured failure output. The worker becomes responsible for temp-file safety and stage-aware failure records; the loader stays small and focused on returning either a model or a structured failure classification.

**Tech Stack:** Python, pandas, PyTorch, Transformers, PEFT, Hugging Face Hub, pytest

---

### Task 1: Add curated-table parsing and internal job normalization

**Files:**
- Modify: `esd_experiment/src/run_experiment.py`
- Test: `esd_experiment/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing tests for curated-table ingestion**

```python
import pandas as pd
from pathlib import Path

from esd_experiment.src.run_experiment import load_model_list


def test_load_model_list_accepts_curated_table_columns(tmp_path: Path):
    csv_path = tmp_path / "curated.csv"
    pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "revision_norm": "rev-a",
                "source_model": "base/model",
                "base_model_relation": "adapter",
                "loader_scenario": "adapter_requires_base",
                "primary_type_bucket": "adapter",
            },
            {
                "model_id": "org/model-b",
                "revision_norm": None,
                "source_model": "",
                "base_model_relation": "source",
                "loader_scenario": "standard_transformers",
                "primary_type_bucket": "base_source",
            },
        ]
    ).to_csv(csv_path, index=False)

    df = load_model_list(str(csv_path))

    assert list(df["model_id"]) == ["org/model-a", "org/model-b"]
    assert "revision_norm" in df.columns
    assert "loader_scenario" in df.columns
    assert "primary_type_bucket" in df.columns


def test_load_model_list_still_accepts_old_three_column_format(tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [{"model_id": "org/model-a", "base_model_relation": "adapter", "source_model": "base/model"}]
    ).to_csv(csv_path, index=False)

    df = load_model_list(str(csv_path))

    assert df.loc[0, "model_id"] == "org/model-a"
    assert df.loc[0, "base_model_relation"] == "adapter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_run_experiment.py -k curated -v`
Expected: FAIL because the current loader path does not guarantee the curated columns are preserved or normalized.

- [ ] **Step 3: Implement minimal curated-table normalization in `run_experiment.py`**

```python
def load_model_list(csv_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "model_id" not in df.columns:
        raise ValueError(f"CSV must have 'model_id' column. Found columns: {list(df.columns)}")

    defaults = {
        "base_model_relation": "",
        "source_model": "",
        "revision_norm": "",
        "loader_scenario": "",
        "primary_type_bucket": "",
        "lineage_status": "",
        "candidate_source": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    for column in defaults:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["model_id"] = df["model_id"].astype(str).str.strip()
    df = df[df["model_id"] != ""]
    df = df[df["model_id"] != "nan"]

    if limit is not None and limit > 0:
        df = df.head(limit)

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_run_experiment.py -k curated -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/run_experiment.py esd_experiment/tests/test_run_experiment.py
git commit -m "feat: accept curated table columns"
```

### Task 2: Pass normalized job fields through command generation

**Files:**
- Modify: `esd_experiment/src/run_experiment.py`
- Modify: `esd_experiment/src/worker.py`
- Test: `esd_experiment/tests/test_run_experiment.py`

- [ ] **Step 1: Write the failing test for worker command generation**

```python
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from esd_experiment.src.run_experiment import generate_commands


def test_generate_commands_passes_curated_loader_fields(tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "model_id": "org/model-a",
                "revision_norm": "rev-a",
                "source_model": "base/model",
                "base_model_relation": "adapter",
                "loader_scenario": "adapter_requires_base",
                "primary_type_bucket": "adapter",
            }
        ]
    )
    args = SimpleNamespace(
        fix_fingers="xmin_mid",
        evals_thresh=1e-5,
        bins=100,
        filter_zeros=True,
        use_svd=False,
        parallel_esd=True,
        overwrite=False,
    )

    commands = generate_commands(df, tmp_path, args)

    assert "--revision 'rev-a'" in commands[0]
    assert "--loader_scenario 'adapter_requires_base'" in commands[0]
    assert "--primary_type_bucket 'adapter'" in commands[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_run_experiment.py -k generate_commands -v`
Expected: FAIL because the command does not yet pass the curated fields.

- [ ] **Step 3: Implement minimal command passthrough**

```python
if row.get("revision_norm"):
    cmd_parts.append(f"--revision '{row['revision_norm']}'")
if row.get("loader_scenario"):
    cmd_parts.append(f"--loader_scenario '{row['loader_scenario']}'")
if row.get("primary_type_bucket"):
    cmd_parts.append(f"--primary_type_bucket '{row['primary_type_bucket']}'")
```

Also add matching parser args to `worker.py`:

```python
parser.add_argument("--revision", type=str, default="", help="Optional model revision")
parser.add_argument("--loader_scenario", type=str, default="", help="Curated loader scenario hint")
parser.add_argument("--primary_type_bucket", type=str, default="", help="Curated type bucket for logging")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_run_experiment.py -k generate_commands -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/run_experiment.py esd_experiment/src/worker.py esd_experiment/tests/test_run_experiment.py
git commit -m "feat: pass curated job fields to worker"
```

### Task 3: Add structured loader failures and scenario dispatch

**Files:**
- Modify: `esd_experiment/src/model_loader.py`
- Test: `esd_experiment/tests/test_model_loader.py`

- [ ] **Step 1: Write the failing tests for structured loader failure classification**

```python
from esd_experiment.src.model_loader import LoaderFailure, classify_loader_scenario_support


def test_classify_loader_scenario_support_rejects_gguf():
    failure = classify_loader_scenario_support("quantized_alt_format")
    assert isinstance(failure, LoaderFailure)
    assert failure.reason == "unsupported_loader_scenario"
    assert failure.stage == "load"


def test_classify_loader_scenario_support_accepts_standard_transformers():
    assert classify_loader_scenario_support("standard_transformers") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_model_loader.py -k scenario_support -v`
Expected: FAIL because the helper and failure type do not exist yet.

- [ ] **Step 3: Implement minimal structured loader failure support**

```python
from dataclasses import dataclass


@dataclass
class LoaderFailure(Exception):
    stage: str
    reason: str
    message: str

    def __str__(self) -> str:
        return self.message


def classify_loader_scenario_support(loader_scenario: Optional[str]) -> Optional[LoaderFailure]:
    scenario = (loader_scenario or "").strip().lower()
    if scenario in {"", "standard_transformers", "adapter_requires_base", "quantized_transformers_native", "multimodal_transformers"}:
        return None
    if scenario == "quantized_alt_format":
        return LoaderFailure("load", "unsupported_loader_scenario", "GGUF-style repos are not supported by the current loader")
    return LoaderFailure("load", "unsupported_loader_scenario", f"Unsupported loader scenario: {loader_scenario}")
```

And thread `loader_scenario` into `load_model(...)`, raising `LoaderFailure` early for unsupported scenarios.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_model_loader.py -k scenario_support -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/model_loader.py esd_experiment/tests/test_model_loader.py
git commit -m "feat: classify unsupported loader scenarios"
```

### Task 4: Harden adapter and revision-aware loading without rewriting the loader

**Files:**
- Modify: `esd_experiment/src/model_loader.py`
- Test: `esd_experiment/tests/test_model_loader.py`

- [ ] **Step 1: Write the failing tests for adapter base resolution and revision forwarding**

```python
from unittest.mock import Mock, patch

import pytest

from esd_experiment.src.model_loader import LoaderFailure, load_model


@patch("esd_experiment.src.model_loader.is_adapter_model", return_value=True)
def test_load_model_raises_structured_failure_for_unresolved_adapter(mock_is_adapter):
    with patch("esd_experiment.src.model_loader.resolve_base_model", side_effect=RuntimeError("missing base")):
        with pytest.raises(LoaderFailure) as exc:
            load_model("org/adapter", base_model_relation="adapter", source_model=None, loader_scenario="adapter_requires_base")
    assert exc.value.reason == "adapter_base_unresolved"


@patch("esd_experiment.src.model_loader.hf_from_pretrained")
def test_load_model_forwards_revision_to_standard_load(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model
    model, is_adapter = load_model("org/model", revision="rev-a", loader_scenario="standard_transformers")
    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.kwargs["revision"] == "rev-a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_model_loader.py -k "adapter or revision" -v`
Expected: FAIL because adapter resolution errors are not yet wrapped as structured failures.

- [ ] **Step 3: Implement minimal loader hardening**

```python
try:
    if is_adapter_model(repo_id, base_model_relation):
        try:
            model = load_and_merge_adapter(
                adapter_repo=repo_id,
                base_repo=source_model,
                device_map=device_map,
                torch_dtype=torch_dtype,
                revision=revision,
            )
        except RuntimeError as exc:
            raise LoaderFailure("load", "adapter_base_unresolved", str(exc)) from exc
        return model, True
except LoaderFailure:
    raise
except Exception as exc:
    raise LoaderFailure("load", "model_load_error", str(exc)) from exc
```

Also extend `load_and_merge_adapter(...)` and `hf_from_pretrained(...)` so `revision` is consistently forwarded.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_model_loader.py -k "adapter or revision" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/model_loader.py esd_experiment/tests/test_model_loader.py
git commit -m "feat: harden adapter and revision-aware loading"
```

### Task 5: Replace empty placeholder outputs with temp-file finalization

**Files:**
- Modify: `esd_experiment/src/worker.py`
- Test: `esd_experiment/tests/test_worker.py`

- [ ] **Step 1: Write the failing tests for temp-file behavior**

```python
from pathlib import Path

from esd_experiment.src.worker import finalize_output_path, temp_output_path


def test_temp_output_path_is_hidden_sidecar(tmp_path: Path):
    final_path = tmp_path / "stats" / "org--model.csv"
    temp_path = temp_output_path(final_path)
    assert temp_path.name == ".org--model.csv.tmp"


def test_finalize_output_path_renames_temp_file(tmp_path: Path):
    final_path = tmp_path / "stats" / "org--model.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_output_path(final_path)
    temp_path.write_text("alpha\n1.0\n")
    finalize_output_path(temp_path, final_path)
    assert final_path.exists()
    assert final_path.read_text() == "alpha\n1.0\n"
    assert not temp_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_worker.py -k temp_output -v`
Expected: FAIL because the helpers do not exist and the worker still touches the final CSV directly.

- [ ] **Step 3: Implement temp-file helpers and use them in `worker.py`**

```python
def temp_output_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.tmp")


def finalize_output_path(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(final_path)
```

Then replace:

```python
output_file.touch()
```

with temp-path creation logic, and ensure final rename happens only after `save_results(...)` succeeds.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_worker.py -k temp_output -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/worker.py esd_experiment/tests/test_worker.py
git commit -m "fix: write model outputs atomically"
```

### Task 6: Add structured failure records and stage-aware classification

**Files:**
- Modify: `esd_experiment/src/worker.py`
- Test: `esd_experiment/tests/test_worker.py`

- [ ] **Step 1: Write the failing tests for structured failure recording**

```python
import json
from pathlib import Path

from esd_experiment.src.worker import record_failure


def test_record_failure_writes_jsonl_and_text_summary(tmp_path: Path):
    output_dir = tmp_path / "results"
    record_failure(
        output_dir=output_dir,
        model_id="org/model",
        stage="load",
        reason="repo_gated",
        message="access denied",
        attempt=1,
    )

    jsonl_path = output_dir / "logs" / "failure_records.jsonl"
    text_path = output_dir / "logs" / "failed_models.txt"

    record = json.loads(jsonl_path.read_text().strip())
    assert record["model_id"] == "org/model"
    assert record["stage"] == "load"
    assert record["reason"] == "repo_gated"
    assert "org/model" in text_path.read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_worker.py -k record_failure -v`
Expected: FAIL because `record_failure` only writes a text line today.

- [ ] **Step 3: Implement structured failure recording**

```python
import json
from datetime import datetime, timezone


def record_failure(output_dir: Path, model_id: str, stage: str, reason: str, message: str, attempt: int):
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = logs_dir / "failure_records.jsonl"
    text_path = logs_dir / "failed_models.txt"

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "stage": stage,
        "reason": reason,
        "message": message,
        "attempt": attempt,
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(text_path, "a") as f:
        f.write(f"{model_id}\t{stage}\t{reason}\t{message}\n")
```

Update worker exception handling so loader failures, analysis empties, and save failures call this helper with the right stage/reason.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_worker.py -k record_failure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/worker.py esd_experiment/tests/test_worker.py
git commit -m "feat: record structured worker failures"
```

### Task 7: Narrow retry policy and detect empty analysis results

**Files:**
- Modify: `esd_experiment/src/worker.py`
- Test: `esd_experiment/tests/test_worker.py`

- [ ] **Step 1: Write the failing tests for empty-analysis and retry decisions**

```python
from esd_experiment.src.worker import classify_retryable_failure, validate_metrics_output


def test_validate_metrics_output_rejects_empty_longname_rows():
    metrics = {"longname": [], "alpha": []}
    stage, reason = validate_metrics_output(metrics)
    assert stage == "analyze"
    assert reason == "analysis_empty"


def test_classify_retryable_failure_is_false_for_unsupported_scenarios():
    assert classify_retryable_failure(stage="load", reason="unsupported_loader_scenario") is False


def test_classify_retryable_failure_is_true_for_generic_load_errors():
    assert classify_retryable_failure(stage="load", reason="model_load_error") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest esd_experiment/tests/test_worker.py -k "retryable or validate_metrics" -v`
Expected: FAIL because those helpers do not exist.

- [ ] **Step 3: Implement minimal retry and analysis validation helpers**

```python
def validate_metrics_output(metrics: dict) -> tuple[str, str] | None:
    longnames = metrics.get("longname", []) or []
    if longnames and longnames[-1] is None:
        longnames = longnames[:-1]
    if not longnames:
        return ("analyze", "analysis_empty")
    return None


def classify_retryable_failure(stage: str, reason: str) -> bool:
    if reason in {"unsupported_loader_scenario", "adapter_base_unresolved", "repo_missing_or_private", "repo_gated"}:
        return False
    if reason in {"model_load_error", "cuda_oom", "analysis_exception", "save_error"}:
        return True
    return False
```

Use these in the attempt loop so only likely transient failures retry.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest esd_experiment/tests/test_worker.py -k "retryable or validate_metrics" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add esd_experiment/src/worker.py esd_experiment/tests/test_worker.py
git commit -m "fix: narrow retries and reject empty analysis outputs"
```

### Task 8: Update docs and run targeted verification

**Files:**
- Modify: `README.md`
- Modify: `esd_experiment/README.md`
- Test: `esd_experiment/tests/test_model_loader.py`
- Test: `esd_experiment/tests/test_run_experiment.py`
- Test: `esd_experiment/tests/test_worker.py`

- [ ] **Step 1: Update the docs to show curated-table usage first**

```markdown
## Model Input

The main experiment path now accepts the curated canonical table directly, for example:

```bash
python esd_experiment/run_experiment.py \
    --model_list /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2.csv \
    --output_dir results/ \
    --gpus 0 1 2 3
```

Minimum required column:
- `model_id`

Preferred curated columns:
- `revision_norm`
- `source_model`
- `base_model_relation`
- `loader_scenario`
- `primary_type_bucket`

The old three-column CSV remains supported for simple cases.
```

- [ ] **Step 2: Run focused tests**

Run: `pytest esd_experiment/tests/test_run_experiment.py esd_experiment/tests/test_model_loader.py esd_experiment/tests/test_worker.py -q`
Expected: PASS

- [ ] **Step 3: Run one smoke command against the curated table with a tiny limit**

Run:

```bash
python esd_experiment/run_experiment.py \
  --model_list /scratch/kinshuk/Model-Research/data/curated/phase2_v3/model_zoo_phase2.csv \
  --output_dir /tmp/model_zoo_curated_smoke \
  --gpus 0 \
  --limit 1
```

Expected:
- command starts successfully
- parses curated-table columns without schema errors
- either analyzes one small model successfully or records a structured failure without leaving an empty stats CSV

- [ ] **Step 4: Commit**

```bash
git add README.md esd_experiment/README.md
git commit -m "docs: document curated table experiment input"
```
