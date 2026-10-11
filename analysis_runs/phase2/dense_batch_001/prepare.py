"""One-off first-batch selection and metadata review. Never launch workers."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCE = PROJECT / "data/curated/model_zoo_phase2.csv"

with tempfile.TemporaryDirectory(prefix="model-zoo-batch-metadata-", dir="/tmp") as cache:
    os.environ.update(HF_HOME=cache, CUDA_VISIBLE_DEVICES="", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      HF_HUB_DISABLE_PROGRESS_BARS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN_PATH",
                 "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_XET_CACHE", "TRANSFORMERS_CACHE",
                 "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(name, None)
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    data = pd.read_csv(SOURCE, keep_default_na=False, low_memory=False)
    sizes = pd.to_numeric(data["#Params (B)"], errors="coerce").fillna(
        pd.to_numeric(data["#Params (B)_lb"], errors="coerce"))
    eligible = data.loader_scenario.eq("standard_transformers") & sizes.gt(0) & sizes.le(3)
    for name in ("is_quantized", "is_moe", "is_multimodal", "is_adapter"):
        eligible &= ~data[name].astype(str).str.lower().eq("true")
    pool = data.loc[eligible].copy()
    pool["metadata_params_billion"] = sizes[eligible]
    pool["downloads_numeric"] = pd.to_numeric(pool.downloads, errors="coerce").fillna(0)
    pool = pool.sort_values(["downloads_numeric", "model_id"], ascending=[False, True])
    selected, publishers, parents = [], set(), set()
    for lower, upper in ((0, 1), (1, 3)):
        count = 0
        for row in pool.loc[pool.metadata_params_billion.gt(lower) & pool.metadata_params_billion.le(upper)].to_dict("records"):
            publisher = row["model_id"].split("/")[0]
            parent = str(row["source_model"]).removesuffix(" (Merge)").strip()
            parent = parent if parent and parent != "unknown" else row["model_id"]
            if publisher in publishers or parent in parents:
                continue
            selected.append({"model_id": row["model_id"], "metadata_params_billion": row["metadata_params_billion"],
                             "size_bin": f"{lower}-{upper}B", "metadata_parent": parent,
                             "candidate_source": row["candidate_source"], "primary_type_bucket": row["primary_type_bucket"]})
            publishers.add(publisher)
            parents.add(parent)
            count += 1
            if count == 12:
                break
    pd.DataFrame(selected).to_csv(ROOT / "candidates.csv", index=False)
    with (ROOT / "preparation.log").open("x") as log:
        result = subprocess.run([sys.executable, "esd_experiment/run_experiment.py", "--prepare_only",
                                 "--model_list", str(ROOT / "candidates.csv"), "--output_dir", str(ROOT)],
                                cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    if result.returncode not in (0, 1) or not (ROOT / "models.csv").exists():
        raise RuntimeError("Preparation failed; see preparation.log")
    pinned = pd.read_csv(ROOT / "models.csv", keep_default_na=False)
    reviews = []
    api = HfApi(token=False)
    for row in pinned.to_dict("records"):
        repo, revision = row["model_id"], row["revision_norm"]
        review = {"model_id": repo, "revision": revision, "ready": False}
        try:
            if row["pin_status"] != "pinned":
                raise ValueError(row["pin_error"])
            info = api.model_info(repo, revision=revision, files_metadata=True, timeout=30)
            if info.sha != revision or info.gated or info.private:
                raise ValueError("Requires public ungated pinned access")
            files = {item.rfilename: item.size for item in info.siblings}
            config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision, token=False)).read_text())
            if config.get("quantization_config") or config.get("auto_map") or "adapter_config.json" in files:
                raise ValueError("Quantized, custom-code or adapter checkpoint held for separate coverage work")
            if "model.safetensors.index.json" in files:
                index = json.loads(Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision, token=False)).read_text())
                weights = sorted(set(index["weight_map"].values()))
            else:
                weights = ["model.safetensors"]
            sizes_by_file = {name: files.get(name) for name in weights}
            if not weights or not all(isinstance(size, int) and size > 0 for size in sizes_by_file.values()):
                raise ValueError("No complete safetensors size metadata; review separately")
            total = sum(sizes_by_file.values())
            review.update(weight_bytes=total, weight_files=sizes_by_file, model_type=config.get("model_type"))
            if total > 6.5e9:
                raise ValueError("Weight files exceed this batch's 6.5 GB ceiling")
            review["ready"] = True
        except Exception as error:
            review["hold_reason"] = str(error)
        reviews.append(review)
        print(repo, "ready" if review["ready"] else review["hold_reason"], flush=True)
    ready_ids = {row["model_id"] for row in reviews if row["ready"]}
    # Keep the normal launch manifest usable; exclusions and their pins stay in the review.
    pinned.loc[pinned.model_id.isin(ready_ids)].to_csv(ROOT / "models.csv", index=False)
    report = {"source": str(SOURCE.relative_to(PROJECT)), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "target_rows": len(data), "rows_with_numeric_size": int(sizes.notna().sum()),
              "eligible_metadata_rows": len(pool), "selected": len(selected), "ready": len(ready_ids),
              "loader_counts": data.loader_scenario.value_counts().to_dict(),
              "models": reviews, "weights_downloaded": False, "workers_launched": False,
              "selection": "Up to 12 per size bin, downloads then model_id; at most one publisher and metadata parent. Operational slice, not representative sampling.",
              "limits": "One worker; authorized idle GPUs 3-7; 12 GiB cache, 16 GiB sampled RSS, stages 900/900/120 seconds, 1 hour overall stop. No automatic scale-up."}
    (ROOT / "batch_review.json").write_text(json.dumps(report, indent=2))
print("Prepared only; temporary metadata cache removed.")
