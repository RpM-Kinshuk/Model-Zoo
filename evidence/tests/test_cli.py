import json

import pandas as pd
import pytest

from evidence.cli import main
from conftest import run_cli, run_cli_code


def test_diff_reports_facts_and_uncalibrated(child_dir, parent_dir):
    out = run_cli(["diff", str(child_dir), str(parent_dir), "--json"])
    assert out["relatedness"]["verdict"] == "uncalibrated"
    assert out["summary"]["counts"]["reshaped"] == 1 and out["summary"]["counts"]["changed"] > 0
    assert "predicted" not in json.dumps(out)              # no capability predictions


def test_check_exit_codes(child_dir, unrelated_dir, null_file):
    assert run_cli_code(["check", str(child_dir)]) == 0                         # uncalibrated
    assert run_cli_code(["check", str(unrelated_dir), "--null", str(null_file)]) == 1


def test_resume_skips_done_and_overwrites_tmp(run_dir, models_csv):
    args = ["batch", "--models", str(models_csv), "--manifests", str(run_dir / "manifests"),
            "--measurement_mode", "row_blocks", "--out", str(run_dir / "row_blocks"), "--workers", "2"]
    main(args); done = sorted((run_dir / "row_blocks").glob("*.safetensors"))
    mtimes = [p.stat().st_mtime_ns for p in done]
    (run_dir / "row_blocks" / "extra.row_blocks.fingerprint.safetensors.tmp").write_bytes(b"partial")
    main(args)
    assert [p.stat().st_mtime_ns for p in done] == mtimes
    assert not (run_dir / "row_blocks" / "extra.row_blocks.fingerprint.safetensors.tmp").exists()


def test_no_access_recorded_once(run_dir, models_csv_with_denied, denying_reader, fake_api):
    args = ["manifest", "--models", str(models_csv_with_denied), "--out", str(run_dir / "manifests")]
    main(args, reader_factory=lambda: denying_reader, api=fake_api)
    main(args, reader_factory=lambda: denying_reader, api=fake_api)
    assert (pd.read_csv(run_dir / "manifests" / "failures.csv").reason == "no_access").sum() == 1


def test_tokens_never_written(run_dir, models_csv, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_marker")
    main(["manifest", "--models", str(models_csv), "--out", str(run_dir / "manifests")])
    for p in run_dir.rglob("*"):
        if p.is_file():
            assert b"hf_secret_marker" not in p.read_bytes()


def test_similarity_alignment_and_calibration_guards(child_dir, parent_dir, null_file, tmp_path):
    import copy
    import numpy as np
    from evidence.compare import diff_fingerprints, pair_similarity
    from evidence.fingerprint import fingerprint_model, load_fingerprint
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest

    reader = HubReader()
    parent = build_manifest(str(parent_dir), "main", reader, None)
    child = build_manifest(str(child_dir), "main", reader, None)
    a = load_fingerprint(fingerprint_model(child, reader, "row_blocks", tmp_path / "fp", parent=parent))
    b = load_fingerprint(fingerprint_model(parent, reader, "row_blocks", tmp_path / "fp"))
    null = json.loads(null_file.read_text())
    diff = diff_fingerprints(a, b, null)
    assert diff["relatedness"]["verdict"] == "related"
    assert diff["relatedness"]["p_value"] == 1 / (null["n_pairs"] + 1)
    assert pair_similarity(a, b)["n_shared"] == 2  # includes the fixed embedding block
    assert any("rank_confidence" in v for v in diff["views"].values())
    other = copy.deepcopy(b)
    other.meta["repo_id"] = "lab/different-parent"
    assert not any("delta_status" in v for v in diff_fingerprints(a, other)["views"].values())
    for field, value in (("tier", "full_weights"), ("row_block", 128), ("sketch_seed", 1), ("sketch_k", 32)):
        other = copy.deepcopy(b)
        other.meta[field] = value
        guarded = diff_fingerprints(a, other, null)["relatedness"]
        assert guarded["verdict"] == "uncalibrated" and guarded["reason"]
        assert "p_value" not in guarded
    for field in ("rows", "shape", "geometry"):
        other = copy.deepcopy(b)
        if field == "rows":
            other.rows[0] += 1
        elif field == "shape":
            other.shape[0, 1] += 1
        else:
            other.geometry[0] = "head"
        assert pair_similarity(a, other)["n_shared"] == 1
        assert diff_fingerprints(a, other, null)["relatedness"]["verdict"] == "uncalibrated"
    tied = dict(null, null_quantiles=[1., 1., 1.], n_pairs=999, related_threshold=1.)
    identical = diff_fingerprints(b, b, tied)["relatedness"]
    assert identical["p_value"] == 1.0
    compressed = dict(null, null_quantiles=[-.1, .1], n_pairs=999)
    assert diff_fingerprints(a, b, compressed)["relatedness"]["p_value"] == .001
    for invalid in (dict(null, statistic="other"), dict(null, n_pairs=0),
                    dict(null, null_quantiles=[.1, -.1]), dict(null, null_quantiles=[np.nan])):
        guarded = diff_fingerprints(a, b, invalid)["relatedness"]
        assert guarded["verdict"] == "uncalibrated" and "p_value" not in guarded


def test_batch_outputs_workers_and_settings(run_dir, models_csv, monkeypatch):
    from evidence.fingerprint import load_fingerprint
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    args = ["batch", "--models", str(models_csv), "--manifests", str(run_dir / "manifests"),
            "--measurement_mode", "row_blocks", "--out", str(run_dir / "row_blocks"), "--workers", "2", "--sketch_k", "32"]
    assert main(args) == 0
    paths = sorted((run_dir / "row_blocks").glob("*.safetensors"))
    assert len(paths) == 2
    assert all(load_fingerprint(p).meta["sketch_k"] == 32 for p in paths)
    args[-1] = "64"
    assert main(args) == 0
    assert all(load_fingerprint(p).meta["sketch_k"] == 64 for p in paths)
    summary = json.loads((run_dir / "row_blocks" / "summary.json").read_text())
    assert len(summary["models"]) == 2
    for record in summary["models"]:
        assert record["bytes_read"] > 0 and record["range_reads"] > 0
        assert record["stratum"] == "text:<1B" and record["view_counts"]["matrix"] == 2


def test_full_weights_resume_requires_companions_and_parent_cache(run_dir, models_csv):
    from evidence.fingerprint import load_fingerprint
    args = ["batch", "--models", str(models_csv), "--manifests", str(run_dir / "manifests"),
            "--measurement_mode", "full_weights", "--out", str(run_dir / "full_weights"), "--also_row_block_sizes", "128,256",
            "--with_parent", "--parent_cache", str(run_dir / "cache")]
    assert main(args) == 0
    paths = sorted((run_dir / "full_weights").glob("*.full_weights.fingerprint.safetensors"))
    assert len(paths) == 2 and list((run_dir / "cache").glob("*.bin"))
    child = next(p for p in paths if load_fingerprint(p).meta["parent"])
    assert load_fingerprint(child).meta["parent"]["repo_id"] == "lab/parent"
    companion = child.with_name(child.name.replace(".full_weights.", ".row_blocks_128."))
    companion.unlink()
    assert main(args) == 0 and companion.is_file()
    spectra = child.with_name(child.name.replace("fingerprint.safetensors", "spectra.h5"))
    spectra.unlink()
    assert main(args) == 0 and spectra.is_file()


def test_manifest_pins_bases_and_local_reads_are_offline(child_dir, tmp_path, monkeypatch):
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest
    def reject(*args, **kwargs):
        raise AssertionError("network call in local CLI")
    monkeypatch.setattr(HubReader, "_request", reject)
    path = tmp_path / "selected.csv"
    pd.DataFrame([{"repo_id": "lab/child", "revision": ""}]).to_csv(path, index=False)
    out = tmp_path / "selected"
    args = ["manifest", "--models", str(path), "--out", str(out), "--include_declared_bases"]
    assert main(args) == 0
    manifests = [json.loads(p.read_text()) for p in out.glob("*.manifest.json")]
    assert {m["repo_id"] for m in manifests} == {"lab/parent", "lab/child"}
    assert all(m["revision"].startswith("local:") for m in manifests)
    assert not any(str(tmp_path) in p.read_text() for p in out.iterdir())
    child = build_manifest(str(child_dir), "main", HubReader(), None)
    pd.DataFrame([{"repo_id": "lab/child", "revision": child["revision"]}]).to_csv(path, index=False)
    assert main(args) == 0
    assert run_cli(["diff", str(child_dir), "--json"])["b"]["repo_id"] == "lab/parent"


def test_batch_records_real_failures_and_finishes_other_models(run_dir, models_csv, tmp_path):
    models = pd.read_csv(models_csv).fillna("")
    models.loc[len(models)] = {"repo_id": "lab/missing", "revision": "", "stratum": "text:<1B", "modality": "text"}
    path = tmp_path / "broken.csv"
    models.to_csv(path, index=False)
    args = ["batch", "--models", str(path), "--manifests", str(run_dir / "manifests"),
            "--measurement_mode", "row_blocks", "--out", str(run_dir / "broken")]
    assert main(args) == 2
    failures = pd.read_csv(run_dir / "broken" / "failures.csv")
    assert list(failures.repo_id) == ["lab/missing"]
    assert "FileNotFoundError" in failures.iloc[0].message
    assert len(list((run_dir / "broken").glob("*.safetensors"))) == 2


def test_frame_sample_and_fingerprint_commands(hub_stats_sample, parent_dir, tmp_path):
    stats, frame, models = (tmp_path / name for name in ("stats.parquet", "frame.csv", "sample.csv"))
    hub_stats_sample.to_parquet(stats, index=False)
    assert run_cli(["frame", "--hub_stats", str(stats), "--out", str(frame)])["n_models"] > 0
    assert run_cli(["sample", "--frame", str(frame), "--plan", "pilot", "--out", str(models)])["n_models"] > 0
    assert "revision" in pd.read_csv(models)
    out = run_cli(["fingerprint", str(parent_dir), "--out", str(tmp_path / "fp"), "--json"])
    assert out["n_views"] == 2 and out["revision"].startswith("local:")
    assert run_cli_code(["diff", "lab/missing"]) == 2


def test_unnamed_local_checkpoint_fallback_with_explicit_out(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from evidence.fingerprint import load_fingerprint
    from evidence.hub import HubReader

    checkpoint = tmp_path / "external-checkpoint"
    checkpoint.mkdir()
    save_file({"linear.weight": torch.arange(512).reshape(32, 16).float()}, checkpoint / "model.safetensors")
    def reject(*args, **kwargs):
        raise AssertionError("local checkpoint attempted network access")
    monkeypatch.setattr(HubReader, "_request", reject)
    for label, data_root in (("outside", str(tmp_path / "data")), ("unset", None)):
        if data_root:
            monkeypatch.setenv("ZOO_DATA", data_root)
        else:
            monkeypatch.delenv("ZOO_DATA", raising=False)
        output = tmp_path / label
        assert main(["fingerprint", str(checkpoint), "--out", str(output)]) == 0
        fp = load_fingerprint(next(output.glob("*.fingerprint.safetensors")))
        assert fp.meta["repo_id"] == str(checkpoint)
        assert fp.meta["revision"].startswith("local:")


def test_manifest_retry_denied_rebuilds_selected_manifest(run_dir, models_csv_with_denied,
                                                          denying_reader, fake_reader, fake_api):
    args = ["manifest", "--models", str(models_csv_with_denied), "--out", str(run_dir / "retry-manifests"), "--workers", "1"]
    assert main(args, reader_factory=lambda: denying_reader, api=fake_api) == 2
    assert main(args + ["--retry_denied"], reader_factory=lambda: fake_reader, api=fake_api) == 0
    manifest = json.loads(next((run_dir / "retry-manifests").glob("*.manifest.json")).read_text())
    assert manifest["access"] == "granted" and fake_reader.stats["resolves"] > 0


def test_batch_denied_retry_keeps_resolved_pin(run_dir, tmp_path, denying_reader, fake_reader, fake_api):
    import copy
    from conftest import SHA
    from evidence.manifest import build_manifest, write_manifest

    directory = run_dir / "pinned-denied"
    denied = build_manifest("org/gated", SHA, denying_reader, fake_api)
    assert denied["access"] == "no_access"
    write_manifest(denied, directory)
    models = tmp_path / "unpinned.csv"
    pd.DataFrame([{"repo_id": "org/gated", "revision": ""}]).to_csv(models, index=False)
    calls = []
    class MovingApi:
        def model_info(self, repo_id, revision="main", **kwargs):
            calls.append(revision)
            info = copy.copy(fake_api.model_info(repo_id, revision="main" if revision == "main" else revision))
            if revision == "main":
                info.sha = "b" * 40
            return info
    args = ["batch", "--models", str(models), "--manifests", str(directory), "--measurement_mode", "row_blocks",
            "--out", str(run_dir / "pinned-output"), "--retry_denied", "--workers", "1"]
    assert main(args, reader_factory=lambda: fake_reader, api=MovingApi()) == 0
    assert calls == [SHA]
    manifest = json.loads(next(directory.glob("*.manifest.json")).read_text())
    assert manifest["revision"] == SHA and manifest["access"] == "granted"


def test_batch_extraction_denial_is_skipped_until_retry(run_dir, models_csv):
    from evidence.hub import AccessDenied, HubReader

    class LaterDeniedReader(HubReader):
        attempts = 0
        def resolve(self, *args, **kwargs):
            self.attempts += 1
            raise AccessDenied("Access revoked after metadata collection")
    reader = LaterDeniedReader()
    args = ["batch", "--models", str(models_csv), "--manifests", str(run_dir / "manifests"),
            "--measurement_mode", "row_blocks", "--out", str(run_dir / "revoked"), "--workers", "1"]
    assert main(args, reader_factory=lambda: reader) == 2
    attempts = reader.attempts
    assert attempts == 2
    assert main(args, reader_factory=lambda: reader) == 2
    assert reader.attempts == attempts
    failures = pd.read_csv(run_dir / "revoked" / "failures.csv")
    assert len(failures) == 2 and set(failures.reason) == {"no_access"}
    assert main(args + ["--retry_denied"]) == 0
    assert len(list((run_dir / "revoked").glob("*.fingerprint.safetensors"))) == 2


def test_include_bases_finalizes_reports_after_selected_failure(child_dir, tmp_path):
    models = tmp_path / "selected-with-missing.csv"
    pd.DataFrame([{"repo_id": "lab/child", "revision": ""},
                  {"repo_id": "lab/missing", "revision": ""}]).to_csv(models, index=False)
    output = tmp_path / "bases-with-failure"
    args = ["manifest", "--models", str(models), "--out", str(output), "--include_declared_bases"]
    assert main(args) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert {r["repo_id"] for r in summary["models"]} == {"lab/child", "lab/parent", "lab/missing"}
    failures = pd.read_csv(output / "failures.csv")
    assert list(failures.repo_id) == ["lab/missing"]
    assert len(list(output.glob("*.manifest.json"))) == 2


def test_declared_base_denied_retry_keeps_resolved_pin(tmp_path, denying_reader, fake_reader, fake_api):
    import copy
    from conftest import SHA
    from evidence.manifest import build_manifest, write_manifest

    output = tmp_path / "base-retry"
    child = build_manifest("org/m", SHA, fake_reader, fake_api)
    child["declared_bases"] = ["org/gated"]
    write_manifest(child, output)
    write_manifest(build_manifest("org/gated", SHA, denying_reader, fake_api), output)
    models = tmp_path / "child.csv"
    pd.DataFrame([{"repo_id": "org/m", "revision": SHA}]).to_csv(models, index=False)
    calls = []

    class MovingApi:
        def model_info(self, repo_id, revision="main", **kwargs):
            calls.append(revision)
            info = copy.copy(fake_api.model_info(repo_id, revision=revision))
            if revision == "main":
                info.sha = "b" * 40
            return info

    args = ["manifest", "--models", str(models), "--out", str(output),
            "--include_declared_bases", "--retry_denied"]
    assert main(args, reader_factory=lambda: fake_reader, api=MovingApi()) == 0
    assert calls == [SHA]
    bases = [json.loads(p.read_text()) for p in output.glob("*.manifest.json")]
    assert [(m["revision"], m["access"]) for m in bases if m["repo_id"] == "org/gated"] == [(SHA, "granted")]


@pytest.mark.parametrize("command", ["fingerprint", "batch"])
def test_cli_forwards_spectrum_compute_device(command, tiny_checkpoint, tmp_path, monkeypatch):
    import evidence.cli as module
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, write_manifest

    original = module.fingerprint_model
    devices = []
    source = tmp_path / "device-source"
    source.mkdir()
    for path in tiny_checkpoint.iterdir():
        if path.is_file():
            (source / path.name).write_bytes(path.read_bytes())

    def cpu_fixture_run(*args, **kwargs):
        devices.append(kwargs.pop("compute_device"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fingerprint_model", cpu_fixture_run)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    args = [command, "--measurement_mode", "full_weights", "--out", str(tmp_path / "device-out"),
            "--compute_device", "cuda:1"]
    if command == "fingerprint":
        args.append(str(source))
    else:
        directory = tmp_path / "device-manifests"
        manifest = build_manifest(str(source), "main", HubReader(), None)
        write_manifest(manifest, directory)
        models = tmp_path / "device-models.csv"
        pd.DataFrame([{"repo_id": str(source), "revision": manifest["revision"]}]).to_csv(models, index=False)
        args += ["--models", str(models), "--manifests", str(directory), "--workers", "1"]
    assert main(args) == 0
    assert devices and set(devices) == {"cuda:1"}


def test_batch_defaults_match_target(tmp_path, monkeypatch):
    import evidence.cli as cli
    monkeypatch.setenv("ZOO_DATA", str(tmp_path / "data"))
    captured = []
    monkeypatch.setattr(cli, "_run_many", lambda args, *unused: captured.append(args) or 0)
    common = ["batch", "--models", "models.csv", "--manifests", "manifests", "--out", str(tmp_path)]
    assert main(common + ["--measurement_mode", "full_weights"]) == 0
    args = captured[-1]
    assert args.with_parent and args.delta_blocks == (256,)
    assert args.parent_cache == tmp_path / "data/cache/parent-blocks"
    assert args.parent_cache_max_gib == 64
    assert args.also_row_block_sizes == (128, 256, 512)
    assert main(common + ["--measurement_mode", "row_blocks"]) == 0
    assert captured[-1].also_row_block_sizes == ()


@pytest.mark.parametrize("stratum,k,experts", [("lab", 128, True), ("anchor", 128, True), ("census", 64, False), ("", 64, False)])
def test_batch_auto_sketch_k_and_experts_follow_stratum(local_manifest, tmp_path, stratum, k, experts):
    from evidence.cli import _batch_options, _parser
    args = _parser().parse_args(["batch", "--models", "unused", "--manifests", "unused",
                                "--measurement_mode", "full_weights", "--out", str(tmp_path)])
    options = _batch_options(args, dict(repo_id=local_manifest["repo_id"], stratum=stratum), set())
    assert options["sketch_k"] == k
    assert options["all_experts"] is experts
    assert options["cache_self"] is experts


def test_opt_outs_reproduce_previous_batch_outputs(local_manifest, tmp_path, monkeypatch):
    import evidence.cli as cli
    import evidence.fingerprint as fp
    from safetensors import safe_open
    from evidence.hub import HubReader
    from evidence.manifest import write_manifest
    monkeypatch.setattr(fp.time, "monotonic", lambda: 0.)
    manifests = tmp_path / "manifests"
    write_manifest(local_manifest, manifests)
    models = tmp_path / "models.csv"
    models.write_text("repo_id,revision\n" + local_manifest["repo_id"] + "," + local_manifest["revision"] + "\n")
    baseline = fp.fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path / "old")
    assert main(["batch", "--models", str(models), "--manifests", str(manifests),
                 "--measurement_mode", "full_weights", "--out", str(tmp_path / "new"), "--no-with_parent",
                 "--no-parent_cache", "--sketch_k", "64", "--also_row_block_sizes", "none", "--workers", "1"]) == 0
    result = next((tmp_path / "new").glob("*.safetensors"))
    # Safetensors header map ordering is not stable; payload bytes and metadata are.
    with safe_open(baseline, framework="numpy") as a, safe_open(result, framework="numpy") as b:
        assert a.metadata() == b.metadata()
        assert a.keys() == b.keys()
        for key in a.keys():
            assert a.get_tensor(key).tobytes() == b.get_tensor(key).tobytes()
    assert len(list((tmp_path / "new").glob("*.safetensors"))) == 1


def test_workers_auto_spreads_over_assigned_gpus(monkeypatch):
    import torch
    from evidence.cli import _execution_policy, _init_worker
    import evidence.cli as cli
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _execution_policy("auto", "auto") == (["cuda:0", "cuda:1"], 2)
    devices, workers = _execution_policy("auto", "4")
    assert workers == 4
    assigned = []
    monkeypatch.setattr(cli, "_DEVICE", None)
    import multiprocessing
    counter = multiprocessing.Value("i", 0)
    for _ in range(workers):
        _init_worker(lambda: object(), None, devices, counter)
        assigned.append(cli._DEVICE)
    assert assigned == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]



def test_workers_auto_on_cpu(monkeypatch):
    from evidence.cli import _execution_policy
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 96)
    assert _execution_policy("auto", "auto") == (["cpu"], 8)
    monkeypatch.setattr("os.cpu_count", lambda: 4)
    assert _execution_policy("auto", "auto") == (["cpu"], 1)


def test_batch_resume_respects_delta_policy(local_manifest, tmp_path):
    from evidence.manifest import write_manifest
    from evidence.fingerprint import load_fingerprint
    manifest = dict(local_manifest, declared_bases=[local_manifest["repo_id"]])
    write_manifest(manifest, tmp_path / "manifests")
    models = tmp_path / "models.csv"
    models.write_text("repo_id,revision\n" + manifest["repo_id"] + "," + manifest["revision"] + "\n")
    args = ["batch", "--models", str(models), "--manifests", str(tmp_path / "manifests"),
            "--measurement_mode", "full_weights", "--out", str(tmp_path / "out"), "--workers", "1"]
    assert main(args) == 0
    path = next((tmp_path / "out").glob("*.full_weights.fingerprint.safetensors"))
    assert set(load_fingerprint(path).delta["status"]) == {"not_computed"}
    assert main(args + ["--full_delta"]) == 0
    assert set(load_fingerprint(path).delta["status"]) == {"identical"}
    assert main(args) == 0
    assert set(load_fingerprint(path).delta["status"]) == {"not_computed"}


def test_batch_accepts_existing_maximum_sketch_override(monkeypatch, tmp_path):
    import evidence.cli as cli
    captured = []
    monkeypatch.setattr(cli, "_run_many", lambda args, *unused: captured.append(args) or 0)
    assert main(["batch", "--models", "unused", "--manifests", "unused", "--measurement_mode", "row_blocks",
                 "--out", str(tmp_path), "--sketch_k", "256"]) == 0
    assert captured[0].sketch_k == "256"
