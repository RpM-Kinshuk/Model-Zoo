from types import SimpleNamespace
import threading
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "esd_experiment", ROOT / "esd_experiment/src"):
    sys.path.insert(0, str(path))

import pytest


def test_job_resources_auto_from_weight_bytes():
    from sizing import job_resources
    gib = 1024**3
    resource = job_resources(20*gib, 24*gib, 2, 2*gib)
    assert resource["num_gpus"] == 2
    assert resource["ram_limit_gib"] == 16
    assert resource["cache_limit_gib"] == 24
    assert resource["stage_timeout_seconds"]["fingerprint"] == 28800
    assert job_resources(40*gib, 24*gib, 2, 2*gib)["oversized"]


def test_prefetch_admission_fallback_and_owned_cache(tmp_path):
    from prefetch import Prefetcher
    from gputracker.supervision import WorkerJob, WorkerStateTracker
    from dataclasses import replace
    tracker = WorkerStateTracker(tmp_path / "logs", run_id="test", cache_root=tmp_path / "cache")
    jobs = [WorkerJob("true", worker_id=str(i), prefetch_files=(("org/model", "a"*40, "model.safetensors", 10),)) for i in range(3)]
    started = []
    def download(job, cache, stop):
        started.append(job.worker_id)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "prepared").write_text("ok")
        if job.worker_id == "1":
            raise OSError("offline")
    pf = Prefetcher(jobs, tracker, max_models=1, max_bytes=1024**3, downloader=download)
    try:
        prepared = pf.claim(jobs[0], lambda: None)
        assert started == ["0"] and Path(prepared.cache_path).is_dir()
        record = tracker.start_worker(prepared, [0])
        assert record.cache_path == Path(prepared.cache_path)
        pf.launched(prepared)
        fallback = pf.claim(jobs[1], lambda: None)
        assert not fallback.cache_path
        pf.launched(fallback)
        tracker.finish_worker("0", returncode=0)
        assert not record.cache_path.exists()
    finally:
        pf.close()
        tracker.close()
    assert not tracker._cache_path_for("2").exists()


def test_prefetch_zero_and_oversized_budget_never_download(tmp_path):
    from prefetch import Prefetcher
    from gputracker.supervision import WorkerJob, WorkerStateTracker
    tracker = WorkerStateTracker(tmp_path / "logs", run_id="test", cache_root=tmp_path / "cache")
    job = WorkerJob("true", worker_id="job", prefetch_files=(("org/m", "a"*40, "model.safetensors", 10),))
    for count, budget in [(0, 100000000), (1, 1)]:
        pf = Prefetcher([job], tracker, max_models=count, max_bytes=budget, downloader=lambda *args: pytest.fail("unexpected download"))
        assert pf.claim(job, lambda: None) == job
        pf.launched(job)
        pf.close()
    tracker.close()


def test_worker_cache_cannot_adopt_arbitrary_path(tmp_path):
    from gputracker.supervision import WorkerJob, WorkerStateTracker
    tracker = WorkerStateTracker(tmp_path / "logs", run_id="test", cache_root=tmp_path / "cache")
    try:
        with pytest.raises(ValueError, match="cache"):
            tracker.start_worker(WorkerJob("true", worker_id="bad", cache_path=str(tmp_path)), [0])
    finally:
        tracker.close()


def test_sizing_defaults_explicit_controls_and_checkpoint(tmp_path, monkeypatch):
    import sys
    import json
    import pandas as pd
    import gpustat
    import run_experiment as runner
    monkeypatch.setattr(sys, "argv", ["runner", "--model_list", "models.csv", "--output_dir", str(tmp_path), "--gpus", "0", "1"])
    args = runner.parse_args()
    assert args.num_gpus_per_job == args.max_weight_gib == args.max_worker_ram_gib == args.stage_timeout_seconds == "auto"
    assert args.prefetch_models == 2 and args.worker_cache_root.endswith("/hf_worker_cache")
    monkeypatch.setattr(gpustat.GPUStatCollection, "new_query", lambda: SimpleNamespace(gpus=[{"memory.total": 24*1024}]*2))
    rows = pd.DataFrame([dict(model_id="org/a", revision_norm="a"*40, source_model="", base_model_relation="", download_bytes=20*1024**3),
                         dict(model_id="org/b", revision_norm="b"*40, source_model="", base_model_relation="", download_bytes=40*1024**3)])
    ready, large = runner.size_models(rows, args)
    assert ready.iloc[0]._job_resources["num_gpus"] == 2
    assert large.iloc[0].reason == "too_large_for_full_mode"
    args.num_gpus_per_job, args.max_worker_ram_gib, args.max_worker_cache_gib = 2, 33, 44
    args.stage_timeout_seconds = dict(load=99)
    ready, _ = runner.size_models(rows.iloc[:1], args)
    job = runner.generate_worker_jobs(ready, tmp_path, args)[0]
    assert job.num_gpus == 2 and job.ram_limit_gib == job.cache_limit_gib == 0
    runner.create_runtime_config(args, tmp_path / "config.json")
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["max_worker_ram_gib"] == 33 and cfg["max_worker_cache_gib"] == 44
    assert cfg["stage_timeout_seconds"] == dict(load=99)
    args.analysis_source, args.num_gpus_per_job = "checkpoint", "auto"
    ready, large = runner.size_models(rows, args)
    assert large.empty and all(row["num_gpus"] == 1 for row in ready._job_resources)


def test_oversized_models_get_real_t1_package(tmp_path, monkeypatch):
    import sys
    import json
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from evidence.manifest import build_manifest
    from evidence.hub import HubReader
    from evidence.sidecar import EvidenceSidecar
    import evidence_bridge
    import run_experiment as runner
    source = tmp_path / "source"
    source.mkdir()
    save_file({"linear.weight": torch.randn(512, 64)}, source / "model.safetensors")
    manifest = build_manifest(str(source), "main", HubReader(), None)
    monkeypatch.setenv("ZOO_DATA", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["runner", "--model_list", "models.csv", "--output_dir", str(tmp_path)])
    args = runner.parse_args()
    args.oversized_device = "cpu"
    args.evidence_device = "gpu"  # T1 device must override the full-mode sidecar setting.
    folder = tmp_path / "evidence" / runner.safe_filename(manifest["repo_id"] + "@" + manifest["revision"])
    from evidence.manifest import write_manifest
    write_manifest(manifest, folder)
    # Exercise the real runner-to-sidecar arguments and parent-bearing resume.
    parent = dict(manifest, repo_id="lab/parent", source=dict(zoo_data_path="source"))
    parent_folder = tmp_path / "evidence" / runner.safe_filename(parent["repo_id"] + "@" + parent["revision"])
    write_manifest(parent, parent_folder)
    models = pd.DataFrame([dict(model_id=manifest["repo_id"], revision_norm=manifest["revision"], reason="too_large_for_full_mode", source_model=parent["repo_id"] + "@" + parent["revision"])])
    runner.extract_row_block_evidence(models, tmp_path, args, logging.getLogger(__name__))
    result = pd.read_csv(tmp_path / "oversized_models.csv")
    assert result.t1_status.tolist() == ["published"]
    packages = list((tmp_path / "packages").rglob("package.json"))
    assert len(packages) == 1 and json.loads(packages[0].read_text())["kind"] == "t1_only"
    assert json.loads(packages[0].read_text())["conventions"]["evidence"]["device"] == "cpu"
    # Resume uses the compatible block fingerprint; it does not reread weights.
    monkeypatch.setattr("evidence.fingerprint.fingerprint_model", lambda *args, **kw: pytest.fail("repeated T1 measurement"))
    runner.extract_row_block_evidence(models, tmp_path, args, logging.getLogger(__name__))
    assert pd.read_csv(tmp_path / "oversized_models.csv").t1_status.tolist() == ["published"]


def test_prefetch_shutdown_before_tracker_registration_cleans_cache(tmp_path):
    from prefetch import Prefetcher
    from gputracker.supervision import WorkerJob, WorkerStateTracker
    tracker = WorkerStateTracker(tmp_path / "logs", run_id="test", cache_root=tmp_path / "cache")
    job = WorkerJob("true", worker_id="pending", prefetch_files=(("org/m", "a"*40, "model.safetensors", 10),))
    def download(job, cache, stop):
        cache.mkdir(parents=True)
        (cache / "blob").write_bytes(b"ok")
    pf = Prefetcher([job], tracker, max_models=1, max_bytes=1024**3, downloader=download)
    ready = pf.claim(job, lambda: None)
    pf.launched(ready)
    pf.close()
    assert not Path(ready.cache_path).exists()
    tracker.close()


def test_prefetch_worker_ids_cannot_escape_run_directory(tmp_path):
    from gputracker.supervision import WorkerStateTracker
    tracker = WorkerStateTracker(tmp_path / "logs", run_id="test", cache_root=tmp_path / "cache")
    try:
        for name in ("../outside", str(tmp_path / "outside")):
            assert tracker._cache_path_for(name).resolve().is_relative_to(tmp_path / "cache/test")
    finally:
        tracker.close()


def test_oversized_stop_does_not_start_evidence(tmp_path, monkeypatch):
    import pandas as pd
    import run_experiment as runner
    import evidence_bridge
    monkeypatch.setattr(evidence_bridge, "create_sidecar", lambda *args: pytest.fail("started after stop"))
    runner.run_oversized_evidence_jobs(pd.DataFrame([dict(model_id="org/a")]), tmp_path, SimpleNamespace(),
                             logging.getLogger(__name__), stopped=lambda: True)
    assert pd.read_csv(tmp_path / "oversized_models.csv").t1_status.tolist() == ["interrupted"]


def test_per_job_limits_override_live_globals(tmp_path, monkeypatch):
    import psutil
    from gputracker.gputracker import ChildThread, WorkerJob
    dispatcher = SimpleNamespace(lock=threading.Lock(), config=dict(max_worker_ram_gib=1, max_worker_cache_gib=0, stage_timeout_seconds={"load": 10}))
    job = WorkerJob("true", ram_limit_gib=2, stage_timeout_seconds={"load": 99})
    worker = ChildThread("test", 1, [0], job, logging.getLogger(__name__), dispatcher)
    process = SimpleNamespace(children=lambda **kw: [], memory_info=lambda: SimpleNamespace(rss=int(1.5*1024**3)))
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    assert worker._resource_limit_reason(SimpleNamespace(pid=42), None) == (None, None)
    assert worker._current_stale_config()[2] == {"load": 99}


def test_prefetch_load_timeout_zero_preserves_disabled_limit(tmp_path, monkeypatch):
    import prefetch
    from gputracker.supervision import WorkerJob
    calls = []
    class Process:
        pid = 42
        returncode = 0
        def poll(self):
            calls.append(1)
            return 0 if len(calls) > 1 else None
    monkeypatch.setattr(prefetch.subprocess, "Popen", lambda *args, **kw: Process())
    job = WorkerJob("true", prefetch_files=(("org/m", "a"*40, "model.safetensors", 10),), stage_timeout_seconds={"load": 0})
    prefetch._download(job, tmp_path / "cache", threading.Event())


def test_oversized_jobs_use_supervision_without_prefetch(tmp_path, monkeypatch):
    import pandas as pd
    import run_experiment as runner
    calls = []
    class Dispatch:
        def __init__(self, **kw):
            calls.append(kw)
            for job in kw['bash_command_list']:
                assert job.cpu_only and job.ram_limit_gib >= 16
                assert 'row_block_worker.py' in job.command and not job.prefetch_files
        def start(self):
            pass
        def wait_for_completion(self):
            for job in calls[0]['bash_command_list']:
                Path(job.terminal_status_path).write_text('{"status":"success","t1_status":"published"}')
    monkeypatch.setattr(runner, 'DispatchThread', Dispatch)
    args = SimpleNamespace(oversized_t1=True, evidence=True, oversized_device='cpu', max_concurrent_jobs=2,
                           worker_cache_root=str(tmp_path/'cache'), max_worker_ram_gib='auto', max_worker_cache_gib='auto', stage_timeout_seconds='auto')
    dispatcher = SimpleNamespace(error=None, shutdown_event=threading.Event(), drain_event=threading.Event())
    rows = pd.DataFrame([dict(model_id='org/a', revision_norm='a'*40), dict(model_id='org/b', revision_norm='b'*40)])
    runner.run_oversized_evidence_jobs(rows, tmp_path, args, logging.getLogger(__name__), dispatcher=dispatcher)
    assert len(calls[0]['bash_command_list']) == 2
    assert calls[0]['prefetch_models'] == 0
    assert pd.read_csv(tmp_path/'oversized_models.csv').t1_status.tolist() == ['published','published']


def test_oversized_cpu_workers_publish_through_real_supervisor(tmp_path, monkeypatch):
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, write_manifest
    import run_experiment as runner
    from gputracker.gputracker import GPUDispatcher
    monkeypatch.setattr(GPUDispatcher, '_instance', None)
    monkeypatch.setenv('ZOO_DATA', str(tmp_path))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(sys, 'argv', ['runner','--model_list','unused.csv','--output_dir',str(tmp_path),
                                    '--oversized_device','cpu','--max_concurrent_jobs','2','--cpu_threads_per_gpu','1',
                                    '--worker_cache_root',str(tmp_path/'cache')])
    args = runner.parse_args()
    rows = []
    for index in range(2):
        source = tmp_path / f'source{index}'
        source.mkdir()
        save_file({'linear.weight': torch.randn(32, 16)}, source/'model.safetensors')
        manifest = build_manifest(str(source), 'main', HubReader(), None)
        folder = tmp_path/'evidence'/runner.safe_filename(manifest['repo_id']+'@'+manifest['revision'])
        write_manifest(manifest, folder)
        rows.append(dict(model_id=manifest['repo_id'],revision_norm=manifest['revision']))
    runner.run_oversized_evidence_jobs(pd.DataFrame(rows), tmp_path, args, logging.getLogger(__name__))
    result = pd.read_csv(tmp_path/'oversized_models.csv')
    assert result.t1_status.tolist() == ['published','published']
    assert len(list((tmp_path/'packages').rglob('package.json'))) == 2
    assert not list((tmp_path/'cache').rglob('hub'))  # private caches removed after process exit


def test_cpu_only_queue_remains_bounded_with_gpu_limited_config(tmp_path):
    from gputracker.gputracker import DispatchThread, WorkerJob
    dispatcher = SimpleNamespace(config={'max_concurrent_jobs': None},lock=threading.Lock())
    thread = DispatchThread('cpu', [WorkerJob('true',cpu_only=True)], logging.getLogger(__name__),
                            dispatcher, max_concurrent_jobs=2)
    active = [SimpleNamespace(is_alive=lambda: True)] * 2
    assert not thread._has_job_slot(active)
