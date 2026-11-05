"""Sidecar parity, conservative runner reuse, and transactional publication."""
import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from evidence.fingerprint import fingerprint_model, load_fingerprint
from evidence.hub import HubReader
from evidence.sidecar import EvidenceSidecar, publish_full_weight_spectra
from measurement_config import measurement_config, safe_filename
from net_esd.core import compute_esd_for_weight


def config():
    return measurement_config(SimpleNamespace(analysis_source="checkpoint"))


def measure(items):
    cfg = config()
    return {name: compute_esd_for_weight(name, tensor, cfg["evals_thresh"], cfg["bins"],
            cfg["fix_fingers"], cfg["xmin_pos"], cfg["conv_norm"], cfg["filter_zeros"],
            cfg["use_svd"], True, None, tensor.numel(), cfg["compute_dtype"])
            for name, tensor in items}


def weights(manifest):
    return load_file(Path(manifest["repo_id"]) / "model.safetensors")


def runner_results(manifest):
    result = {}
    for name, tensor in weights(manifest).items():
        if tensor.ndim != 2:
            continue
        record = measure([(name, tensor)])[name]
        module, _, attribute = name.rpartition(".")
        result["runner:" + name] = dict(record, module_name=module, weight_attribute=attribute, slice="")
    return result


def loader(manifest):
    tensors = weights(manifest)
    return lambda name, bounds: tensors[name][bounds[0]:bounds[1]]


def test_sidecar_outputs_match_standalone_full_weights(local_manifest, tmp_path):
    standalone = tmp_path / "standalone"
    expected_path = fingerprint_model(local_manifest, HubReader(), "full_weights", standalone,
                                      also_row_block_sizes=(128, 256, 512))
    out = tmp_path / "sidecar"
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    counts = sidecar.finish({}, lambda *args: None, measure, config())
    assert counts == dict(reused=0, measured=2, gaps=0)
    for path in standalone.glob("*.safetensors"):
        a, b = load_fingerprint(path), load_fingerprint(out / path.name)
        assert a.names == b.names and a.sha256 == b.sha256
        np.testing.assert_array_equal(a.sketch, b.sketch)
        for x, y in zip(a.spectra, b.spectra):
            np.testing.assert_array_equal(x, y)
        for key in a.delta:
            np.testing.assert_equal(a.delta[key], b.delta[key])
    assert (out / expected_path.name).exists()
    with h5py.File(next(standalone.glob("*.h5"))) as a, h5py.File(next(out.glob("*.h5"))) as b:
        for key in a["layers"]:
            np.testing.assert_equal(a["layers"][key][:], b["layers"][key][:])


def test_finish_reuses_runner_spectra_for_matching_views(local_manifest, tmp_path):
    sidecar = EvidenceSidecar(local_manifest, tmp_path / "out")
    sidecar.start()
    counts = sidecar.finish(runner_results(local_manifest), loader(local_manifest),
                            lambda items: pytest.fail("unexpected second round"), config())
    assert counts == dict(reused=2, measured=0, gaps=0)
    with h5py.File(next((tmp_path / "out").glob("*.h5"))) as h:
        assert set(h["layers/spectrum_source"].asstr()[:]) == {"runner"}
        assert json.loads(h.attrs["measurement_config_json"]) == config()


def test_finish_remeasures_views_whose_loaded_block_differs(local_manifest, tmp_path):
    loaded, second_round = loader(local_manifest), []
    def changed(name, bounds):
        block = loaded(name, bounds)
        return block + 1 if "q_proj" in name else block
    def second(items):
        second_round.extend(name for name, _ in items)
        return measure(items)
    sidecar = EvidenceSidecar(local_manifest, tmp_path / "out")
    sidecar.start()
    assert sidecar.finish(runner_results(local_manifest), changed, second, config()) == dict(reused=1, measured=1, gaps=0)
    assert second_round == ["model.layers.0.self_attn.q_proj.weight"]
    with h5py.File(next((tmp_path / "out").glob("*.h5"))) as h:
        provenance = dict(zip(h["layers/longname"].asstr()[:], h["layers/spectrum_source"].asstr()[:]))
        assert provenance[second_round[0]] == "second_round"


def test_full_weights_marker_published_after_spectra(local_manifest, tmp_path, monkeypatch):
    import evidence.sidecar as module
    out = tmp_path / "out"
    original, markers = module.os.replace, []
    def replace(source, destination):
        destination = Path(destination)
        if destination.parent == out and destination.name.endswith(".full_weights.fingerprint.safetensors"):
            assert destination.with_name(destination.name.replace(".full_weights.fingerprint.safetensors", ".full_weights.spectra.h5")).is_file()
            assert list(out.glob("*.manifest.json"))
            markers.append(destination)
        return original(source, destination)
    monkeypatch.setattr(module.os, "replace", replace)
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    sidecar.finish({}, lambda *args: None, measure, config())
    assert len(markers) == 1


def test_cancel_publishes_nothing(local_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    entered, release = Event(), Event()
    original = module.two_sided_sketch
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "two_sided_sketch", blocked)
    out = tmp_path / "out"
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    assert entered.wait(3)
    cancel = Thread(target=sidecar.cancel)
    cancel.start()
    release.set()
    cancel.join(3)
    assert not cancel.is_alive()
    assert not list(out.iterdir())


def test_write_spectra_false_stages_marker_without_measuring(local_manifest, tmp_path, monkeypatch):
    monkeypatch.setattr("evidence.fingerprint.compute_esd_for_weight", lambda *args: pytest.fail("spectrum measured"))
    path = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path,
                             write_spectra=False, also_row_block_sizes=(256,))
    assert path.exists() and not path.name.endswith(".full_weights.fingerprint.safetensors")
    assert not list(tmp_path.glob("*.h5"))
    assert not list(tmp_path.glob("*.full_weights.fingerprint.safetensors"))
    assert list(tmp_path.glob("*.row_blocks.fingerprint.safetensors"))


@pytest.mark.parametrize("kind", ["slice", "shape", "dtype"])
def test_finish_rejects_unsafe_runner_reuse(local_manifest, tmp_path, kind):
    results = runner_results(local_manifest)
    for record in results.values():
        if kind == "slice":
            record["slice"] = "[0]"
        if kind == "shape":
            record["M"] += 1
    loaded = loader(local_manifest)
    if kind == "dtype":
        loaded = lambda name, bounds: loader(local_manifest)(name, bounds).double()
    sidecar = EvidenceSidecar(local_manifest, tmp_path / "out")
    sidecar.start()
    assert sidecar.finish(results, loaded, measure, config()) == dict(reused=0, measured=2, gaps=0)


def test_publish_full_weight_spectra_does_not_replace_existing_completed_evidence(local_manifest, tmp_path):
    existing = fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path)
    old_bytes = existing.read_bytes()
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path, write_spectra=False)
    results = [(name, dict(result, spectrum_source="second_round")) for name, result in
               measure([(name, tensor) for name, tensor in weights(local_manifest).items() if tensor.ndim == 2]).items()]
    with pytest.raises(FileExistsError):
        publish_full_weight_spectra(tmp_path, local_manifest, results, config())
    assert existing.read_bytes() == old_bytes


def test_publish_failure_leaves_no_completion_marker(local_manifest, tmp_path, monkeypatch):
    import evidence.sidecar as module
    out = tmp_path / "out"
    original = module.os.replace
    def replace(source, destination):
        if Path(destination).parent == out and str(destination).endswith(".full_weights.fingerprint.safetensors"):
            raise OSError("interrupted publication")
        return original(source, destination)
    monkeypatch.setattr(module.os, "replace", replace)
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    with pytest.raises(OSError, match="interrupted"):
        sidecar.finish({}, lambda *args: None, measure, config())
    assert not list(out.iterdir())


def test_missing_second_round_result_publishes_nothing(local_manifest, tmp_path):
    out = tmp_path / "out"
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    with pytest.raises(ValueError, match="did not measure"):
        sidecar.finish({}, lambda *args: None, lambda items: {}, config())
    assert not list(out.iterdir())


def test_cancel_during_final_marker_rename_rolls_back(local_manifest, tmp_path, monkeypatch):
    from concurrent.futures import CancelledError
    import evidence.sidecar as module
    out = tmp_path / "out"
    entered, release, errors = Event(), Event(), []
    original = module.os.replace
    def replace(source, destination):
        if Path(destination).parent == out and str(destination).endswith(".full_weights.fingerprint.safetensors"):
            entered.set()
            assert release.wait(3)
        return original(source, destination)
    monkeypatch.setattr(module.os, "replace", replace)
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    def finish():
        try:
            sidecar.finish({}, lambda *args: None, measure, config())
        except BaseException as exc:
            errors.append(exc)
    finishing = Thread(target=finish)
    finishing.start()
    assert entered.wait(3)
    cancelling = Thread(target=sidecar.cancel)
    cancelling.start()
    assert sidecar._cancelled.wait(3)
    release.set()
    finishing.join(3)
    cancelling.join(3)
    assert not finishing.is_alive() and not cancelling.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], CancelledError)
    assert not list(out.iterdir())


def test_failed_promotion_restores_existing_incomplete_files(local_manifest, tmp_path, monkeypatch):
    import evidence.sidecar as module
    out = tmp_path / "out"
    out.mkdir()
    safe = safe_filename(f"{local_manifest['repo_id']}@{local_manifest['revision']}")
    old = {f"{safe}.manifest.json": b"original manifest", f"{safe}.row_blocks.fingerprint.safetensors": b"original row-block"}
    for name, data in old.items():
        (out / name).write_bytes(data)
    original = module.os.replace
    def replace(source, destination):
        if Path(destination).parent == out and str(destination).endswith(".full_weights.fingerprint.safetensors"):
            raise OSError("interrupted publication")
        return original(source, destination)
    monkeypatch.setattr(module.os, "replace", replace)
    sidecar = EvidenceSidecar(local_manifest, out)
    sidecar.start()
    with pytest.raises(OSError, match="interrupted"):
        sidecar.finish({}, lambda *args: None, measure, config())
    assert {path.name: path.read_bytes() for path in out.iterdir()} == old


def test_sidecar_streams_expert_views_instead_of_full_fused_sources(moe_manifest, tmp_path, monkeypatch):
    import evidence.fingerprint as module
    def reject_full_source(*args):
        pytest.fail("whole fused expert source loaded despite byte budget")
    monkeypatch.setattr(module._Sources, "full", reject_full_source)
    batches = []
    def second_round(items):
        batches.append(len(items))
        return measure(items)
    sidecar = EvidenceSidecar(moe_manifest, tmp_path / "out", max_inflight_bytes=1024,
                              sketch_k=4, all_experts=True)
    sidecar.start()
    counts = sidecar.finish({}, lambda *args: None, second_round, config())
    assert counts == dict(reused=0, measured=8, gaps=0)
    assert batches == [1] * 8


def test_published_spectrum_device_describes_supplied_results(local_manifest, tmp_path):
    fingerprint_model(local_manifest, HubReader(), "full_weights", tmp_path, write_spectra=False)
    supplied = [(name, dict(result, spectrum_source="runner", compute_device="cuda:1"))
                for name, result in measure([(name, tensor) for name, tensor in weights(local_manifest).items()
                                            if tensor.ndim == 2]).items()]
    publish_full_weight_spectra(tmp_path, local_manifest, supplied, config())
    with h5py.File(next(tmp_path.glob("*.h5"))) as h:
        assert h.attrs["compute_device"] == "cuda:1"
        assert set(h["layers/compute_device"].asstr()[:]) == {"cuda:1"}


def test_batched_spectra_matches_streaming_layout_and_values(tmp_path):
    from evidence.fingerprint import _Spectra
    rows = [
        ("a.weight", dict(eigs=np.array([1., 2.], np.float32), alpha=2., fit_status="ok", extra=3.)),
        ("b.weight", dict(eigs=np.array([0., 4., 5.], np.float32), alpha=np.float64(2.5), fit_status="small", late=7)),
        ("c.weight", dict(eigs=np.array([6.], np.float32), alpha=3., fit_status="ok")),
    ]
    for label in ("stream", "batch"):
        writer = _Spectra(tmp_path / f"{label}.h5", torch.device("cpu"), config())
        try:
            if label == "stream":
                for name, result in rows:
                    writer.add_result(name, result)
            else:
                writer.add_results(rows)
            writer.close(publish=True)
        except BaseException:
            writer.close()
            raise
    with h5py.File(tmp_path / "stream.h5") as left, h5py.File(tmp_path / "batch.h5") as right:
        assert dict(left.attrs) == dict(right.attrs)
        assert list(left["layers"]) == list(right["layers"])
        for key in left["layers"]:
            a, b = left["layers"][key], right["layers"][key]
            assert a.dtype == b.dtype and a.maxshape == b.maxshape
            np.testing.assert_equal(a[:], b[:])
        assert left["eigs"].dtype == right["eigs"].dtype
        assert left["eigs"].maxshape == right["eigs"].maxshape
        for a, b in zip(left["eigs"][:], right["eigs"][:]):
            np.testing.assert_array_equal(a, b)


def test_sidecar_publishes_spectra_in_batch(local_manifest, tmp_path, monkeypatch):
    from evidence.fingerprint import _Spectra
    monkeypatch.setattr(_Spectra, "add_result", lambda *args: pytest.fail("row-by-row spectrum write"))
    sidecar = EvidenceSidecar(local_manifest, tmp_path / "out")
    sidecar.start()
    assert sidecar.finish(runner_results(local_manifest), loader(local_manifest), measure, config()) == dict(reused=2, measured=0, gaps=0)
