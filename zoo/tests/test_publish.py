import hashlib
import json
import os
from pathlib import Path

import h5py
import pytest

from zoo.publish import package_complete, package_inputs, publish_package, publish_packages, current_conventions_id
from zoo.testing import write_legacy_stats_csv


def test_publish_creates_sharded_package_with_hashes(package_run):
    r = package_run
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert path == r.packages / hashlib.sha256(r.safe.encode()).hexdigest()[:2] / r.safe / current_conventions_id()
    metadata = json.loads((path / "package.json").read_text())
    assert metadata["complete"] and metadata["kind"] == "full"
    assert metadata["model_key"] == "lab/tiny@" + r.manifest["revision"]
    assert metadata["evidence"] == dict(reused=2, measured=0, gaps=0)
    # Current runs package the HDF5 layer record alone.
    assert "esd/metrics.h5" in metadata["files"] and "esd/stats.csv" not in metadata["files"]
    assert not (path / "esd/stats.csv").exists()
    assert len(metadata["files"]) == 8
    for name, digest in metadata["files"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    assert not list(path.rglob("*.bin"))
    selected, skipped = package_inputs(r.packages)
    assert skipped == 0 and [meta["kind"] for _, meta in selected] == ["full"]


def test_legacy_run_with_matching_csv_publishes_it_unchanged(package_run):
    r = package_run
    legacy_csv = write_legacy_stats_csv(r.run / "metrics/lab--tiny.h5", r.run / "stats/lab--tiny.csv")
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    metadata = json.loads((path / "package.json").read_text())
    assert len(metadata["files"]) == 9
    assert (path / "esd/stats.csv").read_bytes() == legacy_csv.read_bytes()
    assert metadata["files"]["esd/stats.csv"] == hashlib.sha256(legacy_csv.read_bytes()).hexdigest()
    selected, skipped = package_inputs(r.packages)
    assert skipped == 0 and len(selected) == 1 and selected[0][1]["kind"] == "full"
    assert package_complete(r.run, "lab/tiny", r.packages, r.root)


def test_legacy_run_with_mismatching_csv_is_not_published(package_run):
    r = package_run
    legacy_csv = write_legacy_stats_csv(r.run / "metrics/lab--tiny.h5", r.run / "stats/lab--tiny.csv")
    rows = legacy_csv.read_text().splitlines()
    legacy_csv.write_text("\n".join(rows[:-1]) + "\n")  # One layer row fewer than the HDF5.
    assert publish_package(r.run, "lab/tiny", r.packages, r.root) is None
    assert not r.packages.exists() or not list(r.packages.rglob("package.json"))


def test_existing_legacy_package_is_never_replaced_by_hdf5_only_run(package_run):
    r = package_run
    legacy_csv = write_legacy_stats_csv(r.run / "metrics/lab--tiny.h5", r.run / "stats/lab--tiny.csv")
    legacy = publish_package(r.run, "lab/tiny", r.packages, r.root)
    before = {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()}
    # The same model and conventions measured again without a CSV share the
    # package destination; the different inventory is refused, not merged.
    legacy_csv.unlink()
    with pytest.raises(FileExistsError, match="only in existing package: esd/stats.csv"):
        publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert not package_complete(r.run, "lab/tiny", r.packages, r.root)
    assert {path.relative_to(legacy): path.read_bytes() for path in legacy.rglob("*") if path.is_file()} == before
    assert [path.parent for path in r.packages.rglob("package.json")] == [legacy]
    # Readers keep accepting the older package with its verified stats.csv.
    selected, _ = package_inputs(r.packages)
    assert [folder for folder, _ in selected] == [legacy] and "esd/stats.csv" in selected[0][1]["files"]


def test_publish_retains_source_snapshot_and_checks_its_identity(package_run):
    from evidence.source_metadata import read_source_metadata, source_metadata_path
    r = package_run
    manifest_path = r.folder / (r.safe + ".manifest.json")
    source = source_metadata_path(manifest_path)
    assert source.is_file()
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    packaged = package / source.name
    assert packaged.read_bytes() == source.read_bytes()
    metadata = json.loads((package / "package.json").read_text())
    assert metadata["files"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert read_source_metadata(packaged)["identity"]["repo_id"] == "lab/tiny"


def test_publish_rejects_present_invalid_source_snapshot(package_run):
    import gzip
    from evidence.source_metadata import source_metadata_path
    r = package_run
    source = source_metadata_path(r.folder / (r.safe + ".manifest.json"))
    snapshot = json.loads(gzip.decompress(source.read_bytes()))
    snapshot["identity"]["repo_id"] = "lab/wrong"
    source.write_bytes(gzip.compress(json.dumps(snapshot).encode(), mtime=0))
    with pytest.raises(ValueError, match="identity"):
        publish_package(r.run, "lab/tiny", r.packages, r.root)


def test_legacy_package_explicitly_records_missing_source_capture(package_run):
    from evidence.source_metadata import source_metadata_path
    r = package_run
    source_metadata_path(r.folder / (r.safe + ".manifest.json")).unlink()
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    metadata = json.loads((package / "package.json").read_text())
    assert metadata["source_metadata"]["status"] == "not_captured"


def test_quarantine_and_prepared_rewrite_keep_original_hub_card(package_run):
    from evidence.manifest import write_manifest
    from evidence.source_metadata import SourceMetadataCapture, source_metadata_path, write_source_metadata
    from evidence_bridge import _manifest, quarantine_evidence
    r = package_run
    manifest_path = r.folder / (r.safe + ".manifest.json")
    capture = SourceMetadataCapture("lab/tiny", r.manifest["revision"], source_kind="hub_api")
    capture.record_json("cardData.json", {"model-index": [{"name": "fixture", "results": []}]})
    source = write_source_metadata(capture.snapshot(), manifest_path)
    original = source.read_bytes()
    backup = quarantine_evidence(r.run, "lab/tiny", r.manifest["revision"])
    assert source.read_bytes() == original
    prepared = _manifest(r.run, "lab/tiny", r.manifest["revision"])
    staged = write_manifest(prepared, r.root / "prepared-stage")
    assert source_metadata_path(staged).read_bytes() == original
    # Restore the already measured fixture artifacts; no new model execution.
    for path in backup.iterdir():
        if path.name.endswith((".safetensors", ".h5")):
            os.link(path, r.folder / path.name)
    (r.folder / "evidence.json").write_bytes((backup / "evidence.json").read_bytes())
    write_manifest(dict(prepared), r.folder)
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert (package / source.name).read_bytes() == original


def test_publish_is_idempotent(package_run):
    r = package_run
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    before = (path / "package.json").read_bytes()
    assert publish_packages([r.run], r.packages, r.root) == [path]
    assert (path / "package.json").read_bytes() == before


def test_publish_never_overwrites_and_separates_conventions(package_run):
    r = package_run
    first = publish_package(r.run, "lab/tiny", r.packages, r.root)
    before = (first / "package.json").read_bytes()
    metrics = r.run / "metrics/lab--tiny.h5"
    # Producers atomically replace outputs; hardlinked package inodes stay immutable.
    replacement = metrics.with_suffix(".new")
    replacement.write_bytes(metrics.read_bytes())
    with h5py.File(replacement, "r+") as h:
        cfg = json.loads(h.attrs["measurement_config_json"])
        cfg["fix_fingers"] = "xmin_peak"
        h.attrs["measurement_config_json"] = json.dumps(cfg)
    os.replace(replacement, metrics)
    # A full package must also have matching evidence measurement settings.
    spectra = next(r.folder.glob("*.spectra.h5"))
    replacement = spectra.with_suffix(".new")
    replacement.write_bytes(spectra.read_bytes())
    with h5py.File(replacement, "r+") as h:
        h.attrs["measurement_config_json"] = json.dumps(cfg)
    os.replace(replacement, spectra)
    second = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert first != second and first.parent == second.parent
    assert (first / "package.json").read_bytes() == before
    second_before = (second / "package.json").read_bytes()
    # Same conventions, different inventory: a legacy CSV now sits beside the HDF5.
    write_legacy_stats_csv(metrics, r.run / "stats/lab--tiny.csv")
    with pytest.raises(FileExistsError):
        publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert (first / "package.json").read_bytes() == before
    assert (second / "package.json").read_bytes() == second_before
    assert not (second / "esd/stats.csv").exists()


def test_publish_skips_incomplete_models(package_run):
    r = package_run
    next(r.folder.glob("*.full_weights.fingerprint.safetensors")).unlink()
    assert publish_package(r.run, "lab/tiny", r.packages, r.root) is None
    assert not list(r.packages.rglob("package.json"))


def test_publish_row_blocks_only_package_without_esd_files(package_run):
    r = package_run
    for path in (r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.full_weights.*")):
        path.unlink()
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    metadata = json.loads((path / "package.json").read_text())
    assert metadata["kind"] == "row_blocks_only" and metadata["complete"]
    assert not (path / "esd").exists()


def test_publish_falls_back_to_copy_across_filesystems(package_run, monkeypatch):
    import errno
    import zoo.publish as module
    r = package_run
    def cross_device(*args):
        raise OSError(errno.EXDEV, "cross filesystem")
    monkeypatch.setattr(module.os, "link", cross_device)
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert (path / "esd/metrics.h5").is_file()


def test_row_blocks_preliminary_and_full_packages_coexist_without_overwrite(package_run):
    from zoo.build import build_zoo, open_tables
    r = package_run
    stash = r.root / "stash"
    stash.mkdir()
    saved = []
    for path in [r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.full_weights.*")]:
        other = stash / path.name
        os.replace(path, other)
        saved.append((other, path))
    preliminary = publish_package(r.run, "lab/tiny", r.packages, r.root)
    original = (preliminary / "package.json").read_bytes()
    for other, path in saved:
        os.replace(other, path)
    full = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert full != preliminary and (preliminary / "package.json").read_bytes() == original
    zoo = build_zoo(packages=r.packages, data_root=r.root, out=r.root / "built")
    tables = open_tables(zoo)
    assert len(tables["models"]) == 1 and len(tables["layers"]) == 2


def test_missing_summary_infers_nondefault_sketch_from_artifacts(package_run):
    from evidence.fingerprint import fingerprint_tensors
    from safetensors.torch import load_file
    r = package_run
    for path in [r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.full_weights.*"),
                 *r.folder.glob("*.row_blocks_*"), r.folder / "evidence.json"]:
        path.unlink()
    fingerprint_tensors(load_file(r.root / "source/model.safetensors").items(),
                        r.folder / f"{r.safe}.row_blocks.fingerprint.safetensors", r.manifest, sketch_k=128)
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    meta = json.loads((path / "package.json").read_text())
    assert meta["conventions"]["evidence"]["sketch_k"] == 128
    assert meta["conventions_id"] != current_conventions_id()


def test_atomic_replacement_during_publish_cannot_mislabel_settings(package_run, monkeypatch):
    import zoo.publish as module
    r = package_run
    original = module._candidate
    changed = False
    def candidate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            changed = True
            for source in [r.run / "metrics/lab--tiny.h5", next(r.folder.glob("*.spectra.h5"))]:
                replacement = source.with_suffix(".new")
                replacement.write_bytes(source.read_bytes())
                with h5py.File(replacement, "r+") as h:
                    cfg = json.loads(h.attrs["measurement_config_json"])
                    cfg["fix_fingers"] = "xmin_peak"
                    h.attrs["measurement_config_json"] = json.dumps(cfg)
                os.replace(replacement, source)
        return result
    monkeypatch.setattr(module, "_candidate", candidate)
    with pytest.raises(ValueError, match="changed|inconsistent"):
        publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert not list(r.packages.rglob("package.json"))


def test_pinned_publication_does_not_scan_other_models(package_run, monkeypatch):
    import zoo.publish as module
    r = package_run
    monkeypatch.setattr(module, "_manifest_files", lambda run: pytest.fail("global manifest scan for a pinned key"))
    assert publish_package(r.run, "lab/tiny@" + r.manifest["revision"], r.packages, r.root).is_dir()


def test_default_build_accepts_gpu_row_blocks_and_prefers_cpu_when_both_exist(package_run):
    from zoo.build import build_zoo, open_tables
    r = package_run
    for path in [r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.full_weights.*")]:
        path.unlink()
    summary = r.folder / "evidence.json"
    settings = dict(sketch_k=64, all_experts=False, with_parent=True, device="gpu")
    summary.write_text(json.dumps(dict(settings=settings)))
    gpu = publish_package(r.run, "lab/tiny", r.packages, r.root)
    built = build_zoo(packages=r.packages, data_root=r.root, out=r.root / "built")
    assert len(open_tables(built)["models"]) == 1
    settings["device"] = "cpu"
    summary.write_text(json.dumps(dict(settings=settings)))
    cpu = publish_package(r.run, "lab/tiny", r.packages, r.root)
    build_zoo(packages=r.packages, data_root=r.root, out=built)
    path = open_tables(built)["models"].iloc[0].row_block_fingerprint
    assert str(cpu.relative_to(r.root)) in path and gpu != cpu


def test_same_inventory_with_different_bytes_is_refused_without_changes(package_run):
    r = package_run
    existing = publish_package(r.run, "lab/tiny", r.packages, r.root)
    before = {path.relative_to(existing): path.read_bytes() for path in existing.rglob("*") if path.is_file()}
    metrics = r.run / "metrics/lab--tiny.h5"
    replacement = metrics.with_suffix(".new")  # Replace, never edit a hardlinked artifact in place.
    replacement.write_bytes(metrics.read_bytes())
    with h5py.File(replacement, "r+") as h:
        h.attrs["unrelated_note"] = "changes the bytes, not the inventory"
    os.replace(replacement, metrics)
    with pytest.raises(FileExistsError):
        publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert {path.relative_to(existing): path.read_bytes() for path in existing.rglob("*") if path.is_file()} == before
