import hashlib
import json
import os
from pathlib import Path

import h5py
import pytest

from zoo.publish import publish_package, publish_packages, current_conventions_id


def test_publish_creates_sharded_package_with_hashes(package_run):
    r = package_run
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert path == r.packages / hashlib.sha256(r.safe.encode()).hexdigest()[:2] / r.safe / current_conventions_id()
    metadata = json.loads((path / "package.json").read_text())
    assert metadata["complete"] and metadata["kind"] == "full"
    assert metadata["model_key"] == "lab/tiny@" + r.manifest["revision"]
    assert metadata["evidence"] == dict(reused=2, measured=0, gaps=0)
    assert "esd/stats.csv" in metadata["files"] and "esd/metrics.h5" in metadata["files"]
    assert len(metadata["files"]) == 8
    for name, digest in metadata["files"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    assert not list(path.rglob("*.bin"))


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
    stats = r.run / "stats/lab--tiny.csv"
    replacement = stats.with_suffix(".new")
    replacement.write_bytes(stats.read_bytes() + b"\n")
    os.replace(replacement, stats)
    with pytest.raises(FileExistsError):
        publish_package(r.run, "lab/tiny", r.packages, r.root)
    assert (first / "package.json").read_bytes() == before


def test_publish_skips_incomplete_models(package_run):
    r = package_run
    next(r.folder.glob("*.t2.fingerprint.safetensors")).unlink()
    assert publish_package(r.run, "lab/tiny", r.packages, r.root) is None
    assert not list(r.packages.rglob("package.json"))


def test_publish_t1_only_package_without_esd_files(package_run):
    r = package_run
    for path in (r.run / "stats/lab--tiny.csv", r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.t2.*")):
        path.unlink()
    path = publish_package(r.run, "lab/tiny", r.packages, r.root)
    metadata = json.loads((path / "package.json").read_text())
    assert metadata["kind"] == "t1_only" and metadata["complete"]
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


def test_t1_preliminary_and_full_packages_coexist_without_overwrite(package_run):
    from zoo.build import build_zoo, open_tables
    r = package_run
    stash = r.root / "stash"
    stash.mkdir()
    saved = []
    for path in [r.run / "stats/lab--tiny.csv", r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.t2.*")]:
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
    for path in [r.run / "stats/lab--tiny.csv", r.run / "metrics/lab--tiny.h5", *r.folder.glob("*.t2.*"),
                 *r.folder.glob("*.t1b*"), r.folder / "evidence.json"]:
        path.unlink()
    fingerprint_tensors(load_file(r.root / "source/model.safetensors").items(),
                        r.folder / f"{r.safe}.t1.fingerprint.safetensors", r.manifest, sketch_k=128)
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
