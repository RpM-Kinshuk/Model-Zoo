"""Resume checks compare the small canonical records, not spectral payloads."""

import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from measurement_config import (
    FORMAT_VERSION, NUMERICS_VERSION, artifact_compatibility, measurement_config,
)


def write_pair(tmp_path, *, names=("encoder.layers.0.q", "decoder.layers.0.q"),
               alphas=(2.5, math.nan), full_name="org/model"):
    csv_path = tmp_path / "model.csv"
    h5_path = tmp_path / "model.h5"
    config = measurement_config(SimpleNamespace(), model_id=full_name)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "longname", "alpha"])
        writer.writerows((full_name, name, "" if math.isnan(alpha) else alpha)
                         for name, alpha in zip(names, alphas))
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["numerics_version"] = NUMERICS_VERSION
        h5.attrs["measurement_config_json"] = json.dumps(config)
        h5.attrs["full_name"] = full_name
        h5.create_dataset("layers/longname", data=names, dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("layers/alpha", data=alphas)
    return csv_path, h5_path, config


def replace_csv(csv_path, rows, columns=("model_id", "longname", "alpha")):
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_streamed_identity_checks_accept_nan_and_quoted_unicode_names(tmp_path):
    pair = write_pair(tmp_path, names=("", "projection,quoted\"\nα"))

    assert artifact_compatibility(*pair) == (True, "compatible")


@pytest.mark.parametrize("rows,reason", [
    ([], "row count"),
    ([("org/model", "encoder.layers.0.q", 2.5)], "row count"),
    ([("org/model", "encoder.layers.0.q", 2.5), ("org/model", "decoder.layers.0.q", ""),
      ("org/model", "extra", 2)], "row count"),
    ([("org/model", "decoder.layers.0.q", ""), ("org/model", "encoder.layers.0.q", 2.5)], "identities or order"),
    ([("org/model", "encoder.layers.0.q", 2.5), ("org/model", "encoder.layers.0.q", "")], "identities or order"),
    ([("org/other", "encoder.layers.0.q", 2.5), ("org/model", "decoder.layers.0.q", "")], "model identity"),
    ([("org/model", "encoder.layers.0.q", 2.7), ("org/model", "decoder.layers.0.q", "")], "alpha values"),
    ([("org/model", "encoder.layers.0.q", 2.5), ("org/model", "decoder.layers.0.q", 0)], "alpha values"),
    ([("org/model", "encoder.layers.0.q"), ("org/model", "decoder.layers.0.q", "")], "malformed CSV"),
    ([("org/model", "encoder.layers.0.q", 2.5, "extra"), ("org/model", "decoder.layers.0.q", "")], "malformed CSV"),
])
def test_mismatched_csv_never_resumes(tmp_path, rows, reason):
    csv_path, h5_path, config = write_pair(tmp_path)
    replace_csv(csv_path, rows)

    compatible, message = artifact_compatibility(csv_path, h5_path, config)

    assert not compatible
    assert reason in message


@pytest.mark.parametrize("columns", [
    ("model_id", "alpha"), ("model_id", "longname"),
    ("longname", "alpha"), ("model_id", "longname", "alpha", "alpha"),
])
def test_missing_or_duplicate_csv_columns_are_rejected(tmp_path, columns):
    csv_path, h5_path, config = write_pair(tmp_path)
    replace_csv(csv_path, [], columns)

    assert artifact_compatibility(csv_path, h5_path, config) == (
        False, "missing or duplicate CSV identity/alpha columns",
    )


def test_duplicate_hdf5_identities_are_rejected_even_when_csv_agrees(tmp_path):
    pair = write_pair(tmp_path, names=("same", "same"))

    assert artifact_compatibility(*pair) == (False, "duplicate canonical layer identities")


def test_csv_model_id_is_optional_only_without_hdf5_full_name(tmp_path):
    csv_path, h5_path, config = write_pair(tmp_path)
    with h5py.File(h5_path, "a") as h5:
        del h5.attrs["full_name"]
    replace_csv(csv_path, [("encoder.layers.0.q", 2.5), ("decoder.layers.0.q", "")],
                columns=("longname", "alpha"))

    assert artifact_compatibility(csv_path, h5_path, config) == (True, "compatible")


@pytest.mark.parametrize("contents", [
    b"model_id,longname,alpha\norg/model,encoder.layers.0.q,not-a-number\n",
    b'model_id,longname,alpha\norg/model,"unterminated,2.5\n',
    b"model_id,longname,alpha\norg/model,\xff,2.5\n",
])
def test_invalid_csv_contents_return_incompatible_without_raising(tmp_path, contents):
    csv_path, h5_path, config = write_pair(tmp_path)
    csv_path.write_bytes(contents)

    compatible, reason = artifact_compatibility(csv_path, h5_path, config)

    assert not compatible
    assert "unreadable or invalid CSV/HDF5" in reason


@pytest.mark.parametrize("method", ["stat", "open"])
def test_csv_stat_and_read_errors_return_incompatible(monkeypatch, tmp_path, method):
    csv_path, h5_path, config = write_pair(tmp_path)
    original = getattr(Path, method)

    def fail(path, *args, **kwargs):
        if path == csv_path:
            raise PermissionError("synthetic denied read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, fail)

    compatible, reason = artifact_compatibility(csv_path, h5_path, config)

    assert not compatible
    assert "PermissionError" in reason


def test_resume_does_not_scan_saved_eigenvalues(monkeypatch, tmp_path):
    csv_path, h5_path, config = write_pair(tmp_path)
    with h5py.File(h5_path, "a") as h5:
        h5.create_dataset("eigs", data=[1., 2.])
    original = h5py.Dataset.__getitem__

    def guard(dataset, key, *args, **kwargs):
        if dataset.name == "/eigs":
            pytest.fail("Resume must not read spectral payloads")
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guard)

    assert artifact_compatibility(csv_path, h5_path, config) == (True, "compatible")
