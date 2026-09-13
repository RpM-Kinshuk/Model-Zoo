"""The depth-only dashboard must not silently discard canonical layers."""

import json
from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "clustering" / "unsupervised"))
from alpha_cluster_dashboard.data import load_alpha_records, records_to_frame


def write_alpha(path, *, canonical=True, status="complete"):
    with h5py.File(path, "w") as handle:
        alpha = handle.create_dataset("alpha", data=[[2.5], [np.nan]])
        alpha.attrs["num_layers"] = 2
        alpha.attrs["module_names_json"] = json.dumps(["self_attn.q_proj"])
        if status is not None:
            alpha.attrs["view_status"] = status
        if canonical:
            handle.attrs["format_version"] = "2.0"
            alpha.attrs["canonical_records"] = "/layers"
            alpha.attrs["unmapped_layer_count"] = 0
            handle.create_dataset("layers/longname", data=[
                "model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj",
            ], dtype=h5py.string_dtype())
            handle.create_dataset("layers/alpha", data=[2.5, np.nan])


def test_complete_view_preserves_missing_fits_and_filename_identity(tmp_path):
    path = tmp_path / "org--model.h5"
    write_alpha(path)
    with h5py.File(path, "a") as handle:
        handle.attrs["full_name"] = "different/metadata-name"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        records = load_alpha_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.model_id == "org--model"
    assert record.display_name == "org/model"
    assert record.view_status == "complete"
    assert record.num_layers == 2
    np.testing.assert_equal(record.alpha, [[2.5], [np.nan]])
    assert records_to_frame(records).iloc[0]["view_status"] == "complete"


@pytest.mark.parametrize("status", [None, "partial", "unavailable", "unknown"])
def test_incomplete_or_unspecified_canonical_view_is_excluded(tmp_path, status):
    write_alpha(tmp_path / "org--model.h5", status=status)
    with pytest.warns(UserWarning, match=f"view_status is {status or 'missing'}"):
        assert load_alpha_records(tmp_path) == []


def test_complete_status_cannot_override_unmapped_layers(tmp_path):
    path = tmp_path / "org--model.h5"
    write_alpha(path)
    with h5py.File(path, "a") as handle:
        handle["alpha"].attrs["unmapped_layer_count"] = 1
    with pytest.warns(UserWarning, match="omits canonical layers"):
        assert load_alpha_records(tmp_path) == []


@pytest.mark.parametrize("canonical_marker", ["format_version", "canonical_records"])
def test_current_file_without_canonical_records_is_not_legacy(tmp_path, canonical_marker):
    path = tmp_path / "org--model.h5"
    write_alpha(path, canonical=False, status=None)
    with h5py.File(path, "a") as handle:
        if canonical_marker == "format_version":
            handle.attrs["format_version"] = "2.0"
        else:
            handle["alpha"].attrs["canonical_records"] = "/layers"
    with pytest.warns(UserWarning, match="missing canonical /layers"):
        assert load_alpha_records(tmp_path) == []


def test_legacy_files_are_labeled_with_one_warning(tmp_path):
    for name in ("a", "b"):
        write_alpha(tmp_path / f"{name}.h5", canonical=False, status=None)
    with pytest.warns(UserWarning, match="Loaded 2 legacy root-only") as caught:
        records = load_alpha_records(tmp_path)
    assert len(caught) == 1
    assert [record.model_id for record in records] == ["a", "b"]
    assert records_to_frame(records)["view_status"].tolist() == ["legacy", "legacy"]


def test_partial_root_only_view_is_not_accepted_as_legacy(tmp_path):
    write_alpha(tmp_path / "partial.h5", canonical=False, status="partial")
    with pytest.warns(UserWarning, match="view_status is partial"):
        assert load_alpha_records(tmp_path) == []


@pytest.mark.parametrize("mutation", [
    "missing_alpha", "missing_labels", "duplicate_labels", "non_string_labels",
    "wrong_shape", "empty_view", "corrupt",
])
def test_malformed_file_does_not_hide_other_records(tmp_path, mutation):
    broken = tmp_path / "broken.h5"
    write_alpha(broken)
    write_alpha(tmp_path / "valid.h5")
    if mutation == "corrupt":
        broken.write_text("not an HDF5 file")
    else:
        with h5py.File(broken, "a") as handle:
            if mutation == "missing_alpha":
                del handle["alpha"]
            elif mutation == "missing_labels":
                del handle["alpha"].attrs["module_names_json"]
            elif mutation == "duplicate_labels":
                handle["alpha"].attrs["module_names_json"] = '["q", "q"]'
            elif mutation == "non_string_labels":
                handle["alpha"].attrs["module_names_json"] = '[7]'
            elif mutation == "wrong_shape":
                handle["alpha"].attrs["num_layers"] = 1
            elif mutation == "empty_view":
                handle["alpha"].attrs["module_names_json"] = '[]'
    with pytest.warns(UserWarning, match="Skipping broken.h5"):
        records = load_alpha_records(tmp_path)
    assert [record.model_id for record in records] == ["valid"]


def test_bytes_status_is_understood(tmp_path):
    write_alpha(tmp_path / "model.h5", status=np.bytes_("complete"))
    assert load_alpha_records(tmp_path)[0].view_status == "complete"


def test_empty_directory_is_quiet(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_alpha_records(tmp_path) == []
