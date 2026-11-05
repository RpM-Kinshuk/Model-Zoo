import json
from pathlib import Path
import shutil

import h5py
import pandas as pd
import pytest

from zoo.schema import MODELS_COLUMNS, SCORES_COLUMNS, VIEWS_COLUMNS
from zoo.tables import build_tables, duplicate_groups
from zoo.testing import SHA1, SHA2, SHA3, SHA4, SHA5, SHA8, SHA9, SHA10, SHA11, SHA12


def test_models_and_coverage(tables):
    m = tables["models"].set_index("model_key")
    assert m.loc["org/t0only@" + SHA8, "coverage_status"] == "metadata_only"
    assert m.loc["org/gated@" + SHA9, "coverage_status"] == "no_access"
    assert m.loc["org/child@" + SHA2, "n_views"] > 0 and not Path(m.loc["org/child@" + SHA2, "row_block_fingerprint"]).is_absolute()
    assert m.loc["org/child-vocab@" + SHA3, "arch_signature"] == m.loc["org/parent@" + SHA1, "arch_signature"]
    assert list(tables["models"].model_key) == sorted(tables["models"].model_key)
    assert m.loc["org/decider@" + SHA11, "system_one"] and m.loc["org/decider@" + SHA11, "num_labels"] == 2
    assert m.loc["org/parent-GGUF@" + SHA12, "coverage_status"] == "metadata_only"


def test_duplicates_grouped(tables):
    m = tables["models"].set_index("model_key")
    assert m.loc["org/dup@" + SHA5, "dup_group"] == m.loc["org/child@" + SHA2, "dup_group"]
    assert m.loc["org/sibling@" + SHA4, "dup_group"] != m.loc["org/child@" + SHA2, "dup_group"]


def _full_measurement(root):
    import torch
    from safetensors.torch import save_file
    from evidence.fingerprint import fingerprint_model
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest, write_manifest

    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    generator = torch.Generator().manual_seed(310)
    save_file({"linear.weight": torch.randn(512, 64, generator=generator)}, checkpoint / "model.safetensors")
    manifest = build_manifest(str(checkpoint), "main", HubReader(), None)
    write_manifest(manifest, root / "manifests")
    fp = fingerprint_model(manifest, HubReader(), "full_weights", root / "fingerprints", also_row_block_sizes=(256,))
    return manifest, fp


def test_full_spectra_pointer_and_tier_preference_fallback(tmp_path):
    manifest, full_weights = _full_measurement(tmp_path)
    key = manifest["repo_id"] + "@" + manifest["revision"]
    combined = build_tables(tmp_path / "manifests", [tmp_path / "fingerprints"], tmp_path)
    row = combined["models"].set_index("model_key").loc[key]
    assert row.row_block_fingerprint and row.full_weight_fingerprint and row.full_weight_spectra
    assert not Path(row.full_weight_spectra).is_absolute()
    with h5py.File(tmp_path / row.full_weight_spectra) as handle:
        assert len(handle["eigs"][0]) == 64  # Full matrix spectrum, not block fits.
        from safetensors.torch import load_file
        full_weight = load_file(tmp_path / "checkpoint" / "model.safetensors")["linear.weight"]
        assert sum(handle["eigs"][0]) == pytest.approx(full_weight.square().sum().item(), rel=1e-5)
    assert combined["views"].tier.eq("row_blocks").all() and combined["views"].r0.eq(128).all()
    fallback = build_tables(tmp_path / "manifests", [full_weights], tmp_path)
    fallback_row = fallback["models"].iloc[0]
    assert not fallback_row.row_block_fingerprint and fallback_row.full_weight_spectra == row.full_weight_spectra
    assert fallback["views"].tier.eq("full_weights").all() and fallback["views"].r0.eq(0).all()
    assert fallback_row.dup_group == key
    assert manifest["repo_id"] in key


def test_duplicate_input_and_orphan_identity_are_explicit_errors(universe, tmp_path):
    path = next(universe.fingerprints.glob("*.row_blocks.fingerprint.safetensors"))
    copied = tmp_path / "other" / path.name
    copied.parent.mkdir()
    shutil.copy2(path, copied)
    with pytest.raises(ValueError) as error:
        build_tables(universe.manifests, [path, copied], universe.root)
    assert str(path) in str(error.value) and str(copied) in str(error.value)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="manifest|identity"):
        build_tables(empty, [path], universe.root)


def test_coverage_frame_and_scores_keep_measurements_separate(universe, tmp_path):
    from zoo.scores import load_scores_csv
    from measurement_config import safe_filename

    key = "org/child@" + SHA2
    fingerprint = universe.fingerprints / (safe_filename(key) + ".row_blocks.fingerprint.safetensors")
    # An extra unsupported tensor makes coverage partial without removing the
    # existing matrices or inventing a zero-valued view.
    from safetensors import safe_open
    from safetensors.numpy import save_file
    with safe_open(fingerprint, framework="numpy") as handle:
        metadata = dict(handle.metadata())
        arrays = {name: handle.get_tensor(name) for name in handle.keys()}
    coverage = json.loads(metadata["coverage"])
    coverage.append(dict(name="packed.weight", status="skipped", reason="unsupported_representation", decode="plain", geometry="matrix"))
    metadata["coverage"] = json.dumps(coverage)
    save_file(arrays, fingerprint, metadata=metadata)
    frame = tmp_path / "models.csv"
    pd.DataFrame([dict(repo_id="org/child", revision=SHA2, modality="text", pipeline_tag="text-generation",
                       downloads=12, params=12345, stratum="text:<1B", inclusion_prob=.2)]).to_csv(frame, index=False)
    score_source = tmp_path / "scores.csv"
    score_source.write_text("repo,benchmark,value\norg/child,acc,0.6\n")
    models = pd.DataFrame([dict(model_key=key, repo_id="org/child", revision=SHA2)])
    scores = load_scores_csv(score_source, {"repo_id": "repo", "benchmark": "benchmark", "value": "value", "source": "fixture"}, models)
    result = build_tables(universe.manifests, [universe.fingerprints], universe.root, frame, [scores])
    row = result["models"].set_index("model_key").loc[key]
    assert row.coverage_status == "partial" and row.inclusion_prob == .2 and row.stratum == "text:<1B"
    assert row.skipped_tensors and any(item["reason"] == "unsupported_representation" for item in row.skipped_tensors)
    assert result["models"].set_index("model_key").loc["org/child@" + SHA10, "stratum"] == "lab"
    assert len(result["scores"]) == 1 and not result["scores"].revision_verified.any()
    assert result["predictions"].empty
    assert set(result["models"]) == set(MODELS_COLUMNS) and set(result["views"]) == set(VIEWS_COLUMNS)
    assert set(result["scores"]) == set(SCORES_COLUMNS)


def test_duplicate_groups_require_matching_row_blocks_geometry_and_coordinates():
    base = dict(view_index=0, name="weight", geometry="matrix", n_rows=512, n_cols=64,
                r0=128, r1=384, block_sha256="a" * 64, tier="row_blocks")
    rows = [dict(base, model_key="a"), dict(base, model_key="b"),
            dict(base, model_key="coords", r0=0, r1=256),
            dict(base, model_key="geometry", geometry="head"),
            dict(base, model_key="columns", n_cols=128),
            dict(base, model_key="full", tier="full_weights")]
    groups = duplicate_groups(pd.DataFrame(rows))
    assert groups["a"] == groups["b"] == "a"
    assert all(groups[key] == key for key in ("coords", "geometry", "columns", "full"))


def test_paths_cannot_escape_data_root_and_empty_inputs_stay_typed(universe, tmp_path):
    with pytest.raises(ValueError, match="data_root|outside"):
        build_tables(universe.manifests, [universe.fingerprints], tmp_path / "other-root")
    empty = tmp_path / "empty"
    empty.mkdir()
    result = build_tables(empty, [], tmp_path)
    assert all(table.empty for table in result.values())
    assert list(result["models"].columns) == MODELS_COLUMNS


@pytest.mark.parametrize("gated", ["unexpected", "false", ""])
def test_unknown_hub_gating_strings_are_explicit_errors(tmp_path, gated):
    manifest = dict(repo_id="org/model", revision=SHA1, access="metadata_only", tensors=[], gated=gated)
    (tmp_path / "model.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gated"):
        build_tables(tmp_path, [], tmp_path)
