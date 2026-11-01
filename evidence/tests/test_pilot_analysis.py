import json

import numpy as np
import pandas as pd
import pytest

from evidence.pilot_analysis import (lineage_pairs, tier_agreement, spectra_agreement,
                                     throughput_report, write_calibration, write_g0_report)


def test_tier_agreement_metrics(synthetic_run):
    out = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert out["auroc_t1"] == 1.0 and out["auroc_t2"] == 1.0
    assert out["spearman_within_related"] > 0.8 and out["decision_agreement"] >= 0.9
    assert set(out["byte_curve"]) == {128, 256, 512}


def test_g0_report_marks_pass_fail(synthetic_run, tmp_path):
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")
    text = (tmp_path / "g0.md").read_text()
    assert ("PASS" in text or "FAIL" in text) and "false_matches_per_query" in text


def test_calibration_is_compatible_empirical_and_excludes_sanity(synthetic_run, tmp_path):
    from evidence.compare import diff_fingerprints
    from evidence.fingerprint import load_fingerprint
    out = tmp_path / "calibration.json"
    pairs = synthetic_run.pairs.copy()
    sanity = pairs.iloc[[0]].assign(label="independent", source="hub_declared_base")
    write_calibration(pd.concat([pairs, sanity]), synthetic_run.t1, out)
    null = json.loads(out.read_text())
    assert null["n_pairs"] == 3 and len(null["null_quantiles"]) == 3
    assert null["statistic"] == "median_cosine" and null["sketch_k"] == 64
    fps = [load_fingerprint(p) for p in synthetic_run.t1.glob("*.fingerprint.safetensors")]
    relatedness = diff_fingerprints(fps[0], fps[0], null)["relatedness"]
    assert relatedness["verdict"] != "uncalibrated" and relatedness["p_value"] >= .25
    with pytest.raises(ValueError, match="independent"):
        write_calibration(pairs[pairs.label != "independent"], synthetic_run.t1, out)
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_measurements_are_visible_and_do_not_pass(tmp_path):
    t1, t2 = tmp_path / "t1", tmp_path / "t2"
    t1.mkdir(); t2.mkdir()
    pairs = pd.DataFrame([dict(a="a@rev", b="b@rev", label="independent", source="lab")])
    out = tier_agreement(pairs, t1, t2)
    assert out["auroc_t1"] is None and out["n_missing_pairs"] == 1
    with pytest.raises(ValueError, match="independent"):
        write_calibration(pairs, t1, tmp_path / "null.json")
    write_g0_report(tmp_path, tmp_path / "g0.md")
    report = (tmp_path / "g0.md").read_text()
    assert "FAIL" in report and "unavailable" in report and "PASS" not in report


def test_spectra_use_full_artifacts_and_disclose_sorted_rank_limit(synthetic_run):
    out = spectra_agreement(synthetic_run.t1, synthetic_run.t2)
    assert out["n_matrices"] == 9 and out["n_alpha_pairs"] == 9
    assert np.isfinite(out["median_abs_alpha_diff"])
    assert out["median_per_matrix_spearman"] == pytest.approx(1.)
    assert "sorted" in out["spearman_note"]


def test_throughput_uses_requested_denominator_and_current_summary(synthetic_run):
    path = synthetic_run.t1 / "summary.json"
    summary = json.loads(path.read_text())
    summary["models"][0].update(bytes_read=100, elapsed_seconds=20)
    summary["models"] += [dict(repo_id="missing", revision="x", stratum="text:<1B", status="failed",
                               reason="error", bytes_read=0, elapsed_seconds=1, resolves=0),
                          dict(repo_id="gated", revision="x", stratum="text:<1B", status="failed",
                               reason="no_access", bytes_read=0, elapsed_seconds=1, resolves=0)]
    path.write_text(json.dumps(summary))
    (synthetic_run.t1 / "failures.csv").write_text("repo_id,stage,reason,message\nold,batch,error,historical\n")
    out = throughput_report(synthetic_run.root).iloc[0]
    assert out.n_models == 11 and out.n_eligible == 10 and out.failure_rate == pytest.approx(.1)
    assert out.failure_rate_by_reason == {"error": pytest.approx(.1)}
    assert out.bytes_read_median > 100 and out.n_completed == 9


def test_lineage_uses_graph_and_certified_reference_pairs(tmp_path):
    models = [dict(model_id=n, kind="derived", path=n, base_family="same", license="mit")
              for n in ("p", "a", "b", "g", "i", "j", "s", "t")]
    for row in models[4:]:
        row["kind"] = "independent"
    pd.DataFrame(models).to_csv(tmp_path / "models.csv", index=False)
    edges = pd.DataFrame([dict(child="a", parent="p", role="parent"),
                          dict(child="b", parent="p", role="parent"),
                          dict(child="g", parent="a", role="parent"),
                          dict(child="g", parent="b", role="base")])
    edges.to_csv(tmp_path / "edges.csv", index=False)
    (tmp_path / "reference_pairs.json").write_text(json.dumps({"pairs": [
        dict(model_ids=["i", "j"], a="lab/i@local:x", b="lab/j@local:x", label="independent", source="verified_hub_training_records"),
        dict(model_ids=["s", "t"], a="lab/s@local:x", b="lab/t@local:x", label="sibling", source="verified_hub_training_records")]}))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name in ("p", "a", "b", "g", "i", "j", "s", "t"):
        (manifests / f"{name}.manifest.json").write_text(json.dumps(dict(
            repo_id=f"lab/{name}", revision="local:x", tensors=[dict(name="linear.weight", shape=[512,64])],
            declared_bases=["lab/p"] if name == "g" else [])))
    out = lineage_pairs(tmp_path / "edges.csv", manifests)
    labels = {(r.a, r.b): r.label for r in out.itertuples() if r.source == "lab"}
    assert labels[("lab/a@local:x", "lab/p@local:x")] == "parent"
    assert labels[("lab/g@local:x", "lab/p@local:x")] == "grandparent"
    assert any(r.label == "sibling" and {r.a, r.b} == {"lab/a@local:x", "lab/b@local:x"} for r in out.itertuples())
    assert len(out[out.label == "independent"]) == 1
    assert not any(r.label == "parent" and r.a == "lab/g@local:x" and r.b == "lab/b@local:x" for r in out.itertuples())
    assert set(out[out.label == "independent"].iloc[0][["a", "b"]]) == {"lab/i@local:x", "lab/j@local:x"}


def test_unknown_stratum_and_partial_cost_coverage_cannot_pass(synthetic_run, tmp_path):
    summary_path = synthetic_run.t1 / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["models"][0].pop("elapsed_seconds")
    for row in summary["models"]:
        row["stratum"] = "unknown"
    summary_path.write_text(json.dumps(summary))
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")
    report = (tmp_path / "g0.md").read_text()
    assert "PASS unknown/" not in report
    row = throughput_report(synthetic_run.root).iloc[0]
    assert not row.measurement_coverage_complete and row.n_missing_elapsed == 1


def test_calibration_requires_parent_settings_match_null(synthetic_run, tmp_path, monkeypatch):
    from evidence import pilot_analysis as analysis
    real = analysis._fingerprints
    parent_key = synthetic_run.pairs.loc[synthetic_run.pairs.label == "parent", "a"].iloc[0]
    def altered(directory, tier):
        result = real(directory, tier)
        # Shift the entire related subset together: pairwise matching alone
        # must not permit a different projection to set this null's threshold.
        for key, fp in result.items():
            if "independent" not in key:
                fp.meta["sketch_seed"] += 1
        assert parent_key in result
        return result
    monkeypatch.setattr(analysis, "_fingerprints", altered)
    with pytest.raises(ValueError, match="settings"):
        write_calibration(synthetic_run.pairs, synthetic_run.t1, tmp_path / "null.json")


def test_spectral_partial_coverage_is_explicit(synthetic_run, tmp_path):
    # A missing full model must not make agreement on the rest certify G0.
    next(synthetic_run.t2.glob("*.t2.fingerprint.safetensors")).unlink()
    result = spectra_agreement(synthetic_run.t1, synthetic_run.t2)
    assert result["n_models_requested"] == 9 and result["n_models_matched"] == 8
    assert not result["measurement_coverage_complete"]
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")
    assert "FAIL spectral_measurement_coverage" in (tmp_path / "g0.md").read_text()


def test_missing_full_spectrum_companion_stays_unavailable(synthetic_run, tmp_path):
    next(synthetic_run.t2.glob("*.t2.spectra.h5")).unlink()
    result = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert result["n_missing_pairs"] > 0
    spectral = spectra_agreement(synthetic_run.t1, synthetic_run.t2)
    assert not spectral["measurement_coverage_complete"]
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")


def test_calibration_requires_all_requested_parent_pairs(synthetic_run, tmp_path, monkeypatch):
    from evidence import pilot_analysis as analysis
    real = analysis._fingerprints
    missing = synthetic_run.pairs.loc[synthetic_run.pairs.label == "parent", "a"].iloc[0]
    def partial(directory, tier):
        result = real(directory, tier)
        result.pop(missing)
        return result
    monkeypatch.setattr(analysis, "_fingerprints", partial)
    with pytest.raises(ValueError, match="parent"):
        write_calibration(synthetic_run.pairs, synthetic_run.t1, tmp_path / "null.json")


def test_t1_false_matches_use_own_tpr_operating_point(synthetic_run, monkeypatch):
    from evidence import pilot_analysis as analysis
    real_score, real_false_matches = analysis._score, analysis._false_matches
    thresholds = []
    def shifted(a, b, k=64):
        value = real_score(a, b, k)
        return value - .5 if value is not None and a.meta["tier"] == "t1" else value
    def capture(pairs, fps, threshold):
        thresholds.append(threshold)
        return real_false_matches(pairs, fps, threshold)
    monkeypatch.setattr(analysis, "_score", shifted)
    monkeypatch.setattr(analysis, "_false_matches", capture)
    result = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert thresholds[0] < result["threshold_t2_tpr_0_9"] - .4
    assert result["achieved_tpr_t1"] >= .9
    assert thresholds[0] == result["threshold_t1_tpr_0_9"]


def test_parent_retrieval_counts_reverse_descendants_as_negatives(monkeypatch):
    from evidence import pilot_analysis as analysis
    pairs = pd.DataFrame([dict(a="G", b="C", label="parent", source="lab"),
                          dict(a="C", b="P", label="parent", source="lab")])
    monkeypatch.setattr(analysis, "_score", lambda *args: 1.)
    result = analysis._false_matches(pairs, dict(G=object(), C=object(), P=object()), .5)
    # G is a false immediate-parent candidate for C. Two query rows => 1/2.
    assert result["false_matches_per_query"] == .5


def test_lineage_missing_mapping_population_survives_csv(tmp_path):
    names = ("p", "c", "i", "j", "k")
    pd.DataFrame([dict(model_id=n, kind="independent" if n in "ijk" else "derived",
                       path=n, base_family="same", license="mit") for n in names]).to_csv(tmp_path / "models.csv", index=False)
    pd.DataFrame([dict(child="c", parent="p", role="parent")]).to_csv(tmp_path / "edges.csv", index=False)
    (tmp_path / "reference_pairs.json").write_text(json.dumps({"pairs": [
        dict(model_ids=[a,b], a=f"lab/{a}@local:x", b=f"lab/{b}@local:x", label="independent", source="certified")
        for a,b in (("i","j"), ("i","k"), ("j","k"))]}))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for n in names[:-1]:
        (manifests / f"{n}.manifest.json").write_text(json.dumps(dict(
            repo_id=f"lab/{n}", revision="local:x", tensors=[dict(name="weight", shape=[512,64])], declared_bases=[])))
    pairs = lineage_pairs(tmp_path / "edges.csv", manifests)
    assert len(pairs) == 4
    path = tmp_path / "pairs.csv"
    pairs.to_csv(path, index=False)
    pairs = pd.read_csv(path, keep_default_na=False)
    result = tier_agreement(pairs, tmp_path / "t1", tmp_path / "t2")
    assert result["n_pairs_requested"] == 4 and result["n_missing_manifest_models"] == 1
    with pytest.raises(ValueError, match="mapping"):
        write_calibration(pairs, tmp_path / "t1", tmp_path / "null.json")


def test_default_t1_companions_supply_related_and_null_inputs(synthetic_run, tmp_path):
    for path in synthetic_run.t1.glob("*.fingerprint.safetensors"):
        path.unlink()
    result = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert result["n_missing_pairs"] == 0 and result["auroc_t1"] == 1.
    assert spectra_agreement(synthetic_run.t1, synthetic_run.t2)["measurement_coverage_complete"]
    write_calibration(synthetic_run.pairs, synthetic_run.t1, tmp_path / "null.json")
    assert json.loads((tmp_path / "null.json").read_text())["n_pairs"] == 3


def test_spectral_cohort_excludes_intentional_t1_only_census(synthetic_run, tmp_path):
    import torch
    from evidence.fingerprint import fingerprint_tensors
    fingerprint_tensors([("linear.weight", torch.ones(512,64))],
                        synthetic_run.t1 / "census.fingerprint.safetensors",
                        dict(repo_id="lab/census", revision="local:x"))
    result = spectra_agreement(synthetic_run.t1, synthetic_run.t2)
    assert result["n_models_requested"] == 9 and result["n_models_matched"] == 9
    assert result["measurement_coverage_complete"] and result["n_t1_only_models"] == 1


def test_known_empty_small_model_timing_cohort_is_not_applicable(synthetic_run, tmp_path):
    path = synthetic_run.t1 / "summary.json"
    summary = json.loads(path.read_text())
    for row in summary["models"]:
        row.update(stratum="text:9-35B", params=20_000_000_000)
    path.write_text(json.dumps(summary))
    write_g0_report(synthetic_run.root, tmp_path / "g0.md")
    text = (tmp_path / "g0.md").read_text()
    assert "N/A text:9-35B/elapsed_seconds_p90_le_9b" in text
    assert "FAIL text:9-35B/elapsed_seconds_p90_le_9b" not in text


def test_byte_curve_incomplete_height_is_unavailable(synthetic_run):
    next(synthetic_run.t2.glob("*.t1b128.fingerprint.safetensors")).unlink()
    result = tier_agreement(synthetic_run.pairs, synthetic_run.t1, synthetic_run.t2)
    assert result["byte_curve"][128] is None
    assert result["byte_curve_counts"][128]["missing"] > 0
    assert result["byte_curve_counts"][128]["requested"] == len(synthetic_run.pairs)
    assert result["byte_curve"][256] == 1.
