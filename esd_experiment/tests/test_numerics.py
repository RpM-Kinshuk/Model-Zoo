"""Numerical regression checks using small CPU tensors; no model downloads."""

import math
import json
from types import SimpleNamespace

import pytest
import torch

from net_esd.core import compute_esd_for_weight
from net_esd.utils import matrix_entropy_torch, matrix_rank_torch


@pytest.mark.parametrize("is_cuda,expected_kwargs", [(False, {}), (True, {"driver": "gesvd"})])
def test_svd_driver_is_precision_focused_only_on_cuda(monkeypatch, is_cuda, expected_kwargs):
    from net_esd.core import _svdvals_for_measurement
    matrix = SimpleNamespace(is_cuda=is_cuda)
    expected = torch.tensor([2., 1.])

    def svdvals(tensor, **kwargs):
        assert tensor is matrix
        assert kwargs == expected_kwargs
        return expected

    monkeypatch.setattr(torch.linalg, "svdvals", svdvals)
    assert _svdvals_for_measurement(matrix) is expected


@pytest.mark.parametrize("shape", [(4, 3), (2, 4, 3)])
def test_failed_gram_uses_the_same_svd_policy(monkeypatch, shape):
    import net_esd.core as core
    matrix = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    expected = core.squared_singular_values(matrix)
    called = []

    def fail_gram(*args, **kwargs):
        raise RuntimeError("Synthetic eigvalsh convergence failure")

    def svdvals(tensor):
        called.append(tensor)
        return torch.linalg.svdvals(tensor)

    monkeypatch.setattr(torch.linalg, "eigvalsh", fail_gram)
    monkeypatch.setattr(core, "_svdvals_for_measurement", svdvals)
    torch.testing.assert_close(core.squared_singular_values(matrix, use_svd=False), expected)
    assert len(called) == 1 and called[0] is matrix


def analyze(weight, *, method="xmin_mid", filter_zeros=True, use_svd=True, compute_dtype="float32"):
    return compute_esd_for_weight(
        "model.layers.0.proj", weight, 1e-5, 100, method, 2, 0.5,
        filter_zeros, use_svd, True, None, weight.numel(), compute_dtype,
    )


def reference_tail_fit(eigenvalues, xmin):
    """Scalar Pareto MLE and full KS, independent of the vectorized fitter."""
    xmin = float(xmin)
    tail = sorted(float(x) for x in eigenvalues if x >= xmin)
    n = len(tail)
    log_ratios = [math.log1p((x - xmin) / xmin) for x in tail]
    alpha = 1 + n / sum(log_ratios)
    cdf = [-math.expm1((1 - alpha) * log_ratio) for log_ratio in log_ratios]
    D = max(
        max((i + 1) / n - f for i, f in enumerate(cdf)),
        max(f - i / n for i, f in enumerate(cdf)),
    )
    return alpha, D, n


def test_uniform_spectrum_has_unit_normalized_entropy():
    entropy = matrix_entropy_torch(torch.ones(4096), torch.tensor(4096))

    assert entropy.item() == pytest.approx(1.0, abs=1e-12)


def test_entropy_matches_known_probability_distribution():
    eigenvalues = torch.tensor([0.0, 1.0, 3.0])
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(2)

    assert matrix_entropy_torch(eigenvalues, torch.tensor(2)).item() == pytest.approx(expected)


def test_entropy_uses_only_the_numerically_retained_rank():
    eigenvalues = torch.tensor([1e-8, 3.0, 1.0])
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(2)

    assert matrix_entropy_torch(eigenvalues, torch.tensor(2)).item() == pytest.approx(expected)


def test_rank_one_entropy_is_zero_and_zero_rank_is_undefined():
    assert matrix_entropy_torch(torch.tensor([0.0, 4.0]), torch.tensor(1)).item() == 0.0
    assert math.isnan(matrix_entropy_torch(torch.zeros(4), torch.tensor(0)).item())


@pytest.mark.parametrize("method", ["xmin_mid", "xmin_peak", None])
@pytest.mark.parametrize("scale", [0.0, 1.0, 2.0])
def test_constant_spectrum_has_no_power_law_fit(method, scale):
    result = analyze(torch.eye(4) * scale, method=method, filter_zeros=False)

    assert result is not None
    for key in ("alpha", "D", "alpha_weighted", "log_alpha_norm", "fit_xmin"):
        assert math.isnan(result[key]), key
    assert result["n_tail"] == 0
    assert result["num_evals"] == 4
    assert len(result["eigs"]) == 4


@pytest.mark.parametrize("use_svd", [True, False])
def test_midpoint_fit_matches_analytic_solution(use_svd):
    result = analyze(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])), use_svd=use_svd)

    expected_alpha = 1 + 2 / math.log(16 / 9)
    assert result["alpha"] == pytest.approx(expected_alpha)
    assert result["D"] == pytest.approx(0.5)
    assert result["xmin"] == 1.0  # Retained spectrum minimum, for compatibility.
    assert result["fit_xmin"] == 9.0
    assert result["n_tail"] == 2


@pytest.mark.parametrize("use_svd", [True, False])
@pytest.mark.parametrize("eigenvalues", [
    [1., 2., 3., 4., 6.],
    [1., 2., 2., 2., 4., 5., 10.],
])
def test_dks_selects_cutoff_using_full_ks(eigenvalues, use_svd):
    result = analyze(torch.diag(torch.tensor(eigenvalues).sqrt()), method=None, use_svd=use_svd)

    spectrum = result["eigs"].tolist()
    candidates = [
        (xmin, *reference_tail_fit(spectrum, xmin))
        for xmin in sorted(set(spectrum))[:-1]
    ]
    xmin, alpha, D, n = min(candidates, key=lambda row: row[2])
    assert result["fit_xmin"] == pytest.approx(xmin)
    assert result["alpha"] == pytest.approx(alpha)
    assert result["D"] == pytest.approx(D, abs=1e-6)
    assert result["n_tail"] == n


@pytest.mark.parametrize("method", ["xmin_mid", "xmin_peak", None])
def test_exported_cutoff_reconstructs_fit_including_tied_eigenvalues(method):
    result = analyze(torch.diag(torch.tensor([1., 2., 2., 2., 4., 8.]).sqrt()), method=method)

    alpha, D, n = reference_tail_fit(result["eigs"], result["fit_xmin"])
    assert result["alpha"] == pytest.approx(alpha)
    assert result["D"] == pytest.approx(D, abs=1e-6)
    assert result["n_tail"] == n
    if method == "xmin_mid":
        assert result["fit_xmin"] == pytest.approx(2.)
        assert result["n_tail"] == 5


@pytest.mark.parametrize("use_svd", [True, False])
def test_numerical_rank_and_entropy_are_transpose_invariant(use_svd):
    weight = torch.zeros((8, 2))
    weight[0, 0], weight[1, 1] = 10000., 0.005

    tall = analyze(weight, use_svd=use_svd)
    wide = analyze(weight.T, use_svd=use_svd)

    assert tall["matrix_rank"] == wide["matrix_rank"] == torch.linalg.matrix_rank(weight).item()
    assert tall["entropy"] == wide["entropy"] == 0.0


@pytest.mark.parametrize("diagonal", [[2.], [0., 2., 0.]])
@pytest.mark.parametrize("filter_zeros", [True, False])
@pytest.mark.parametrize("use_svd", [True, False])
def test_rank_one_layer_keeps_spectrum_and_nonfit_metrics(diagonal, filter_zeros, use_svd):
    result = analyze(torch.diag(torch.tensor(diagonal)), filter_zeros=filter_zeros, use_svd=use_svd)

    assert result is not None
    assert result["matrix_rank"] == 1
    assert result["entropy"] == 0.0
    assert result["norm"] == result["spectral_norm"] == 4.0
    assert result["stable_rank"] == 1.0
    assert result["log_norm"] == result["log_spectral_norm"] == math.log10(4.)
    assert result["num_evals"] == (1 if filter_zeros else len(diagonal))
    for key in ("alpha", "D", "alpha_weighted", "log_alpha_norm", "fit_xmin"):
        assert math.isnan(result[key]), key
    assert result["n_tail"] == 0
    assert result["eigs"].tolist() == sorted(x*x for x in diagonal)
    assert result["raw_num_evals"] == len(diagonal)
    assert result["fit_status"] == "insufficient_positive_eigenvalues"


@pytest.mark.parametrize("method", ["xmin_mid", "xmin_peak", None])
@pytest.mark.parametrize("filter_zeros", [True, False])
def test_zero_matrix_keeps_layer_with_explicitly_undefined_metrics(method, filter_zeros):
    result = analyze(torch.zeros((4, 4)), method=method, filter_zeros=filter_zeros)

    assert result is not None
    assert result["matrix_rank"] == 0
    assert result["norm"] == result["spectral_norm"] == 0.0
    for key in ("alpha", "D", "entropy", "log_norm", "log_spectral_norm", "stable_rank", "fit_xmin"):
        assert math.isnan(result[key]), key
    assert result["n_tail"] == 0
    assert result["num_evals"] == (0 if filter_zeros else 4)
    assert result["eigs"].tolist() == [0.] * 4
    assert result["raw_num_evals"] == 4
    if filter_zeros:
        assert math.isnan(result["xmin"])
        assert math.isnan(result["xmax"])
    else:
        assert result["xmin"] == result["xmax"] == 0.0


def test_filter_does_not_restore_values_when_every_eigenvalue_is_below_threshold():
    result = analyze(torch.diag(torch.tensor([1e-4, 2e-4])))

    assert result is not None
    assert result["num_evals"] == result["matrix_rank"] == 0
    assert len(result["eigs"]) == result["raw_num_evals"] == 2
    assert result["raw_norm"] == pytest.approx(5e-8)
    assert result["raw_matrix_rank"] == 2
    assert result["fit_status"] == "no_retained_eigenvalues"
    assert math.isnan(result["alpha"])
    assert matrix_rank_torch(torch.empty(0), 2).item() == 0


def test_log_alpha_norm_remains_finite_for_large_weights():
    scale = 1e10
    result = analyze(torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])) * scale)

    alpha = 1 + 2 / math.log(16 / 9)
    expected = alpha * math.log10(16 * scale**2) + math.log10(
        sum((value / 16)**alpha for value in (1, 4, 9, 16))
    )
    assert math.isfinite(result["log_alpha_norm"])
    assert result["log_alpha_norm"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("method", ["xmin_mid", "xmin_peak", None])
def test_zero_eigenvalues_do_not_break_fitting_or_disappear_when_unfiltered(method):
    result = analyze(torch.diag(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])), method=method, filter_zeros=False)

    assert math.isfinite(result["alpha"])
    assert result["alpha"] > 1
    assert result["num_evals"] == 5
    assert result["eigs"][0] == 0.0
    assert 0 <= result["entropy"] <= 1


@pytest.mark.parametrize("method", ["xmin_mid", "xmin_peak", None])
@pytest.mark.parametrize("use_svd", [True, False])
def test_near_constant_fit_has_float64_ks_precision(method, use_svd):
    weight = torch.diag(0.84 + torch.arange(8, dtype=torch.float32) * 1e-7)
    result = analyze(weight, method=method, use_svd=use_svd)

    alpha, D, n = reference_tail_fit(result["eigs"], result["fit_xmin"])
    assert result["alpha"] == pytest.approx(alpha, rel=1e-8)
    assert result["D"] == pytest.approx(D, abs=1e-8)
    assert result["n_tail"] == n
    assert result["fit_status"] == "fitted"


@pytest.mark.parametrize("scale", [1e-150, 1e-8, 1., 1e8, 1e150])
@pytest.mark.parametrize("method", [None, "xmin_mid", "xmin_peak"])
@pytest.mark.parametrize("use_svd", [True, False])
def test_float64_narrow_spectra_retain_fit_precision_away_from_unit_scale(scale, method, use_svd):
    weight = torch.diag(scale * (1 + torch.arange(8, dtype=torch.float64) * 1e-15))
    result = analyze(weight, method=method, use_svd=use_svd,
                     filter_zeros=False, compute_dtype="float64")

    assert result["fit_status"] == "fitted"
    alpha, D, n = reference_tail_fit(result["eigs"], result["fit_xmin"])
    assert result["alpha"] == pytest.approx(alpha, rel=2e-12)
    assert result["D"] == pytest.approx(D, abs=2e-12)
    assert result["n_tail"] == n


def test_float64_tail_ratios_do_not_overflow_for_extreme_spectral_range():
    from net_esd.core import _fit_tail_chunk

    eigs = torch.tensor([1e-300, 1e-200, 1., 1e200, 1e300], dtype=torch.float64)
    alphas, distances = _fit_tail_chunk(eigs, torch.tensor([0, 1]))

    for index, start in enumerate([0, 1]):
        tail = eigs[start:].tolist()
        n = len(tail)
        log_ratios = [math.log(value) - math.log(tail[0]) for value in tail]
        alpha = 1 + n / sum(log_ratios)
        cdf = [-math.expm1((1 - alpha) * log_ratio) for log_ratio in log_ratios]
        D = max(max((i + 1) / n - f for i, f in enumerate(cdf)),
                max(f - i / n for i, f in enumerate(cdf)))
        assert alphas[index].item() == pytest.approx(alpha, rel=1e-12)
        assert distances[index].item() == pytest.approx(D, abs=1e-12)


@pytest.mark.parametrize("method", [None, "xmin_peak"])
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_cutoff_chunking_preserves_fit_and_bounds_candidate_workspace(monkeypatch, method, chunk_size):
    import net_esd.core as core

    weight = torch.diag(torch.linspace(1, 10, 97))
    expected = analyze(weight, method=method)
    original_fit = core._fit_tail_chunk
    seen = []

    def checked_fit(eigs, starts):
        assert eigs.dtype == torch.float64
        assert starts.numel() <= chunk_size
        seen.extend(starts.tolist())
        return original_fit(eigs, starts)

    monkeypatch.setattr(core, "_KS_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(core, "_fit_tail_chunk", checked_fit)
    result = analyze(weight, method=method)

    for key in ("alpha", "D", "fit_xmin", "n_tail"):
        assert result[key] == pytest.approx(expected[key], abs=1e-12)
    assert seen == sorted(set(seen))
    if method is None:
        assert seen == list(range(96))
        candidates = [
            (xmin, *reference_tail_fit(result["eigs"], xmin))
            for xmin in result["eigs"][:-1]
        ]
        xmin, alpha, D, n = min(candidates, key=lambda row: row[2])
        assert result["fit_xmin"] == xmin
        assert result["alpha"] == pytest.approx(alpha)
        assert result["D"] == pytest.approx(D, abs=1e-12)
        assert result["n_tail"] == n


@pytest.mark.parametrize("method", [None, "xmin_mid", "xmin_peak"])
def test_repeated_cutoffs_always_include_all_ties(monkeypatch, method):
    import net_esd.core as core

    monkeypatch.setattr(core, "_KS_CHUNK_SIZE", 2)
    weight = torch.diag(torch.tensor([1., 1., 1., 2., 2., 2., 2., 4., 4., 8.]))
    result = analyze(weight, method=method)
    alpha, D, n = reference_tail_fit(result["eigs"], result["fit_xmin"])

    assert result["alpha"] == pytest.approx(alpha)
    assert result["D"] == pytest.approx(D, abs=1e-12)
    assert result["n_tail"] == n


def test_raw_measurements_and_storage_are_independent_of_absolute_filter():
    weight = torch.diag(torch.tensor([.001, .002, .003, .004]))
    filtered = analyze(weight)
    unfiltered = analyze(weight, filter_zeros=False)
    scaled = analyze(weight * 10)

    assert filtered["num_evals"] == 1
    assert unfiltered["num_evals"] == scaled["num_evals"] == 4
    assert math.isnan(filtered["alpha"])
    assert unfiltered["alpha"] == pytest.approx(scaled["alpha"], rel=1e-5)
    assert filtered["raw_norm"] == pytest.approx(3e-5)
    assert filtered["raw_norm"] == unfiltered["norm"]
    for key in ("raw_norm", "raw_spectral_norm", "raw_num_evals", "raw_matrix_rank", "raw_entropy"):
        assert filtered[key] == unfiltered[key]
    assert filtered["eigs"].tolist() == unfiltered["eigs"].tolist()
    assert filtered["filter_zeros"] is True
    assert unfiltered["filter_zeros"] is False
    assert filtered["evals_thresh"] == 1e-5


def test_rank_entropy_boundary_is_explicit_even_with_default_filter():
    weight = torch.zeros((8, 3))
    weight[0, 0] = weight[1, 1] = 10000.
    threshold = 10000. * 8 * torch.finfo(torch.float32).eps
    weight[2, 2] = threshold * .99
    below = analyze(weight)
    weight[2, 2] = threshold * 1.01
    above = analyze(weight)

    assert below["num_evals"] == above["num_evals"] == 3
    assert below["matrix_rank"] == 2
    assert above["matrix_rank"] == 3
    assert below["entropy"] == pytest.approx(1.)
    assert above["entropy"] == pytest.approx(math.log(2) / math.log(3))
    assert below["raw_norm"] == pytest.approx(above["raw_norm"], rel=1e-12)


@pytest.mark.parametrize("kernel_shape", [(3,), (2, 3), (2, 2, 3)])
@pytest.mark.parametrize("use_svd", [True, False])
def test_standard_convolution_layouts_match_explicit_kernel_slices(kernel_shape, use_svd):
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn((4, 3, *kernel_shape), generator=generator)
    original = weight.clone()
    result = analyze(weight, use_svd=use_svd, compute_dtype="float64")
    slices = weight.double().reshape(4, 3, -1).permute(2, 0, 1) * math.sqrt(.5)
    expected = torch.linalg.svdvals(slices).square().flatten().sort().values

    torch.testing.assert_close(torch.from_numpy(result["eigs"]), expected)
    torch.testing.assert_close(weight, original, rtol=0, atol=0)
    assert result["raw_num_evals"] == math.prod(kernel_shape) * 3
    assert result["weight_layout"] == f"conv{len(kernel_shape)}d_slices"
    assert result["raw_norm"] == pytest.approx(expected.sum().item())


@pytest.mark.parametrize("kernel_shape", [(1,), (1, 1), (1, 1, 1)])
def test_singleton_convolution_normalization_never_mutates_input(kernel_shape):
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3, *kernel_shape)
    original = weight.clone()

    analyze(weight)

    torch.testing.assert_close(weight, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(4,), (2, 2, 2, 2, 2, 2)])
def test_undeclared_weight_layouts_are_not_silently_flattened(shape):
    assert analyze(torch.ones(shape)) is None


def test_compute_precision_is_explicit_and_preserves_available_source_precision():
    weight = torch.diag(1 + torch.arange(4, dtype=torch.float64) * 1e-9)
    default = analyze(weight)
    accurate = analyze(weight, compute_dtype="float64")

    assert default["source_dtype"] == accurate["source_dtype"] == "float64"
    assert default["compute_dtype"] == "float32"
    assert accurate["compute_dtype"] == "float64"
    assert default["fit_status"] == "constant_spectrum"
    assert accurate["fit_status"] == "fitted"
    assert math.isfinite(accurate["alpha"])
    assert str(accurate["eigs"].dtype) == "float64"
    assert accurate["weight_layout"] == "matrix"


def test_higher_compute_precision_cannot_recover_a_float16_loading_round_trip():
    original = torch.diag(torch.tensor([1., 1.0001, 1.0002, 1.0003]))
    native = analyze(original, compute_dtype="float64")
    rounded = analyze(original.half(), compute_dtype="float64")

    assert native["fit_status"] == "fitted"
    assert rounded["source_dtype"] == "float16"
    assert rounded["fit_status"] == "constant_spectrum"


def test_unsupported_compute_dtype_is_rejected():
    with pytest.raises(ValueError, match="compute_dtype"):
        analyze(torch.eye(4), compute_dtype="float16")


@pytest.mark.parametrize("with_fit", [True, False])
def test_worker_saves_real_cpu_spectra_with_missing_fits(tmp_path, monkeypatch, with_fit):
    from net_esd import net_esd_estimator
    from esd_experiment.tests.test_worker import load_worker_module

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([
        torch.nn.ModuleDict({"proj": torch.nn.Linear(4, 4, bias=False)})
        for _ in range(3)
    ])
    with torch.no_grad():
        model.layers[0]["proj"].weight.copy_(
            torch.diag(torch.tensor([1., 2., 3., 4.])) if with_fit else torch.eye(4)
        )
        model.layers[1]["proj"].weight.copy_(torch.diag(torch.tensor([0., 2., 0., 0.])))
        model.layers[2]["proj"].weight.zero_()

    worker = load_worker_module()
    args = SimpleNamespace(
        model_id="test/local-model", revision="", base_model_relation="", source_model="",
        loader_scenario="", primary_type_bucket="", output_dir=str(tmp_path), overwrite=False,
        fix_fingers="xmin_mid", evals_thresh=1e-5, bins=100, filter_zeros=True,
        parallel_esd=False, use_svd=True, save_eigs=True, device_map="cpu", max_retries=0,
    )
    # Only model acquisition is stubbed; computation and artifact writes are real.
    monkeypatch.setattr(worker, "parse_args", lambda: args)
    monkeypatch.setattr(worker, "load_model", lambda **kwargs: (model, False))
    monkeypatch.setattr(worker, "net_esd_estimator", net_esd_estimator)

    assert worker.main() == 0

    saved = worker.pd.read_csv(tmp_path / "stats" / "test--local-model.csv")
    assert saved["longname"].tolist() == [f"layers.{i}.proj" for i in range(3)]
    assert saved["num_evals"].tolist() == [4, 1, 0]
    assert saved["n_tail"].tolist() == [2 if with_fit else 0, 0, 0]
    assert saved["alpha"].notna().sum() == int(with_fit)
    if with_fit:
        assert saved.loc[0, "fit_xmin"] == 9.0
        assert saved.loc[0, "D"] == pytest.approx(0.5)
    else:
        assert saved["fit_xmin"].isna().all()
    with worker.h5py.File(tmp_path / "metrics" / "test--local-model.h5", "r") as h5:
        assert h5.attrs["numerics_version"] == worker.NUMERICS_VERSION
        assert h5["alpha"].shape == (3, 1)
        worker.np.testing.assert_allclose(h5["alpha"][:, 0], saved["alpha"], equal_nan=True)
        assert h5["eigs"][1].tolist() == [0.0, 0.0, 0.0, 4.0]
        assert h5["eigs"][2].tolist() == [0.0] * 4
    status = json.loads((tmp_path / "logs" / "terminal_status" / "test--local-model.json").read_text())
    assert status["status"] == "success"
    assert not (tmp_path / "logs" / "failed_models.txt").exists()
