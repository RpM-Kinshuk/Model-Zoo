import math

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Iterator, List, Tuple, Optional


def matrix_rank_torch(svals: torch.Tensor, N: int, tol: Optional[float] = None) -> torch.Tensor:
    """Numerical rank from singular values; N is the larger matrix dimension."""
    if svals.numel() == 0:
        return svals.new_zeros((), dtype=torch.int64)
    if tol is None:
        tol = svals.max() * N * torch.finfo(svals.dtype).eps # type: ignore
    return torch.sum(svals > tol) # type: ignore


def matrix_entropy_torch(evals: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    """Shannon entropy normalized by log(rank), using the largest rank eigenvalues.

    Numerically discarded eigenvalues carry no probability mass. Rank one has
    entropy zero by convention; zero rank or non-finite spectra are undefined.
    This measures evenness among numerically active directions, and can jump
    when a tiny singular value crosses the numerical-rank threshold.
    """
    undefined = evals.new_tensor(float("nan"), dtype=torch.float64)
    if not torch.isfinite(evals).all():
        return undefined
    positive = evals[evals > 0].double()
    numerical_rank = min(int(rank.item()), positive.numel())
    if numerical_rank <= 0:
        return undefined
    if numerical_rank == 1:
        return positive.new_tensor(0.0)
    if numerical_rank < positive.numel():
        positive = torch.topk(positive, numerical_rank, sorted=False).values

    # Scaling first also avoids overflow when summing very large eigenvalues.
    scaled = positive / positive.max()
    p = scaled / scaled.sum()
    return -torch.special.xlogy(p, p).sum() / math.log(numerical_rank)


def iter_eligible_layers(
    net: nn.Module,
    filter_type: Optional[bool] = True,
    coverage: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Tuple[str, torch.Tensor, int]]:
    """Yield (name, weight_tensor, params_count) for eligible layers.

    Supported layouts are dense Linear/Embedding matrices and Conv1d/2d/3d
    kernels, including subclasses and the declared Hugging Face Conv1D matrix
    layout. With filter_type=False, other ordinary dense 2D weights are an
    explicit opt-in; unknown higher-dimensional or quantized layouts are never
    guessed. Each eligible matrix is measured whole, including fused attention
    projections: names and aspect ratios do not establish a QKV packing layout.
    The legacy Linear aspect-ratio filter remains enabled.

    When provided, coverage receives one record per weight-bearing module,
    including skipped candidates. Known packed attributes and direct matrix
    parameters with nonstandard names are recorded, but never unpacked or
    interpreted as supported layouts. Packed recurrent weights are recorded at
    their owning module, not again at each storage helper. Arbitrary buffers
    are not weight candidates. The estimator changes eligible records to
    analyzed after computation; absence of a power-law fit is not a skip.
    """
    emitted_names = set()
    for name, module in net.named_modules():
        weight_attribute = "weight"
        unsupported_attributes = []
        if not hasattr(module, "weight"):
            unsupported_attributes = [
                attribute for attribute in ("qweight", "weight_packed", "packed_weight")
                if getattr(module, attribute, None) is not None
            ]
            # These declared recurrent modules store learned weights only in
            # packed helpers, with no Tensor weight or named parameters. Keep
            # their missingness visible without calling get_weight/unpacking,
            # or counting each private storage helper as another model layer.
            if not unsupported_attributes and isinstance(
                module, (nn.quantized.dynamic.LSTM, nn.quantized.dynamic.GRU)
            ) and getattr(module, "_all_weight_values", None) is not None:
                unsupported_attributes = ["_all_weight_values"]
            if not unsupported_attributes:
                unsupported_attributes = [
                    attribute for attribute, parameter in module.named_parameters(recurse=False)
                    if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                    and parameter.ndim >= 2
                ]
            if not unsupported_attributes:
                continue
            weight_attribute = unsupported_attributes[0]
        weight_param = getattr(module, weight_attribute, None)
        module_class = type(module)
        record: Dict[str, Any] = {
            "module_name": name,
            "module_type": f"{module_class.__module__}.{module_class.__qualname__}",
            "weight_attribute": weight_attribute,
            "weight_shape": None,
            "weight_dtype": str(getattr(weight_param, "dtype", "unknown")).removeprefix("torch."),
            "weight_layout": "unknown",
            "status": "skipped",
            "reason": "",
            "measurement_names": [],
            "measurement_slices": [],
        }
        if coverage is not None:
            coverage.append(record)
        if unsupported_attributes:
            record["unsupported_weight_attributes"] = unsupported_attributes
            if isinstance(weight_param, torch.Tensor) and not isinstance(
                weight_param, torch.nn.parameter.UninitializedParameter
            ):
                record["weight_shape"] = list(weight_param.shape)
            record["reason"] = "unsupported_weight_attribute"
            continue
        if isinstance(weight_param, torch.nn.parameter.UninitializedParameter):
            record["reason"] = "uninitialized_weight"
            continue
        if not isinstance(weight_param, torch.Tensor):
            record["reason"] = "weight_is_not_a_tensor"
            continue
        record["weight_shape"] = list(weight_param.shape)
        # Quantized tensor subclasses can advertise a floating dtype while
        # storing a packed representation. Do not interpret their raw storage.
        if type(weight_param) not in (torch.Tensor, torch.nn.Parameter) or weight_param.is_quantized:
            record["reason"] = "unsupported_weight_representation"
            continue
        if not weight_param.is_floating_point():
            record["reason"] = "non_floating_weight"
            continue
        if weight_param.layout != torch.strided:
            record["reason"] = "non_dense_weight"
            continue
        if weight_param.device.type == "meta":
            record["reason"] = "meta_weight"
            continue
        weight: torch.Tensor = weight_param.detach()
        if weight.ndim <= 1:
            record["reason"] = "weight_has_fewer_than_two_dimensions"
            continue
        if weight.numel() == 0:
            record["reason"] = "empty_weight"
            continue

        # HF's historically named Conv1D is a transposed dense projection, not
        # a torch Conv1d kernel. Match its declaration without importing HF.
        hf_conv1d = any(
            base.__name__ == "Conv1D" and base.__module__ == "transformers.pytorch_utils"
            for base in module_class.__mro__
        )
        if isinstance(module, (nn.Linear, nn.Embedding)) or hf_conv1d:
            expected_ndim, layout = 2, "matrix"
        elif isinstance(module, nn.Conv1d):
            expected_ndim, layout = 3, "conv1d_slices"
        elif isinstance(module, nn.Conv2d):
            expected_ndim, layout = 4, "conv2d_slices"
        elif isinstance(module, nn.Conv3d):
            expected_ndim, layout = 5, "conv3d_slices"
        elif not filter_type and weight.ndim == 2:
            expected_ndim, layout = 2, "matrix"
        else:
            record["reason"] = "unsupported_module_type"
            continue
        record["weight_layout"] = layout
        if weight.ndim != expected_ndim:
            record["reason"] = "unsupported_weight_shape"
            continue

        if filter_type and isinstance(module, nn.Linear):
            mx, mn = max(weight.shape), min(weight.shape)
            if (mx / mn) >= 8:
                record["reason"] = "linear_aspect_ratio_at_least_8"
                continue

        bias = getattr(module, "bias", None)
        bias_params = bias.numel() if isinstance(bias, torch.Tensor) and bias.requires_grad else 0
        if name in emitted_names:
            raise ValueError(f"Duplicate ESD measurement name: {name!r}")
        emitted_names.add(name)
        record["status"] = "eligible"
        record["measurement_names"] = [name]
        record["measurement_slices"] = [""]
        yield name, weight, weight.numel() + bias_params


def estimate_compute_cost(weight: torch.Tensor, use_svd: bool) -> int:
    """Estimate dense eigenspectrum compute cost for scheduling.

    Using Gram-based path in `_squared_singular_values`:
      - For a 2D matrix A in R^{M x N}, we form G of size min(M,N) and run eigvalsh.
        Cost ~ O(min(M,N)^2 * max(M,N)) to build G + O(min(M,N)^3) for eigvalsh.
      - For Conv weights (O, I, *kernel) we create B = prod(kernel) slices of (O x I) matrices.
        Total cost ~ B * [min(O,I)^2 * max(O,I) + min(O,I)^3].
    Using SVD path:
        - For a 2D matrix A in R^{M x N}, cost ~ O(M*N^2) if M <= N else O(N*M^2).
        - For Conv weights, total cost ~ B * [M*N^2 if M <= N else N*M^2].
    We return an integer proxy of the above expression.
    """
    if weight.ndim == 2:
        M, N = int(weight.shape[0]), int(weight.shape[1])
        m, n = (M, N) if M <= N else (N, M)
        if use_svd:
            return m*n*n
        return m*m*n + m*m*m
    elif weight.ndim > 2:
        O, I = int(weight.shape[0]), int(weight.shape[1])
        B = int(np.prod(weight.shape[2:]))
        m, n = (O, I) if O <= I else (I, O)
        if use_svd:
            return B * (m*n*n)
        return B * (m*m*n + m*m*m)
    return int(weight.numel())
