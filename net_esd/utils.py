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


def _weight_record(name, module, attribute, weight):
    module_class = type(module)
    return {
        "module_name": name,
        "module_type": f"{module_class.__module__}.{module_class.__qualname__}",
        "weight_attribute": attribute,
        "weight_shape": list(weight.shape) if isinstance(weight, torch.Tensor) and not nn.parameter.is_lazy(weight) else None,
        "weight_dtype": str(getattr(weight, "dtype", "unknown")).removeprefix("torch."),
        "weight_layout": "unknown", "status": "skipped", "reason": "",
        "measurement_names": [], "measurement_slices": [],
    }


def _weight_skip_reason(weight):
    """Common materialization/representation checks, before interpreting layout."""
    if nn.parameter.is_lazy(weight):
        return "uninitialized_weight"
    if not isinstance(weight, torch.Tensor):
        return "weight_is_not_a_tensor"
    # Tensor subclasses can advertise floating dtype while storing packed data.
    if type(weight) not in (torch.Tensor, nn.Parameter) or weight.is_quantized:
        return "unsupported_weight_representation"
    if not weight.is_floating_point():
        return "non_floating_weight"
    if weight.layout != torch.strided:
        return "non_dense_weight"
    if weight.device.type == "meta":
        return "meta_weight"
    if weight.ndim <= 1:
        return "weight_has_fewer_than_two_dimensions"
    if weight.numel() == 0:
        return "empty_weight"
    return ""


def iter_eligible_layers(
    net: nn.Module,
    filter_type: Optional[bool] = True,
    coverage: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Tuple[str, torch.Tensor, int]]:
    """Yield (name, weight_tensor, params_count) for eligible layers.

    Supported layouts are dense Linear/Embedding matrices and Conv1d/2d/3d
    kernels, including subclasses and the declared Hugging Face Conv1D matrix
    layout, plus the built-in MultiheadAttention's declared projection matrices.
    With filter_type=False, other ordinary dense 2D weights are an
    explicit opt-in; unknown higher-dimensional or quantized layouts are never
    guessed. Each eligible matrix is measured whole, including fused attention
    projections: names and aspect ratios do not establish a QKV packing layout.
    The legacy Linear aspect-ratio filter remains enabled.

    When provided, coverage receives records identified by module and weight
    attribute, including skipped candidates. MultiheadAttention can have several
    weight records in one module; module counts must group by module_name.
    Known packed attributes and other direct matrix
    parameters with nonstandard names are recorded, but never unpacked or
    interpreted as supported layouts. Packed recurrent weights are recorded at
    their owning module, not again at each storage helper. Arbitrary buffers
    are not weight candidates. The estimator changes eligible records to
    analyzed after computation; absence of a power-law fit is not a skip.
    """
    emitted_names = set()
    for name, module in net.named_modules():
        # Only this declared implementation is covered. Quantizable/custom MHA
        # subclasses can replace the forward projections but retain unused
        # dense parameters from the base class; do not infer their layout.
        if type(module) is nn.MultiheadAttention:
            if module._qkv_same_embed_dim:
                shapes = {"in_proj_weight": (3 * module.embed_dim, module.embed_dim)}
            else:
                shapes = {"q_proj_weight": (module.embed_dim, module.embed_dim),
                          "k_proj_weight": (module.embed_dim, module.kdim),
                          "v_proj_weight": (module.embed_dim, module.vdim)}
            for attribute, shape in shapes.items():
                weight = getattr(module, attribute, None)
                record = _weight_record(name, module, attribute, weight)
                record["weight_layout"] = "matrix"
                if coverage is not None:
                    coverage.append(record)
                record["reason"] = _weight_skip_reason(weight)
                if not record["reason"] and tuple(weight.shape) != shape:
                    record["reason"] = "unsupported_weight_shape"
                if record["reason"]:
                    continue
                measurement_name = f"{name}.{attribute}" if name else attribute
                if measurement_name in emitted_names:
                    raise ValueError(f"Duplicate ESD measurement name: {measurement_name!r}")
                emitted_names.add(measurement_name)
                record.update(status="eligible", measurement_names=[measurement_name], measurement_slices=[""])
                bias = module.in_proj_bias
                bias_params = bias.numel() if isinstance(bias, torch.Tensor) and bias.requires_grad else 0
                if attribute != "in_proj_weight":
                    bias_params //= 3  # One declared bias vector per Q/K/V projection.
                yield measurement_name, weight.detach(), weight.numel() + bias_params
            continue  # out_proj is visited independently through named_modules.
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
        record = _weight_record(name, module, weight_attribute, weight_param)
        if coverage is not None:
            coverage.append(record)
        if unsupported_attributes:
            record["unsupported_weight_attributes"] = unsupported_attributes
            record["reason"] = "unsupported_weight_attribute"
            continue
        record["reason"] = _weight_skip_reason(weight_param)
        if record["reason"]:
            continue
        weight: torch.Tensor = weight_param.detach()

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


def weight_usage_report(net: nn.Module, coverage, measurement_names):
    """Link loaded registered tensors (including aliases) to saved measurements.

    This reads names and metadata, not tensor values or state_dict copies. It
    complements the loader's checkpoint-integrity check; it cannot establish
    what was discarded before loading. Sharing means the same Tensor object,
    not equal values or overlapping storage. Unrecognized buffers are listed
    separately from unaccounted-for parameters, without guessing their purpose.
    """
    by_tensor_id = {}
    by_name = {}
    for kind, get_named_tensors in (("parameter", net.named_parameters), ("buffer", net.named_buffers)):
        for name, tensor in get_named_tensors(remove_duplicate=False):
            if id(tensor) in by_tensor_id:
                record = by_tensor_id[id(tensor)]
                record["aliases"].append(name)
            else:
                shape = None if nn.parameter.is_lazy(tensor) else list(tensor.shape)
                if kind == "buffer":
                    status, reason = "not_applicable", "buffer_not_selected_as_weight"
                elif shape is not None and len(shape) < 2:
                    status, reason = "not_applicable", "non_matrix_parameter"
                else:
                    status, reason = "unresolved", "parameter_not_in_module_coverage"
                record = {
                    "name": name, "aliases": [], "kind": kind, "shape": shape,
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "status": status, "reason": reason, "measurement_names": [],
                }
                by_tensor_id[id(tensor)] = record
            by_name[name] = record

    saved_names = set(measurement_names)
    mapped_names = set()
    for module_record in coverage:
        attributes = module_record.get("unsupported_weight_attributes") or [module_record["weight_attribute"]]
        for attribute in attributes:
            name = ".".join(part for part in (module_record["module_name"], attribute) if part)
            record = by_name.get(name)
            if record is None:
                # Callable/packed helpers or computed properties may not be
                # registered tensors. Keep their original module coverage;
                # any computed spectra stay explicitly unmapped below.
                continue
            measured = saved_names.intersection(module_record["measurement_names"])
            if measured:
                record["measurement_names"] = sorted(set(record["measurement_names"]) | measured)
                record["status"], record["reason"] = "measured", ""
                mapped_names.update(measured)
            elif record["status"] != "measured":
                if module_record["reason"] == "weight_has_fewer_than_two_dimensions":
                    continue
                record["status"] = "skipped" if module_record["status"] == "skipped" else "unresolved"
                record["reason"] = module_record["reason"] or "selected_weight_without_measurement"

    records = list(by_tensor_id.values())
    unmapped = sorted(saved_names - mapped_names)
    counts = {f"{status}_tensors": sum(record["status"] == status for record in records)
              for status in ("measured", "skipped", "not_applicable", "unresolved")}
    counts.update(registered_tensors=len(records),
                  shared_tensors=sum(bool(record["aliases"]) for record in records),
                  unmapped_measurements=len(unmapped))
    return {"scope": "loaded_registered_tensors", "counts": counts,
            "tensors": records, "unmapped_measurements": unmapped}


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
