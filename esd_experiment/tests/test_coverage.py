"""Offline CPU controls for module identity, declared layouts, and coverage."""

import math
import queue
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import net_esd
from net_esd.utils import iter_eligible_layers


class CustomLinear(nn.Linear):
    pass


class UnknownWeight(nn.Module):
    def __init__(self, shape=(4, 4)):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))


def describe(model, **kwargs):
    coverage = []
    layers = list(iter_eligible_layers(model, coverage=coverage, **kwargs))
    return layers, {record["module_name"]: record for record in coverage}


def test_supported_classes_include_subclasses_embeddings_and_declared_hf_conv1d():
    hf_conv_class = type("Conv1D", (UnknownWeight,), {"__module__": "transformers.pytorch_utils"})
    model = nn.ModuleDict({
        "linear": nn.Linear(4, 4),
        "custom": CustomLinear(4, 4),
        "embedding": nn.Embedding(40, 4),
        "hf_projection": hf_conv_class(),
        "conv1d": nn.Conv1d(4, 4, 3),
        "conv2d": nn.Conv2d(4, 4, 3),
        "conv3d": nn.Conv3d(4, 4, 2),
    })

    layers, coverage = describe(model)

    assert [name for name, _, _ in layers] == list(model.keys())
    assert len(coverage) == 7  # No weightless ModuleDict record.
    assert coverage["hf_projection"]["weight_layout"] == "matrix"
    assert coverage["conv1d"]["weight_layout"] == "conv1d_slices"
    assert coverage["conv2d"]["weight_layout"] == "conv2d_slices"
    assert coverage["conv3d"]["weight_layout"] == "conv3d_slices"
    assert all(record["weight_attribute"] == "weight" for record in coverage.values())
    assert all(record["status"] == "eligible" for record in coverage.values())


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 3), (4, 4, 3, 3)])
@pytest.mark.parametrize("filter_type", [True, False])
def test_unknown_layouts_require_explicit_dense_matrix_opt_in(shape, filter_type):
    layers, coverage = describe(UnknownWeight(shape), filter_type=filter_type)

    if not filter_type and len(shape) == 2:
        assert len(layers) == 1
        assert coverage[""]["status"] == "eligible"
    else:
        assert layers == []
        assert coverage[""]["reason"] == "unsupported_module_type"


def test_linear_aspect_ratio_filter_is_reported_and_can_be_disabled():
    model = nn.Linear(4, 32)
    layers, coverage = describe(model)

    assert layers == []
    assert coverage[""]["reason"] == "linear_aspect_ratio_at_least_8"
    assert len(describe(model, filter_type=False)[0]) == 1


def test_known_class_with_unsupported_shape_is_skipped_before_numerics():
    layer = nn.Linear(4, 4)
    layer.weight = nn.Parameter(torch.ones(4, 4, 3))

    layers, coverage = describe(layer)

    assert layers == []
    assert coverage[""]["reason"] == "unsupported_weight_shape"
    assert coverage[""]["weight_shape"] == [4, 4, 3]


def test_skip_reasons_cover_non_measurement_and_unmaterialized_weights():
    integer = UnknownWeight()
    integer.weight = nn.Parameter(torch.ones(4, 4, dtype=torch.int8), requires_grad=False)
    quantized = nn.Module()
    quantized.register_buffer("weight", torch.quantize_per_tensor(torch.ones(4, 4), 0.1, 0, torch.qint8))
    model = nn.ModuleDict({
        "integer": integer,
        "quantized": quantized,
        "lazy": nn.LazyLinear(4),
        "meta": nn.Linear(4, 4, device="meta"),
        "empty": nn.Embedding(0, 4),
        "norm": nn.LayerNorm(4),
        "relu": nn.ReLU(),
    })

    layers, coverage = describe(model)

    assert layers == []
    assert {name: record["reason"] for name, record in coverage.items()} == {
        "integer": "non_floating_weight",
        "quantized": "unsupported_weight_representation",
        "lazy": "uninitialized_weight",
        "meta": "meta_weight",
        "empty": "empty_weight",
        "norm": "weight_has_fewer_than_two_dimensions",
    }


def test_packed_parameter_subclasses_are_not_interpreted_as_dense_weights():
    class PackedParameter(nn.Parameter):
        pass

    layer = nn.Linear(4, 4)
    layer.weight = PackedParameter(torch.ones(4, 4))

    layers, coverage = describe(layer)

    assert layers == []
    assert coverage[""]["reason"] == "unsupported_weight_representation"


def test_callable_quantized_weight_is_reported_without_guessing_unpacking():
    class PackedModule(nn.Module):
        def weight(self):
            raise AssertionError("Weight accessor should not be called")

    layers, coverage = describe(PackedModule())

    assert layers == []
    assert coverage[""]["reason"] == "weight_is_not_a_tensor"


@pytest.mark.parametrize("attribute", ["qweight", "weight_packed", "packed_weight"])
def test_known_packed_attributes_are_visible_without_unpacking(attribute):
    module = nn.Module()
    module.register_buffer(attribute, torch.ones(4, 4, dtype=torch.int32))

    layers, coverage = describe(module, filter_type=False)

    assert layers == []
    assert coverage[""]["status"] == "skipped"
    assert coverage[""]["reason"] == "unsupported_weight_attribute"
    assert coverage[""]["weight_attribute"] == attribute
    assert coverage[""]["weight_shape"] == [4, 4]
    assert coverage[""]["weight_dtype"] == "int32"
    assert coverage[""]["measurement_names"] == []


def test_custom_kernel_parameters_are_reported_but_arbitrary_buffers_are_not():
    custom = nn.Module()
    custom.kernel = nn.Parameter(torch.ones(4, 4))
    custom.other_kernel = nn.Parameter(torch.ones(4, 4, 3))
    custom.offset = nn.Parameter(torch.ones(4))
    buffer_only = nn.Module()
    buffer_only.register_buffer("running_matrix", torch.ones(4, 4))
    model = nn.ModuleDict({"custom": custom, "buffer_only": buffer_only})

    layers, coverage = describe(model)

    assert layers == []
    assert list(coverage) == ["custom"]  # Neither containers nor arbitrary buffers.
    assert coverage["custom"]["weight_attribute"] == "kernel"
    assert coverage["custom"]["unsupported_weight_attributes"] == ["kernel", "other_kernel"]
    assert coverage["custom"]["reason"] == "unsupported_weight_attribute"


@pytest.mark.parametrize("kdim,attributes", [
    (None, ["in_proj_weight"]),
    (4, ["q_proj_weight", "k_proj_weight", "v_proj_weight"]),
])
def test_multihead_attention_projection_missingness_is_explicit(kdim, attributes):
    module = nn.MultiheadAttention(8, 2, kdim=kdim)

    layers, coverage = describe(module)

    assert [name for name, _, _ in layers] == ["out_proj"]
    assert list(coverage) == ["", "out_proj"]
    assert coverage[""]["reason"] == "unsupported_weight_attribute"
    assert coverage[""]["unsupported_weight_attributes"] == attributes
    assert coverage["out_proj"]["weight_attribute"] == "weight"
    assert coverage["out_proj"]["status"] == "eligible"


def test_standard_weight_selection_is_unchanged_when_packed_metadata_exists():
    module = nn.Linear(4, 4)
    module.register_buffer("qweight", torch.ones(4, 4, dtype=torch.int8))

    layers, coverage = describe(module)

    assert len(layers) == 1
    assert coverage[""]["weight_attribute"] == "weight"
    assert coverage[""]["status"] == "eligible"
    assert "unsupported_weight_attributes" not in coverage[""]


def test_legacy_qkv_slices_keep_canonical_module_identity():
    model = nn.ModuleDict({"attn": nn.Linear(4, 12, bias=False)})
    coverage = []

    results = net_esd.net_esd_estimator(model, parallel=False, coverage=coverage)

    assert results["longname"] == ["attn_q", "attn_k", "attn_v"]
    assert results["module_name"] == ["attn"] * 3
    assert results["slice"] == ["q", "k", "v"]
    assert coverage[0]["measurement_names"] == results["longname"]
    assert coverage[0]["measurement_slices"] == results["slice"]
    assert coverage[0]["status"] == "analyzed"
    assert len({len(column) for column in results.values()}) == 1


def test_qkv_split_requires_exact_three_to_one_shape_and_not_an_embedding():
    model = nn.ModuleDict({
        "attn_uneven": nn.Linear(7, 2),
        "attention_embedding": nn.Embedding(12, 4),
    })

    layers, coverage = describe(model)

    assert [name for name, _, _ in layers] == ["attn_uneven", "attention_embedding"]
    assert all(record["measurement_slices"] == [""] for record in coverage.values())


def test_emitted_name_collision_is_rejected_before_computing(monkeypatch):
    model = nn.ModuleDict({"attn": nn.Linear(4, 12), "attn_q": nn.Linear(4, 4)})
    monkeypatch.setattr(net_esd, "compute_esd_for_weight", lambda *args: pytest.fail("Must reject names before computation"))

    with pytest.raises(ValueError, match="Duplicate ESD measurement name: 'attn_q'"):
        net_esd.net_esd_estimator(model, parallel=False)


def test_encoder_decoder_and_nontransformer_module_names_remain_distinct():
    def stack():
        return nn.ModuleDict({"layers": nn.ModuleList([
            nn.ModuleDict({"self_attn": nn.ModuleDict({"q_proj": nn.Linear(4, 4)})})
        ])})

    model = nn.ModuleDict({
        "encoder": stack(), "decoder": stack(),
        "features": nn.Sequential(nn.Conv1d(4, 4, 3)),
        "classifier": nn.Linear(4, 4),
    })
    coverage = []

    result = net_esd.net_esd_estimator(model, parallel=False, coverage=coverage)

    assert result["module_name"] == [
        "encoder.layers.0.self_attn.q_proj",
        "decoder.layers.0.self_attn.q_proj",
        "features.0", "classifier",
    ]
    assert result["longname"] == result["module_name"]
    assert result["slice"] == [""] * 4
    assert [record["status"] for record in coverage] == ["analyzed"] * 4


@pytest.mark.parametrize("conv", [nn.Conv1d(2, 3, 2), nn.Conv2d(2, 3, 2), nn.Conv3d(2, 3, 2)])
def test_standard_convolutions_complete_with_declared_kernel_slice_spectra(conv):
    result = net_esd.net_esd_estimator(conv, parallel=False, compute_dtype="float64")
    spatial_count = math.prod(conv.weight.shape[2:])

    assert result["raw_num_evals"] == [2 * spatial_count]
    assert result["compute_dtype"] == ["float64"]
    assert result["source_dtype"] == ["float32"]
    assert result["module_name"] == [""]


def test_filtered_and_missing_fits_are_analyzed_not_silently_skipped():
    model = nn.ModuleDict({"tiny": nn.Linear(4, 4, bias=False), "constant": nn.Linear(4, 4, bias=False)})
    with torch.no_grad():
        model["tiny"].weight.copy_(torch.eye(4) * 1e-4)
        model["constant"].weight.copy_(torch.eye(4))
    coverage = []

    result = net_esd.net_esd_estimator(model, parallel=False, filter_zeros=True, coverage=coverage)

    assert result["longname"] == ["tiny", "constant"]
    assert result["num_evals"] == [0, 4]
    assert all(math.isnan(alpha) for alpha in result["alpha"])
    assert all(record["status"] == "analyzed" for record in coverage)
    for record, fit_status in zip(coverage, result["fit_status"]):
        assert record["fit_statuses"] == {record["module_name"]: fit_status}
        assert fit_status not in ("fitted", "unknown")


def test_empty_spectrum_return_is_reported_as_skipped(monkeypatch):
    monkeypatch.setattr(net_esd, "compute_esd_for_weight", lambda *args: None)
    coverage = []

    result = net_esd.net_esd_estimator(nn.Linear(4, 4), parallel=False, coverage=coverage)

    assert all(values == [] for values in result.values())
    assert coverage[0]["status"] == "skipped"
    assert coverage[0]["reason"] == "no_spectrum"


def test_partial_slice_coverage_is_explicit(monkeypatch):
    def compute(name, *args):
        return None if name.endswith("_v") else {"longname": name, "fit_status": "fitted"}

    monkeypatch.setattr(net_esd, "compute_esd_for_weight", compute)
    coverage = []

    result = net_esd.net_esd_estimator(nn.ModuleDict({"attn": nn.Linear(4, 12)}), parallel=False, coverage=coverage)

    assert result["slice"] == ["q", "k"]
    assert coverage[0]["status"] == "partially_analyzed"
    assert coverage[0]["reason"] == "no_spectrum_for_some_slices"


def test_thread_backend_preserves_order_identity_and_precision(monkeypatch):
    calls = []

    def compute(name, *args):
        calls.append((name, args[-1]))
        return {"longname": name, "fit_status": "fitted"}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(net_esd, "compute_esd_for_weight", compute)
    model = nn.ModuleDict({"small": nn.Linear(2, 2), "large": nn.Linear(4, 4)})
    coverage = []

    result = net_esd.net_esd_estimator(model, backend="thread", compute_dtype="float64", coverage=coverage)

    assert result["module_name"] == ["small", "large"]
    assert sorted(calls) == [("large", "float64"), ("small", "float64")]
    assert all(record["status"] == "analyzed" for record in coverage)


@pytest.mark.parametrize("parallel", [False, True])
def test_numerical_failures_are_not_converted_to_missing_measurements(monkeypatch, parallel):
    def compute(*args):
        raise RuntimeError("synthetic numerical failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(net_esd, "compute_esd_for_weight", compute)

    with pytest.raises(RuntimeError, match="synthetic numerical failure"):
        net_esd.net_esd_estimator(nn.Linear(4, 4), parallel=parallel)


class FakeProcessContext:
    """Exercise process scheduling/transport without CUDA or subprocesses."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.queues = []
        self.process_args = []
        self.transported = []

    def Queue(self, maxsize=0):
        pending = queue.Queue()
        if not self.queues:
            original_put = pending.put

            def put(task, *args, **kwargs):
                original_put(task, *args, **kwargs)
                if task is not None:
                    index, name, weight, params = task
                    self.transported.append((name, weight.dtype.name, params))
                    error = "synthetic process failure" if self.fail else None
                    self.queues[1].put((index, {"longname": name, "fit_status": "fitted"}, error))

            pending.put = put
        pending.close = lambda: None
        pending.cancel_join_thread = lambda: None
        self.queues.append(pending)
        return pending

    def Process(self, *, target, args):
        self.process_args.append(args)
        return SimpleNamespace(
            start=lambda: None, join=lambda timeout=None: None,
            terminate=lambda: None, is_alive=lambda: False, exitcode=0,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_process_backend_precision_identity_and_error_transport(monkeypatch, fail):
    context = FakeProcessContext(fail=fail)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(net_esd, "get_context", lambda method: context)
    model = nn.ModuleDict({"projection": nn.Linear(4, 4, dtype=torch.bfloat16)})

    if fail:
        with pytest.raises(RuntimeError, match="synthetic process failure"):
            net_esd.net_esd_estimator(model, backend="process", compute_dtype="float64")
    else:
        result = net_esd.net_esd_estimator(model, backend="process", compute_dtype="float64")
        assert result["source_dtype"] == ["bfloat16"]
        assert result["module_name"] == ["projection"]
    assert context.process_args[0][-1] == "float64"
    assert context.transported == [("projection", "float64", 20)]


@pytest.mark.parametrize("phase", ["submission", "collection"])
def test_crashed_process_cannot_block_queue_submission_or_result_collection(monkeypatch, phase):
    class DeadContext(FakeProcessContext):
        def Queue(self, maxsize=0):
            pending = super().Queue(maxsize)

            def full(*args, **kwargs):
                raise queue.Full

            def empty(*args, **kwargs):
                raise queue.Empty

            if len(self.queues) == 1:
                pending.put = full if phase == "submission" else lambda *args, **kwargs: None
            else:
                pending.get = empty
            return pending

    context = DeadContext()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(net_esd, "get_context", lambda method: context)

    with pytest.raises(RuntimeError, match="An ESD subprocess exited"):
        net_esd.net_esd_estimator(nn.Linear(4, 4), backend="process")
