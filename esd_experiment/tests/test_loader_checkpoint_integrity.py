"""Real, tiny local checkpoints: routing must not invent or discard weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from net_esd.utils import iter_eligible_layers


SOURCE = Path(__file__).resolve().parents[1] / "src" / "model_loader.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("checkpoint_integrity_loader", SOURCE)
loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader
SPEC.loader.exec_module(loader)


@pytest.fixture(autouse=True)
def offline_local_loading(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(loader, "get_hf_token", lambda: None)
    # The fixtures have no adapters. Avoid Hub file-list lookups for local paths.
    monkeypatch.setattr(loader, "hf_repo_has_prefix", lambda *args: False)


def tiny_bert(model_cls=transformers.BertModel):
    config = transformers.BertConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=12, max_position_embeddings=16,
    )
    torch.manual_seed(123)
    return model_cls(config).eval()


def empty_loading_info():
    return {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}


@pytest.mark.parametrize("scenario", ["standard_transformers", "standard_causal"])
@pytest.mark.parametrize("class_name", ["BertModel", "BertForMaskedLM", "BertForSequenceClassification"])
def test_declared_architecture_preserves_all_checkpoint_weights_and_measurements(tmp_path, scenario, class_name):
    expected = tiny_bert(getattr(transformers, class_name))
    expected.save_pretrained(tmp_path)

    actual, is_adapter = loader.load_model(
        str(tmp_path), loader_scenario=scenario, device_map="cpu", torch_dtype="auto",
    )

    assert not is_adapter
    assert type(actual) is type(expected)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, weight in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], weight), name
    expected_layers = {name: weight for name, weight, _ in iter_eligible_layers(expected)}
    actual_layers = {name: weight for name, weight, _ in iter_eligible_layers(actual)}
    assert actual_layers.keys() == expected_layers.keys()
    for name in expected_layers:
        assert torch.equal(actual_layers[name], expected_layers[name]), name
    if class_name == "BertForMaskedLM":
        assert actual.get_input_embeddings().weight is actual.get_output_embeddings().weight
    report = actual._model_zoo_loading_info
    assert report["validated"]
    assert report["model_class"] == report["loader_class"] == class_name
    assert report["loading_info"] == empty_loading_info()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_real_incomplete_or_surplus_checkpoint_is_rejected(tmp_path, defect):
    expected = tiny_bert()
    state = expected.state_dict()
    if defect == "missing":
        offending_key = "encoder.layer.0.attention.self.query.weight"
        state.pop(offending_key)
    else:
        offending_key = "unused_projection.weight"
        state[offending_key] = torch.eye(8)
    expected.save_pretrained(tmp_path, state_dict=state)

    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_model(str(tmp_path), loader_scenario="standard_causal", torch_dtype="auto")

    assert exc.value.reason == "checkpoint_weight_mismatch"
    assert offending_key in str(exc.value)


def test_missing_architecture_cannot_silently_create_an_lm_head(tmp_path):
    expected = tiny_bert()
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    with pytest.raises(loader.LoaderFailure, match="cls.predictions") as exc:
        loader.load_model(str(tmp_path), loader_scenario="standard_causal", torch_dtype="auto")

    assert exc.value.reason == "checkpoint_weight_mismatch"


@pytest.mark.parametrize("field,value", [
    ("missing_keys", ["projection.weight"]),
    ("unexpected_keys", ["classifier.weight"]),
    ("unexpected_keys", ["classifier.bias"]),
    ("mismatched_keys", [("projection.weight", (8, 8), (4, 8))]),
    ("error_msgs", ["checkpoint read failed"]),
    ("conversion_errors", {"projection.weight": "checkpoint conversion failed"}),
])
def test_loading_report_failures_are_not_success(monkeypatch, field, value):
    report = empty_loading_info()
    report[field] = value
    model = SimpleNamespace(config=SimpleNamespace(model_type="bert"))
    monkeypatch.setattr(loader, "hf_from_pretrained", lambda *args, **kwargs: (model, report))

    with pytest.raises(loader.LoaderFailure) as exc:
        loader._load_checkpoint_checked(transformers.BertModel, "unused-local-checkpoint")

    assert exc.value.reason == "checkpoint_weight_mismatch"


@pytest.mark.parametrize("returned", [None, SimpleNamespace(), (SimpleNamespace(), {})])
def test_absent_loading_report_is_not_success(monkeypatch, returned):
    monkeypatch.setattr(loader, "hf_from_pretrained", lambda *args, **kwargs: returned)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader._load_checkpoint_checked(transformers.BertModel, "unused-local-checkpoint")
    assert exc.value.reason == "checkpoint_loading_info_missing"


def test_known_historical_gpt2_buffers_are_recorded_and_json_safe(monkeypatch):
    report = empty_loading_info()
    report["unexpected_keys"] = {
        "transformer.h.0.attn.masked_bias", "transformer.h.1.attn.masked_bias",
    }
    model = SimpleNamespace(config=SimpleNamespace(model_type="gpt2"))
    monkeypatch.setattr(loader, "hf_from_pretrained", lambda *args, **kwargs: (model, report))

    assert loader._load_checkpoint_checked(transformers.GPT2LMHeadModel, "unused") is model
    stored = model._model_zoo_loading_info
    assert stored["allowed_unexpected_keys"] == sorted(report["unexpected_keys"])
    assert stored["loading_info"]["unexpected_keys"] == sorted(report["unexpected_keys"])
    json.dumps(stored, allow_nan=False)


@pytest.mark.parametrize("model_type,key", [
    ("bert", "transformer.h.0.attn.masked_bias"),
    ("gpt2", "transformer.h.0.attn.bias"),
    ("gpt2", "transformer.h.0.attn.masked_bias.weight"),
    ("gpt2", "classifier.masked_bias"),
])
def test_unexpected_key_exception_is_narrow(monkeypatch, model_type, key):
    report = empty_loading_info()
    report["unexpected_keys"] = [key]
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    monkeypatch.setattr(loader, "hf_from_pretrained", lambda *args, **kwargs: (model, report))
    with pytest.raises(loader.LoaderFailure):
        loader._load_checkpoint_checked(transformers.GPT2LMHeadModel, "unused")


def test_architecture_lookup_does_not_execute_non_model_callables(monkeypatch):
    config = SimpleNamespace(architectures=["set_seed", "__import__('os')"])
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    assert loader._declared_checkpoint_model_cls("unused") is None


def test_multiple_supported_architectures_fail_without_guessing(monkeypatch):
    config = transformers.BertConfig(architectures=["BertModel", "BertForMaskedLM"])
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader._declared_checkpoint_model_cls("unused")
    assert exc.value.reason == "ambiguous_checkpoint_architecture"


def test_unknown_architecture_keeps_existing_route_but_not_an_unchecked_load(monkeypatch):
    config = transformers.BertConfig(architectures=["UnknownCustomBertModel"])
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    assert loader._declared_checkpoint_model_cls("unused") is None
    report = empty_loading_info()
    report["missing_keys"] = ["custom_head.weight"]
    model = SimpleNamespace(config=config)
    monkeypatch.setattr(loader, "hf_from_pretrained", lambda *args, **kwargs: (model, report))
    with pytest.raises(loader.LoaderFailure, match="custom_head.weight"):
        loader.load_model("unused", loader_scenario="standard_causal")


def test_incompatible_declared_architecture_is_rejected(monkeypatch):
    config = transformers.BertConfig(architectures=["GPT2LMHeadModel"])
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *args, **kwargs: config)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader._declared_checkpoint_model_cls("unused")
    assert exc.value.reason == "checkpoint_architecture_mismatch"


def test_generic_hf_wrapper_still_supports_requested_loading_info(monkeypatch):
    expected = (SimpleNamespace(), empty_loading_info())

    class FakeModel:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            assert kwargs["output_loading_info"]
            return expected

    assert loader.hf_from_pretrained(FakeModel, "unused", output_loading_info=True) is expected
