"""Offline PEFT controls: exact adapter identity and independently merged weights."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import peft
import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file


SOURCE = Path(__file__).resolve().parents[1] / "src" / "model_loader.py"
SPEC = importlib.util.spec_from_file_location("adapter_integrity_loader", SOURCE)
loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader
SPEC.loader.exec_module(loader)


@pytest.fixture(autouse=True)
def offline_loading(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(loader, "get_hf_token", lambda: None)
    monkeypatch.setattr(loader, "hf_repo_has_prefix", lambda *args: False)


def prepare_checkpoint(root, family="llama", *, rslora=False, bias="none"):
    torch.manual_seed(123)
    if family in ("bert", "bert_classifier"):
        config = transformers.BertConfig(
            vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=12, max_position_embeddings=16,
        )
        cls = transformers.BertModel if family == "bert" else transformers.BertForSequenceClassification
        model = cls(config)
        targets = ["query", "value"]
        task = "FEATURE_EXTRACTION" if family == "bert" else "SEQ_CLS"
    elif family.startswith("gpt2"):
        cls = transformers.GPT2LMHeadModel if family == "gpt2" else transformers.GPT2ForTokenClassification
        model = cls(transformers.GPT2Config(
            vocab_size=32, n_embd=8, n_layer=1, n_head=2, n_positions=16,
            n_ctx=16, bos_token_id=1, eos_token_id=2,
        ))
        targets, task = ["c_attn"], "CAUSAL_LM" if family == "gpt2" else "TOKEN_CLS"
    else:
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
        ))
        targets, task = ["q_proj", "v_proj"], "CAUSAL_LM"
    base_path, adapter_path = root / "base", root / "adapter"
    model.save_pretrained(base_path)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    config = peft.LoraConfig(
        task_type=task, target_modules=targets, r=2, lora_alpha=4,
        fan_in_fan_out=family.startswith("gpt2"), use_rslora=rslora, bias=bias,
        modules_to_save=["classifier"] if family.endswith("classifier") else None,
    )
    adapted = peft.get_peft_model(model, config)
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if ".lora_A." in name:
                parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) / 64 + .01)
            elif ".lora_B." in name:
                parameter.copy_(torch.arange(parameter.numel()).reshape(parameter.shape) / 32 - .2)
            elif ".modules_to_save.default." in name or (bias == "all" and name.endswith(".bias")):
                parameter.fill_(.125)
    adapted.peft_config["default"].base_model_name_or_path = str(base_path)
    adapted.save_pretrained(adapter_path, save_embedding_layers=False)
    weights = load_file(str(adapter_path / "adapter_model.safetensors"))
    expected = {name: value.double() for name, value in original.items()}
    # Independent dense formula, not PEFT's get_delta_weight or merge method.
    for key, value in weights.items():
        if key.endswith(".lora_A.weight"):
            stem = key[:-len(".lora_A.weight")]
            delta = weights[stem + ".lora_B.weight"].double() @ value.double()
            if family.startswith("gpt2"):
                delta = delta.T
            name = stem.removeprefix("base_model.model.") + ".weight"
            expected[name] = expected[name] + delta * (4 / (math.sqrt(2) if rslora else 2))
        elif ".lora_" not in key:
            name = key.removeprefix("base_model.model.").replace(".base_layer.", ".")
            expected[name] = value.double()
    return base_path, adapter_path, weights, expected


@pytest.mark.parametrize("family,rslora,bias", [
    ("llama", False, "none"), ("llama", True, "none"),
    ("gpt2", False, "none"), ("gpt2", False, "all"),
    ("gpt2_classifier", False, "none"),
    ("bert", False, "none"), ("bert_classifier", False, "none"),
])
def test_clean_adapter_preserves_identity_and_matches_independent_merge(tmp_path, family, rslora, bias):
    base, adapter, weights, expected = prepare_checkpoint(tmp_path, family, rslora=rslora, bias=bias)
    model, is_adapter = loader.load_model(
        str(adapter), base_model_relation="adapter", source_model=str(base),
        device_map="cpu", torch_dtype="auto", loader_scenario="adapter_requires_base",
    )
    assert is_adapter
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name].double(), expected[name], rtol=1e-6, atol=1e-7)
    if family == "bert":
        assert type(model) is transformers.BertModel
        assert "pooler.dense.weight" in actual
    report = model._model_zoo_loading_info
    assert report["adapter_weights_verified"] and report["safe_merge"]
    details = report["adapter_loading_info"]
    assert details["peft_version"] == peft.__version__
    assert details["adapter_tensor_count"] == len(weights)
    assert len(details["adapter_state_sha256"]) == len(details["adapter_config_sha256"]) == 64
    assert details["checkpoint_dtypes"] == details["loaded_dtypes"] == ["torch.float32"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("defect", ["missing_A", "missing_B", "surplus_lora", "surplus_base", "shape", "nan", "inf", "integer"])
def test_bad_adapter_is_rejected_before_any_checkpoint_tensor_is_applied(tmp_path, monkeypatch, defect):
    base, adapter, weights, _ = prepare_checkpoint(tmp_path)
    a_key = next(key for key in weights if key.endswith(".lora_A.weight"))
    b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
    if defect == "missing_A":
        del weights[a_key]
    elif defect == "missing_B":
        del weights[b_key]
    elif defect == "surplus_lora":
        weights["base_model.model.missing_layer.lora_A.weight"] = torch.ones(2, 8)
    elif defect == "surplus_base":
        weights[a_key.replace(".lora_A.weight", ".base_layer.weight")] = torch.ones(8, 8)
    elif defect == "shape":
        weights[a_key] = weights[a_key][:1]
    elif defect in ("nan", "inf"):
        weights[b_key][0, 0] = float(defect)
    elif defect == "integer":
        weights[a_key] = weights[a_key].to(torch.int32)
    save_file(weights, str(adapter / "adapter_model.safetensors"))

    def must_not_apply(*args, **kwargs):
        pytest.fail("Corrupt adapter reached set_peft_model_state_dict")

    monkeypatch.setattr(peft, "set_peft_model_state_dict", must_not_apply)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    if defect in ("nan", "inf"):
        assert exc.value.reason == "adapter_nonfinite_weights"
    elif defect == "integer":
        assert exc.value.reason == "unsupported_adapter_integrity"
    else:
        assert exc.value.reason == "adapter_checkpoint_mismatch"


def test_finite_inputs_that_overflow_during_merge_are_rejected(tmp_path):
    base, adapter, weights, _ = prepare_checkpoint(tmp_path)
    for value in weights.values():
        value.fill_(1e30)
    save_file(weights, str(adapter / "adapter_model.safetensors"))
    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    assert exc.value.reason == "adapter_merge_nonfinite"


def test_conversion_overflow_is_rejected_before_merge(tmp_path):
    base, adapter, weights, _ = prepare_checkpoint(tmp_path)
    key = next(iter(weights))
    weights[key] = torch.full_like(weights[key], 1e100, dtype=torch.float64)
    save_file(weights, str(adapter / "adapter_model.safetensors"))
    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    assert exc.value.reason == "adapter_nonfinite_weights"


@pytest.mark.parametrize("source,expected_revision", [("unpinned", None), ("pinned", "BASE_PIN"), ("inferred", "BASE_METADATA")])
def test_adapter_and_base_revisions_are_separate(tmp_path, monkeypatch, source, expected_revision):
    base, adapter, _, _ = prepare_checkpoint(tmp_path)
    config = peft.PeftConfig.from_pretrained(adapter)
    config.revision = "BASE_METADATA"
    config.save_pretrained(adapter)
    original = loader._load_checkpoint_checked
    calls = []

    def checked(model_cls, repo, **kwargs):
        calls.append(kwargs.get("revision"))
        return original(model_cls, repo, **kwargs)

    monkeypatch.setattr(loader, "_load_checkpoint_checked", checked)
    source_model = None if source == "inferred" else str(base) + ("@BASE_PIN" if source == "pinned" else "")
    model = loader.load_and_merge_adapter(
        str(adapter), source_model, torch_dtype="auto", revision="ADAPTER_PIN", effective_loader="standard_causal",
    )
    assert calls == [expected_revision]
    report = model._model_zoo_loading_info
    assert report["adapter_requested_revision"] == "ADAPTER_PIN"
    assert report["base_revision_requested"] == (expected_revision or "main")
    assert report["base_revision_loaded"] == (expected_revision or "main")


def test_adapter_task_routing_does_not_use_adapter_revision_for_base_config(tmp_path, monkeypatch):
    base, adapter, _, _ = prepare_checkpoint(tmp_path)
    calls = []

    def resolve(repo, **kwargs):
        calls.append((repo, kwargs.get("revision")))
        return "standard_causal"

    monkeypatch.setattr(loader, "resolve_effective_loader_for_repo", resolve)
    assert loader._resolve_adapter_task_loader(str(adapter), source_model=str(base), revision="ADAPTER_PIN") == "standard_causal"
    assert calls == [(str(base), None)]


def test_conflicting_adapter_base_class_is_an_explicit_failure(tmp_path):
    base, adapter, _, _ = prepare_checkpoint(tmp_path, "bert")
    config = peft.PeftConfig.from_pretrained(adapter)
    config.auto_mapping = {
        "base_model_class": "BertLMHeadModel", "parent_library": "transformers.models.bert.modeling_bert",
    }
    config.save_pretrained(adapter)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    assert exc.value.reason == "adapter_base_architecture_mismatch"


def test_compatible_adapter_base_parent_class_is_allowed(tmp_path):
    base, adapter, _, _ = prepare_checkpoint(tmp_path, "bert")
    config = peft.PeftConfig.from_pretrained(adapter)
    config.auto_mapping = {
        "base_model_class": "BertPreTrainedModel", "parent_library": "transformers.models.bert.modeling_bert",
    }
    config.save_pretrained(adapter)
    model = loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    assert type(model) is transformers.BertModel


@pytest.mark.parametrize("field,value", [
    ("use_dora", True), ("use_qalora", True), ("init_lora_weights", "pissa"),
    ("layer_replication", [(0, 1)]), ("lora_bias", True), ("bias", "lora_only"),
])
def test_unvalidated_adapter_variants_fail_explicitly(tmp_path, field, value):
    base, adapter, _, _ = prepare_checkpoint(tmp_path)
    config = peft.PeftConfig.from_pretrained(adapter)
    setattr(config, field, value)
    config.save_pretrained(adapter)
    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
    assert exc.value.reason == "unsupported_adapter_integrity"


def test_loaded_value_check_detects_silent_payload_changes(tmp_path, monkeypatch):
    base, adapter, _, _ = prepare_checkpoint(tmp_path)
    original = peft.set_peft_model_state_dict

    def change_value(model, state, **kwargs):
        result = original(model, state, **kwargs)
        with torch.no_grad():
            next(parameter for name, parameter in model.named_parameters() if ".lora_A." in name).add_(1)
        return result

    monkeypatch.setattr(peft, "set_peft_model_state_dict", change_value)
    with pytest.raises(loader.LoaderFailure, match="PEFT changed adapter tensor"):
        loader.load_and_merge_adapter(str(adapter), str(base), torch_dtype="auto", effective_loader="standard_causal")
