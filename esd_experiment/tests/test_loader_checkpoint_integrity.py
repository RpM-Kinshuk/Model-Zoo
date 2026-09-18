"""Real, tiny local checkpoints: routing must not invent or discard weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers

from net_esd import net_esd_estimator
from net_esd.utils import iter_eligible_layers, weight_usage_report


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


def test_dense_fallback_measures_a_loaded_bert_head_without_changing_checkpoint(tmp_path):
    config = tiny_bert().config
    config.num_labels = 64  # classifier is 64 x 8: excluded by the legacy filter.
    expected = transformers.BertForSequenceClassification(config).eval()
    expected.save_pretrained(tmp_path)
    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert actual._model_zoo_loading_info["validated"]
    assert "classifier" not in {name for name, _, _ in iter_eligible_layers(actual)}
    coverage = []

    metrics = net_esd_estimator(actual, filter_type=False, parallel=False,
                               fix_fingers="xmin_mid", coverage=coverage)

    index = metrics["longname"].index("classifier")
    reference = torch.linalg.svdvals(expected.classifier.weight.detach().double()).square().sort().values
    torch.testing.assert_close(torch.as_tensor(metrics["eigs"][index]).double(), reference, rtol=5e-5, atol=1e-7)
    usage = weight_usage_report(actual, coverage, metrics["longname"])
    head = next(record for record in usage["tensors"] if record["name"] == "classifier.weight")
    assert head["status"] == "measured" and head["measurement_names"] == ["classifier"]
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name


ENCODER_FAMILIES = ["roberta", "distilbert", "albert", "deberta", "deberta-v2"]


def tiny_encoder(family, auto_cls=transformers.AutoModelForMaskedLM):
    if family == "distilbert":
        dimensions = dict(dim=8, hidden_dim=12, n_layers=2, n_heads=2)
    else:
        dimensions = dict(hidden_size=8, intermediate_size=12, num_hidden_layers=2, num_attention_heads=2)
    if family == "albert":
        dimensions.update(embedding_size=4, num_hidden_groups=1)
    if family.startswith("deberta"):
        dimensions.update(relative_attention=True, pos_att_type=["p2c", "c2p"])
    config = transformers.AutoConfig.for_model(family, vocab_size=32, max_position_embeddings=16, **dimensions)
    torch.manual_seed(123)
    return auto_cls.from_config(config).eval()


@pytest.mark.parametrize("family", ENCODER_FAMILIES)
@pytest.mark.parametrize("head", ["", "ForMaskedLM", "ForSequenceClassification", "ForTokenClassification", "ForQuestionAnswering"])
@pytest.mark.parametrize("declared", [True, False])
def test_encoder_families_preserve_weights_and_layer_coverage(tmp_path, family, head, declared):
    expected = tiny_encoder(family, getattr(transformers, "AutoModel" + head))
    expected.save_pretrained(tmp_path)
    if not declared:
        expected.config.architectures = None
        expected.config.save_pretrained(tmp_path)

    actual, is_adapter = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert not is_adapter
    assert type(actual) is type(expected)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    expected_layers = {name: weight for name, weight, _ in iter_eligible_layers(expected)}
    actual_layers = {name: weight for name, weight, _ in iter_eligible_layers(actual)}
    assert actual_layers.keys() == expected_layers.keys()
    for name, weight in expected_layers.items():
        assert torch.equal(actual_layers[name], weight), name
    selection = actual._model_zoo_loading_info["architecture_selection"]
    assert selection["method"] == ("declared_architecture" if declared else "checkpoint_shapes")
    assert selection["model_class"] == type(expected).__name__


@pytest.mark.parametrize("family", ENCODER_FAMILIES)
def test_missing_metadata_sharded_encoders_preserve_weights(tmp_path, family):
    expected = tiny_encoder(family)
    expected.save_pretrained(tmp_path, max_shard_size="2KB")
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert type(actual) is type(expected)
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert len(actual._model_zoo_loading_info["architecture_selection"]["files"]) > 1


@pytest.mark.parametrize("family", ENCODER_FAMILIES)
@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_incomplete_encoder_checkpoints_do_not_match_another_head(tmp_path, family, defect):
    expected = tiny_encoder(family)
    state = expected.state_dict()
    key = next(name for name in state if "attention" in name and state[name].ndim == 2)
    if defect == "missing":
        del state[key]
    elif defect == "unexpected":
        state["unknown.weight"] = torch.ones(4, 4)
    else:
        state[key] = torch.ones(3, 3)
    expected.save_pretrained(tmp_path, state_dict=state)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert exc.value.reason == "checkpoint_architecture_unresolved"


def test_roberta_decoder_is_not_misidentified_as_pretraining_masked_lm(tmp_path):
    config = tiny_encoder("roberta").config
    config.is_decoder = True
    expected = transformers.RobertaForCausalLM(config)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert type(actual) is transformers.RobertaForCausalLM
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name


@pytest.mark.parametrize("family", ["roberta", "distilbert", "albert", "deberta-v2"])
def test_encoder_multiple_choice_heads_without_metadata(tmp_path, family):
    expected = tiny_encoder(family, transformers.AutoModelForMultipleChoice)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert type(actual) is type(expected)


def test_albert_pretraining_keeps_shared_layers_and_both_heads(tmp_path):
    expected = tiny_encoder("albert", transformers.AutoModelForPreTraining)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert type(actual) is transformers.AlbertForPreTraining
    assert actual.config.num_hidden_layers == 2
    assert len(actual.albert.encoder.albert_layer_groups) == 1
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name


@pytest.mark.parametrize("family", ["distilbert", "albert", "deberta", "deberta-v2"])
def test_encoder_only_family_is_not_inferred_as_a_decoder(family):
    expected = tiny_encoder(family)
    expected.config.is_decoder = True
    shapes = {name: tuple(tensor.shape) for name, tensor in expected.state_dict().items()}

    assert loader._matching_encoder_classes(expected.config, shapes) == []


@pytest.mark.parametrize("family", ["distilbert", "albert", "deberta-v2"])
def test_one_output_encoder_heads_require_metadata_when_ambiguous(tmp_path, family):
    config = tiny_encoder(family).config
    config.num_labels = 1
    expected = transformers.AutoModelForSequenceClassification.from_config(config)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert exc.value.reason == "checkpoint_layout_ambiguous"


@pytest.mark.parametrize("family", ["bert"] + ENCODER_FAMILIES)
@pytest.mark.parametrize("head", ["ForSequenceClassification", "ForTokenClassification"])
def test_multiclass_encoder_heads_are_not_blocked_by_qa_constraints(tmp_path, family, head):
    config = tiny_encoder(family).config
    config.num_labels = 3
    expected = getattr(transformers, "AutoModel" + head).from_config(config)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert type(actual) is type(expected)
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name


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


@pytest.mark.parametrize("class_name", [
    "BertModel", "BertForPreTraining", "BertForMaskedLM", "BertForSequenceClassification",
    "BertForTokenClassification", "BertForQuestionAnswering", "BertForMultipleChoice",
    "BertForNextSentencePrediction",
])
@pytest.mark.parametrize("sharded", [False, True])
def test_missing_bert_architecture_is_selected_without_changing_weights(tmp_path, class_name, sharded):
    expected = tiny_bert(getattr(transformers, class_name))
    expected.save_pretrained(tmp_path, max_shard_size="2KB" if sharded else "5GB")
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, is_adapter = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    assert not is_adapter
    assert type(actual) is type(expected)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, weight in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], weight), name
    selection = actual._model_zoo_loading_info["architecture_selection"]
    assert selection["method"] == "checkpoint_shapes"
    assert selection["model_class"] == class_name
    assert (len(selection["files"]) > 1) == sharded
    assert selection["dtypes"] == ["F32"]


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape"])
def test_missing_metadata_does_not_hide_incomplete_bert_checkpoints(tmp_path, defect):
    expected = tiny_bert(transformers.BertForPreTraining)
    state = expected.state_dict()
    key = "bert.encoder.layer.0.attention.self.query.weight"
    if defect == "missing":
        del state[key]
    elif defect == "unexpected":
        state["unknown.weight"] = torch.ones(4, 4)
    else:
        state[key] = torch.ones(3, 3)
    expected.save_pretrained(tmp_path, state_dict=state)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert exc.value.reason == "checkpoint_architecture_unresolved"


def test_identical_classification_and_multiple_choice_layouts_are_ambiguous(tmp_path):
    config = tiny_bert().config
    config.num_labels = 1
    expected = transformers.BertForSequenceClassification(config)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    with pytest.raises(loader.LoaderFailure, match="BertForMultipleChoice") as exc:
        loader.load_model(str(tmp_path), loader_scenario="standard_causal", torch_dtype="auto")

    assert exc.value.reason == "checkpoint_layout_ambiguous"


def test_declared_wrong_bert_architecture_is_not_silently_overridden(tmp_path, monkeypatch):
    expected = tiny_bert(transformers.BertForPreTraining)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = ["BertForMaskedLM"]
    expected.config.save_pretrained(tmp_path)
    monkeypatch.setattr(loader, "_checkpoint_tensor_shapes", lambda *args: pytest.fail("Must honor declaration"))

    with pytest.raises(loader.LoaderFailure) as exc:
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert exc.value.reason == "checkpoint_weight_mismatch"


@pytest.mark.parametrize("declared", [False, True])
def test_selected_architecture_load_error_does_not_try_another_class(tmp_path, monkeypatch, declared):
    expected = tiny_bert(transformers.BertForPreTraining)
    expected.save_pretrained(tmp_path)
    if not declared:
        expected.config.architectures = None
        expected.config.save_pretrained(tmp_path)
    calls = []

    def fail_load(model_cls, repo_id, **kwargs):
        calls.append(model_cls)
        if not declared:
            assert kwargs["use_safetensors"] is True
        raise RuntimeError("synthetic load error")

    monkeypatch.setattr(loader, "hf_from_pretrained", fail_load)
    monkeypatch.setattr(loader, "_fallback_loader_from_error", lambda *args: pytest.fail("Must not change class"))
    with pytest.raises(RuntimeError, match="synthetic load error"):
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert calls == [transformers.BertForPreTraining]


def test_header_inspection_passes_the_pin_to_every_shard(tmp_path, monkeypatch):
    import transformers.utils.hub as hub
    tiny_bert().save_pretrained(tmp_path, max_shard_size="2KB")
    calls = []

    def resolve(repo_id, filename, **kwargs):
        assert repo_id == "org/pinned-bert"
        assert kwargs["revision"] == "a" * 40
        calls.append(filename)
        path = tmp_path / filename
        return str(path) if path.is_file() else None

    monkeypatch.setattr(hub, "cached_file", resolve)
    shapes, inspection = loader._checkpoint_tensor_shapes("org/pinned-bert", "a" * 40)
    assert len(shapes) == len(tiny_bert().state_dict())
    assert set(calls[2:]) == set(inspection["files"])
    assert calls[:2] == ["model.safetensors", "model.safetensors.index.json"]


def test_header_inspection_rejects_unpinned_remote_inputs(monkeypatch):
    import transformers.utils.hub as hub
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: pytest.fail("Must not fetch moving revision"))
    with pytest.raises(loader.LoaderFailure, match="pinned revision"):
        loader._checkpoint_tensor_shapes("org/bert", "main")


def test_bert_inspection_uses_meta_not_trial_weight_loads(monkeypatch):
    expected = tiny_bert(transformers.BertForPreTraining)
    shapes = {name: tuple(tensor.shape) for name, tensor in expected.state_dict().items()}
    seen = []
    original = transformers.PreTrainedModel.state_dict

    def inspect_state(model, *args, **kwargs):
        state = original(model, *args, **kwargs)
        assert all(tensor.device.type == "meta" for tensor in state.values())
        seen.append(type(model).__name__)
        return state

    monkeypatch.setattr(transformers.PreTrainedModel, "state_dict", inspect_state)
    assert loader._matching_encoder_classes(expected.config, shapes) == [transformers.BertForPreTraining]
    assert len(set(seen)) == 8


def test_decoder_config_distinguishes_bert_causal_and_masked_lm(tmp_path):
    config = tiny_bert().config
    config.is_decoder = True
    expected = transformers.BertLMHeadModel(config)
    expected.save_pretrained(tmp_path)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert type(actual) is transformers.BertLMHeadModel
    assert actual.config.is_decoder


def test_distinct_serialized_embedding_weights_are_not_silently_overwritten(tmp_path):
    from safetensors.torch import save_file
    expected = tiny_bert(transformers.BertForPreTraining)
    state = {name: tensor.clone() for name, tensor in expected.state_dict().items()}
    state["cls.predictions.decoder.weight"].add_(1)
    save_file(state, tmp_path / "model.safetensors", metadata={"format": "pt"})
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)

    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")

    # Transformers can preserve both distinct copies instead of tying them.
    # The important property is that no checkpoint values are overwritten.
    for name, tensor in state.items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert not actual._model_zoo_loading_info["input_output_embeddings_tied"]


def test_missing_metadata_does_not_enable_unrestricted_pickle_loading(tmp_path, monkeypatch):
    tiny_bert().config.save_pretrained(tmp_path)
    torch.save(torch.nn.Linear(2, 2), tmp_path / "pytorch_model.bin")  # A module is not a plain state dict.
    original = torch.load
    calls = []

    def restricted_load(*args, **kwargs):
        calls.append(kwargs["weights_only"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", restricted_load)
    with pytest.raises(loader.LoaderFailure, match="No unrestricted loading") as exc:
        loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert exc.value.reason == "checkpoint_inspection_failed"
    assert calls == [True]


@pytest.mark.parametrize("family", ["bert", "deberta-v2"])
@pytest.mark.parametrize("file_format", ["zip", "legacy", "sharded"])
def test_legacy_encoder_inspection_and_loading_preserve_weights(tmp_path, monkeypatch, family, file_format):
    expected = tiny_bert(transformers.BertForPreTraining) if family == "bert" else tiny_encoder(family)
    expected.config.architectures = None
    expected.config.save_pretrained(tmp_path)
    state = expected.state_dict()
    if file_format == "sharded":
        groups = [dict(list(state.items())[::2]), dict(list(state.items())[1::2])]
        weight_map = {}
        for number, group in enumerate(groups):
            filename = f"pytorch_model-{number}.bin"
            torch.save(group, tmp_path / filename)
            weight_map.update({name: filename for name in group})
        (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    else:
        torch.save(state, tmp_path / "pytorch_model.bin", _use_new_zipfile_serialization=file_format == "zip")

    original = torch.load
    inspections = []

    def checked_load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        result = original(*args, **kwargs)
        if torch._guards.active_fake_mode() is not None:
            assert all(tensor.untyped_storage().device.type == "meta" for tensor in result.values())
            inspections.append(args[0])
        return result

    monkeypatch.setattr(torch, "load", checked_load)
    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert inspections
    assert type(actual) is type(expected)
    for name, tensor in state.items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert actual._model_zoo_loading_info["architecture_selection"]["format"] == "pytorch"
    with pytest.raises(loader.LoaderFailure, match="requires safetensors"):
        loader.checkpoint_tensor_index(str(tmp_path), None)  # Callers must explicitly permit .bin inspection.


@pytest.mark.parametrize("defect", [None, "missing_dimensions", "unknown_weight", "position_ids"])
def test_missing_bert_model_type_requires_complete_config_and_weight_evidence(tmp_path, defect):
    expected = tiny_bert(transformers.BertForPreTraining)
    config = expected.config.to_dict()
    config.pop("model_type")
    config.pop("architectures", None)
    if defect == "missing_dimensions":
        config.pop("num_attention_heads")
    (tmp_path / "config.json").write_text(json.dumps(config))
    state = expected.state_dict()
    state["bert.embeddings.position_ids"] = torch.arange(expected.config.max_position_embeddings)[None]
    if defect == "position_ids":
        state["bert.embeddings.position_ids"] = state["bert.embeddings.position_ids"].flip(1)
    if defect == "unknown_weight":
        state["custom.weight"] = torch.eye(3)
    torch.save(state, tmp_path / "pytorch_model.bin", _use_new_zipfile_serialization=False)
    if defect:
        with pytest.raises((loader.LoaderFailure, ValueError)):
            loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
        return
    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert type(actual) is transformers.BertForPreTraining
    assert actual._model_zoo_loading_info["architecture_selection"]["inferred_model_type"] == "bert"
    for name, tensor in state.items():
        loaded = dict(actual.named_buffers()) if name.endswith(".position_ids") else actual.state_dict()
        assert torch.equal(loaded[name], tensor), name


@pytest.mark.parametrize("defect", [None, "shape", "extra", "missing_bias"])
def test_albert_restores_only_a_complete_stored_pooler(tmp_path, defect):
    expected = tiny_encoder("albert")
    state = expected.state_dict()
    hidden = expected.config.hidden_size
    state["albert.pooler.weight"] = torch.diag(torch.arange(1., hidden + 1))
    state["albert.pooler.bias"] = torch.arange(float(hidden))
    if defect == "shape":
        state["albert.pooler.weight"] = torch.eye(3)
    elif defect == "extra":
        state["unused.weight"] = torch.eye(3)
    elif defect == "missing_bias":
        del state["albert.pooler.bias"]
    expected.save_pretrained(tmp_path, state_dict=state)
    if defect:
        with pytest.raises(loader.LoaderFailure) as exc:
            loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
        assert exc.value.reason == "checkpoint_weight_mismatch"
        return
    actual, _ = loader.load_model(str(tmp_path), device_map="cpu", torch_dtype="auto")
    assert type(actual) is type(expected)
    for name, tensor in state.items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert set(actual._model_zoo_loading_info["restored_checkpoint_keys"]) == {
        "albert.pooler.weight", "albert.pooler.bias"}
    modules = []
    metrics = net_esd_estimator(actual, parallel=False, coverage=modules)
    position = metrics["longname"].index("albert.pooler")
    torch.testing.assert_close(torch.as_tensor(metrics["eigs"][position]), torch.arange(1., hidden + 1).square())
    usage = weight_usage_report(actual, modules, metrics["longname"])
    assert usage["counts"]["unresolved_tensors"] == 0
    with torch.no_grad():
        inputs = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(actual(inputs).logits, expected(inputs).logits)


@pytest.mark.parametrize("defect", ["missing_shard", "wrong_shard", "extra_index_key", "path_escape"])
def test_shard_index_must_account_for_actual_checkpoint_tensors(tmp_path, defect):
    expected = tiny_bert(transformers.BertForPreTraining)
    expected.save_pretrained(tmp_path, max_shard_size="2KB")
    index = tmp_path / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    key = next(iter(data["weight_map"]))
    if defect == "missing_shard":
        data["weight_map"][key] = "absent.safetensors"
    elif defect == "wrong_shard":
        data["weight_map"][key] = next(name for name in data["weight_map"].values() if name != data["weight_map"][key])
    elif defect == "extra_index_key":
        data["weight_map"]["unknown.weight"] = data["weight_map"][key]
    else:
        data["weight_map"][key] = "../outside.safetensors"
    index.write_text(json.dumps(data))

    with pytest.raises(loader.LoaderFailure) as exc:
        loader._checkpoint_tensor_shapes(str(tmp_path), None)
    assert exc.value.reason == "checkpoint_inspection_failed"


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
