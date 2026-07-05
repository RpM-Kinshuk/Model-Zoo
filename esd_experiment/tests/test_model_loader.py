import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))


def _raise_runtime_error(message: str):
    raise RuntimeError(message)

MODULE_PATH = PROJECT_ROOT / "src" / "model_loader.py"
SPEC = importlib.util.spec_from_file_location("model_loader_under_test", MODULE_PATH)
model_loader = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader

fake_torch = ModuleType("torch")
fake_torch.float16 = object()
fake_torch.nn = ModuleType("torch.nn")
fake_torch.nn.Module = object

fake_transformers = ModuleType("transformers")
fake_transformers.AutoModelForCausalLM = type(
    "AutoModelForCausalLM",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_transformers.AutoModelForSeq2SeqLM = type(
    "AutoModelForSeq2SeqLM",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_transformers.AutoModelForSequenceClassification = type(
    "AutoModelForSequenceClassification",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_transformers.AutoModel = type(
    "AutoModel",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_transformers.AutoModelForImageTextToText = type(
    "AutoModelForImageTextToText",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_transformers.AutoConfig = type(
    "AutoConfig",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: _raise_runtime_error("no config"))},
)

fake_peft = ModuleType("peft")
fake_peft.PeftModel = type(
    "PeftModel",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: None)},
)
fake_peft.PeftConfig = type(
    "PeftConfig",
    (),
    {"from_pretrained": classmethod(lambda cls, *args, **kwargs: _raise_runtime_error("no peft config"))},
)

fake_hf_hub = ModuleType("huggingface_hub")
fake_hf_hub.HfApi = type("HfApi", (), {"list_repo_files": lambda self, **kwargs: []})
fake_hf_hub.get_token = lambda: None

original_modules = {
    "torch": sys.modules.get("torch"),
    "torch.nn": sys.modules.get("torch.nn"),
    "transformers": sys.modules.get("transformers"),
    "peft": sys.modules.get("peft"),
    "huggingface_hub": sys.modules.get("huggingface_hub"),
}
try:
    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch.nn
    sys.modules["transformers"] = fake_transformers
    sys.modules["peft"] = fake_peft
    sys.modules["huggingface_hub"] = fake_hf_hub
    sys.modules[SPEC.name] = model_loader
    SPEC.loader.exec_module(model_loader)
finally:
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

LoaderFailure = model_loader.LoaderFailure
classify_loader_scenario_support = model_loader.classify_loader_scenario_support
hf_from_pretrained = model_loader.hf_from_pretrained
load_model = model_loader.load_model
resolve_adapter_effective_loader = model_loader.resolve_adapter_effective_loader
resolve_effective_loader_for_repo = model_loader.resolve_effective_loader_for_repo
resolve_dense_upstream_base_reference = model_loader.resolve_dense_upstream_base_reference


def test_classify_loader_scenario_support_rejects_quantized_alt_format():
    failure = classify_loader_scenario_support("quantized_alt_format")

    assert isinstance(failure, LoaderFailure)
    assert failure.stage == "load"
    assert failure.reason == "unsupported_loader_scenario"


def test_classify_loader_scenario_support_accepts_gguf():
    assert classify_loader_scenario_support("gguf") is None


def test_classify_loader_scenario_support_accepts_standard_transformers():
    assert classify_loader_scenario_support("standard_transformers") is None


@pytest.mark.parametrize(
    "scenario",
    [
        "quantized_transformers_native",
        "multimodal_transformers",
        "gguf",
        "standard_causal",
        "multimodal",
        "gptq",
        "awq",
        "compressed_tensors",
    ],
)
def test_classify_loader_scenario_support_accepts_current_policy_allowlist(scenario):
    assert classify_loader_scenario_support(scenario) is None


def test_resolve_effective_loader_for_repo_prefers_auto_config_t5_seq2seq():
    config = Mock(model_type="t5", architectures=["T5ForConditionalGeneration"])
    with patch("model_loader_under_test.AutoConfig.from_pretrained", return_value=config):
        assert resolve_effective_loader_for_repo("org/t5-model", loader_scenario="standard_transformers") == "seq2seq"


def test_resolve_effective_loader_for_repo_prefers_auto_config_sequence_classification():
    config = Mock(model_type="t5", architectures=["T5ForSequenceClassification"])
    with patch("model_loader_under_test.AutoConfig.from_pretrained", return_value=config):
        assert resolve_effective_loader_for_repo("org/t5-cls", loader_scenario="standard_transformers") == "sequence_classification"


def test_resolve_effective_loader_for_repo_reads_quantization_config():
    config = Mock(model_type="llama", architectures=["LlamaForCausalLM"])
    config.quantization_config = {"quant_method": "gptq"}
    with patch("model_loader_under_test.AutoConfig.from_pretrained", return_value=config):
        assert resolve_effective_loader_for_repo("org/quantized-model", loader_scenario="standard_transformers") == "gptq"


def test_resolve_effective_loader_for_repo_detects_compressed_tensors_quant_method(monkeypatch):
    config = Mock(model_type="llama", architectures=["LlamaForCausalLM"])
    config.quantization_config = Mock(quant_method="compressed-tensors")

    monkeypatch.setattr(model_loader.AutoConfig, "from_pretrained", Mock(return_value=config))

    assert (
        resolve_effective_loader_for_repo(
            "org/compressed-model",
            loader_scenario="standard_transformers",
        )
        == "compressed_tensors"
    )


def test_resolve_effective_loader_for_repo_keeps_explicit_multimodal_hint():
    config = Mock(model_type="t5", architectures=["T5ForConditionalGeneration"])
    with patch("model_loader_under_test.AutoConfig.from_pretrained", return_value=config):
        assert resolve_effective_loader_for_repo("org/multi", loader_scenario="multimodal_transformers") == "multimodal"


@patch("model_loader_under_test.is_adapter_model", return_value=True)
def test_load_model_raises_structured_failure_for_unresolved_adapter(mock_is_adapter):
    with patch(
        "model_loader_under_test.resolve_base_model_reference",
        side_effect=RuntimeError("missing base"),
    ):
        with pytest.raises(LoaderFailure) as exc:
            load_model(
                "org/adapter",
                base_model_relation="adapter",
                source_model=None,
                loader_scenario="adapter_requires_base",
            )

    assert exc.value.reason == "adapter_base_unresolved"
    assert exc.value.stage == "load"
    assert mock_is_adapter.called


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_forwards_revision_to_standard_load(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/model",
        revision="rev-a",
        loader_scenario="standard_transformers",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.kwargs["revision"] == "rev-a"


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_uses_multimodal_auto_class(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/multimodal-model",
        loader_scenario="multimodal_transformers",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForImageTextToText


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_accepts_standard_causal_alias(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/standard-model",
        loader_scenario="standard_causal",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForCausalLM


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_accepts_multimodal_alias(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/multimodal-model",
        loader_scenario="multimodal",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForImageTextToText


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_uses_seq2seq_auto_class(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/seq2seq-model",
        loader_scenario="seq2seq",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForSeq2SeqLM


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_uses_sequence_classification_auto_class(mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/classifier-model",
        loader_scenario="sequence_classification",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForSequenceClassification


def test_load_model_falls_back_to_automodel_for_supported_non_head_config(monkeypatch):
    fallback_model = Mock()
    auto_model_cls = type("AutoModel", (), {})
    calls = []

    def fake_from_pretrained(auto_cls, repo_id, **kwargs):
        calls.append((auto_cls, repo_id, kwargs))
        if auto_cls is model_loader.AutoModelForCausalLM:
            raise RuntimeError(
                "Unrecognized configuration class <class 'transformers.SomeConfig'> "
                "for this kind of AutoModel: AutoModelForCausalLM."
            )
        if auto_cls is auto_model_cls:
            return fallback_model
        raise AssertionError(f"Unexpected auto class {auto_cls}")

    monkeypatch.setattr(model_loader, "AutoModel", auto_model_cls, raising=False)
    monkeypatch.setattr(model_loader, "hf_from_pretrained", fake_from_pretrained)

    model, is_adapter = load_model(
        "org/non-head-model",
        loader_scenario="standard_transformers",
    )

    assert model is fallback_model
    assert is_adapter is False
    assert [call[0] for call in calls] == [
        model_loader.AutoModelForCausalLM,
        auto_model_cls,
    ]


@patch("model_loader_under_test.hf_from_pretrained", side_effect=RuntimeError("Loading an AWQ quantized model requires gptqmodel. Please install it."))
def test_load_model_raises_structured_failure_for_missing_quantized_dependency(mock_from_pretrained):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/quantized-model",
            loader_scenario="quantized_transformers_native",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_dependency_missing"
    assert "gptqmodel" in exc.value.message


@patch("model_loader_under_test.hf_from_pretrained", side_effect=RuntimeError("Tensor.item() cannot be called on meta tensors"))
def test_load_model_raises_structured_failure_for_incompatible_quantized_backend(mock_from_pretrained):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/quantized-model",
            loader_scenario="quantized_transformers_native",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_backend_incompatible"
    assert "incompatible" in exc.value.message


@patch("model_loader_under_test.hf_from_pretrained", side_effect=RuntimeError("Cannot copy out of meta tensor; no data!"))
def test_load_model_classifies_singular_meta_tensor_quantized_backend_failure(mock_from_pretrained):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/quantized-model",
            loader_scenario="quantized_transformers_native",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_backend_incompatible"
    assert mock_from_pretrained.called


@patch(
    "model_loader_under_test.hf_from_pretrained",
    side_effect=[
        RuntimeError(
            "Unrecognized configuration class <class 'transformers.SomeConfig'> "
            "for this kind of AutoModel: AutoModelForCausalLM."
        ),
        RuntimeError("compressed_tensors duplicate template name"),
    ],
)
def test_load_model_classifies_quantized_fallback_automodel_backend_failure(mock_from_pretrained):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/quantized-model",
            loader_scenario="quantized_transformers_native",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_backend_incompatible"
    assert mock_from_pretrained.call_count == 2


def test_resolve_adapter_effective_loader_is_backend_sensitive():
    assert resolve_adapter_effective_loader("gptq") == "gptq"
    assert resolve_adapter_effective_loader("awq") == "awq"
    assert resolve_adapter_effective_loader("compressed-tensors") == "compressed_tensors"
    assert resolve_adapter_effective_loader("standard_causal") == "standard_causal"


def test_resolve_adapter_task_loader_prefers_explicit_source_model_over_peft_base():
    with patch(
        "model_loader_under_test.PeftConfig.from_pretrained",
        return_value=Mock(task_type="", base_model_name_or_path="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"),
    ), patch(
        "model_loader_under_test.resolve_effective_loader_for_repo",
        side_effect=lambda repo_id, **kwargs: "standard_causal" if repo_id == "mistralai/Mistral-7B-Instruct-v0.2" else "gptq",
    ):
        assert (
            model_loader._resolve_adapter_task_loader(
                "org/adapter",
                source_model="mistralai/Mistral-7B-Instruct-v0.2",
            )
            == "standard_causal"
        )


def test_resolve_dense_upstream_base_reference_uses_hub_base_model_tag():
    model_info = Mock(tags=["gptq", "base_model:mistralai/Mistral-7B-Instruct-v0.2"])
    api = Mock()
    api.model_info.return_value = model_info
    with patch("model_loader_under_test.HfApi", return_value=api):
        base_repo, base_revision = resolve_dense_upstream_base_reference(
            "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        )

    assert base_repo == "mistralai/Mistral-7B-Instruct-v0.2"
    assert base_revision is None


@patch("model_loader_under_test.load_and_merge_adapter", return_value=Mock())
@patch("model_loader_under_test._resolve_adapter_task_loader", return_value="gptq")
def test_load_model_allows_adapter_gptq_base(
    mock_resolve_adapter_task_loader,
    mock_load_and_merge_adapter,
):
    model, is_adapter = load_model(
        "org/adapter",
        base_model_relation="adapter",
        source_model="base/model",
        loader_scenario="adapter_requires_base",
    )

    assert is_adapter is True
    assert model is mock_load_and_merge_adapter.return_value
    assert mock_resolve_adapter_task_loader.called
    assert mock_load_and_merge_adapter.called
    assert mock_load_and_merge_adapter.call_args.kwargs["effective_loader"] == "gptq"
    assert mock_load_and_merge_adapter.call_args.kwargs["loader_scenario"] == "adapter_requires_base"


@patch("model_loader_under_test.importlib.import_module")
@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_prepares_compressed_tensors_backend_after_gptq_side_effect(
    mock_from_pretrained,
    mock_import_module,
):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model
    imported = set()

    def fake_import_module(name):
        if name == "compressed_tensors" and "gptqmodel" not in imported:
            imported.add(name)
            raise AssertionError("duplicate template name")
        imported.add(name)
        return Mock()

    mock_import_module.side_effect = fake_import_module

    model, is_adapter = load_model(
        "org/compressed-model",
        loader_scenario="compressed_tensors",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_import_module.call_args_list[0].args == ("compressed_tensors",)
    assert mock_import_module.call_args_list[1].args == ("gptqmodel",)
    assert mock_import_module.call_args_list[2].args == ("compressed_tensors",)


@patch("model_loader_under_test.PeftModel.from_pretrained")
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch(
    "model_loader_under_test.resolve_dense_upstream_base_reference",
    return_value=("mistralai/Mistral-7B-Instruct-v0.2", None),
)
@patch("model_loader_under_test._resolve_adapter_task_loader", side_effect=["gptq", "gptq", "standard_causal"])
def test_load_model_remaps_gptq_adapter_base_to_dense_upstream_base(
    mock_resolve_adapter_task_loader,
    mock_resolve_dense_upstream_base_reference,
    mock_is_adapter,
    mock_from_pretrained,
    mock_peft_from_pretrained,
):
    base_model = Mock()
    merged_model = Mock()
    peft_model = Mock()
    peft_model.merge_and_unload.return_value = merged_model
    mock_from_pretrained.return_value = base_model
    mock_peft_from_pretrained.return_value = peft_model

    model, is_adapter = load_model(
        "org/adapter",
        base_model_relation="adapter",
        source_model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        loader_scenario="adapter_requires_base",
    )

    assert model is merged_model
    assert is_adapter is True
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForCausalLM
    assert mock_from_pretrained.call_args.args[1] == "mistralai/Mistral-7B-Instruct-v0.2"
    assert mock_resolve_dense_upstream_base_reference.called
    assert mock_resolve_adapter_task_loader.call_count == 2
    assert mock_is_adapter.called


@patch(
    "model_loader_under_test.hf_from_pretrained",
    side_effect=RuntimeError("Loading an AWQ quantized model requires gptqmodel. Please install it."),
)
def test_load_model_classifies_quantized_dependency_failure_for_awq_alias(mock_from_pretrained):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/awq-model",
            loader_scenario="awq",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_dependency_missing"
    assert mock_from_pretrained.called


@patch(
    "model_loader_under_test.hf_from_pretrained",
    side_effect=RuntimeError("No module named 'compressed_tensors'"),
)
def test_load_model_classifies_quantized_dependency_failure_for_compressed_tensors(
    mock_from_pretrained,
):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/compressed-model",
            loader_scenario="compressed_tensors",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_dependency_missing"
    assert "compressed-tensors" in exc.value.message


@patch("model_loader_under_test.ensure_optimum_gptq_backend_compat")
@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_applies_gptq_backend_compat_shim(
    mock_from_pretrained,
    mock_ensure_optimum_gptq_backend_compat,
):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/gptq-model",
        loader_scenario="gptq",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_ensure_optimum_gptq_backend_compat.called


@patch(
    "model_loader_under_test.hf_from_pretrained",
    side_effect=RuntimeError("No module named 'compressed_tensors'"),
)
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.resolve_base_model_reference", return_value=("base/compressed-model", None))
@patch("model_loader_under_test._resolve_adapter_task_loader", return_value="compressed_tensors")
def test_load_model_classifies_adapter_compressed_tensors_dependency_failure(
    mock_resolve_adapter_task_loader,
    mock_resolve_base_model_reference,
    mock_is_adapter,
    mock_from_pretrained,
):
    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/adapter",
            base_model_relation="adapter",
            source_model="base/compressed-model",
            loader_scenario="adapter_requires_base",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "quantized_dependency_missing"
    assert "compressed-tensors" in exc.value.message
    assert mock_resolve_adapter_task_loader.called
    assert mock_is_adapter.called


@patch("model_loader_under_test.PeftModel.from_pretrained")
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.resolve_base_model_reference", return_value=("base/model", None))
def test_load_model_uses_seq2seq_base_for_seq2seq_adapter(
    mock_resolve_base_model_reference,
    mock_is_adapter,
    mock_from_pretrained,
    mock_peft_from_pretrained,
):
    base_model = Mock()
    merged_model = Mock()
    peft_model = Mock()
    peft_model.merge_and_unload.return_value = merged_model
    mock_from_pretrained.return_value = base_model
    mock_peft_from_pretrained.return_value = peft_model
    with patch(
        "model_loader_under_test.PeftConfig.from_pretrained",
        return_value=Mock(task_type="SEQ_2_SEQ_LM", base_model_name_or_path="base/model"),
    ):
        model, is_adapter = load_model(
            "org/adapter",
            base_model_relation="adapter",
            loader_scenario="adapter_requires_base",
        )

    assert model is merged_model
    assert is_adapter is True
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForSeq2SeqLM
    assert mock_resolve_base_model_reference.called
    assert mock_is_adapter.called


@patch("model_loader_under_test.PeftModel.from_pretrained")
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.ensure_optimum_gptq_backend_compat")
@patch("model_loader_under_test._resolve_adapter_task_loader", return_value="gptq")
def test_load_model_reports_gptq_adapter_merge_unsupported_when_dequantize_missing(
    mock_resolve_adapter_task_loader,
    mock_ensure_optimum_gptq_backend_compat,
    mock_is_adapter,
    mock_from_pretrained,
    mock_peft_from_pretrained,
):
    base_model = Mock()
    base_model.dequantize.side_effect = NotImplementedError("no gptq dequantize")
    mock_from_pretrained.return_value = base_model

    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/adapter",
            base_model_relation="adapter",
            source_model="base/model",
            loader_scenario="adapter_requires_base",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "adapter_merge_unsupported"
    assert mock_resolve_adapter_task_loader.called
    assert mock_ensure_optimum_gptq_backend_compat.called
    assert mock_is_adapter.called
    assert not mock_peft_from_pretrained.called


@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.resolve_gguf_filename", return_value="model.Q4_K_M.gguf", create=True)
def test_load_model_uses_gguf_file_for_gguf_loader(mock_resolve_gguf_filename, mock_from_pretrained):
    mock_model = Mock()
    mock_from_pretrained.return_value = mock_model

    model, is_adapter = load_model(
        "org/model-gguf",
        loader_scenario="gguf",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForCausalLM
    assert mock_from_pretrained.call_args.kwargs["gguf_file"] == "model.Q4_K_M.gguf"
    assert mock_resolve_gguf_filename.called


def test_resolve_gguf_filename_wraps_repo_inspection_failure():
    with patch(
        "model_loader_under_test.HfApi.list_repo_files",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(LoaderFailure) as exc:
            model_loader.resolve_gguf_filename("org/model-gguf")

    assert exc.value.reason == "repo_inaccessible"


def test_hf_from_pretrained_retries_without_low_cpu_mem_usage_on_meta_tensor_error():
    class _FakeAutoModel:
        calls = []

        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            cls.calls.append(kwargs.copy())
            if kwargs.get("low_cpu_mem_usage", False):
                raise RuntimeError("Cannot copy out of meta tensor; no data!")
            return "ok"

    result = hf_from_pretrained(_FakeAutoModel, "org/model")

    assert result == "ok"
    assert len(_FakeAutoModel.calls) == 2
    assert _FakeAutoModel.calls[0]["low_cpu_mem_usage"] is True
    assert _FakeAutoModel.calls[1]["low_cpu_mem_usage"] is False


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_retries_seq2seq_after_multimodal_t5_misroute(mock_from_pretrained):
    mock_model = Mock()

    def _side_effect(auto_model_cls, *args, **kwargs):
        if auto_model_cls is model_loader.AutoModelForImageTextToText:
            raise RuntimeError(
                "Unrecognized configuration class <class 'transformers.models.t5.configuration_t5.T5Config'> "
                "for this kind of AutoModel: AutoModelForImageTextToText."
            )
        return mock_model

    mock_from_pretrained.side_effect = _side_effect

    model, is_adapter = load_model(
        "org/misrouted-t5-model",
        loader_scenario="multimodal_transformers",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForSeq2SeqLM


@patch("model_loader_under_test.hf_from_pretrained")
def test_load_model_retries_causal_after_multimodal_qwen2_misroute(mock_from_pretrained):
    mock_model = Mock()

    def _side_effect(auto_model_cls, *args, **kwargs):
        if auto_model_cls is model_loader.AutoModelForImageTextToText:
            raise RuntimeError(
                "Unrecognized configuration class <class 'transformers.models.qwen2.configuration_qwen2.Qwen2Config'> "
                "for this kind of AutoModel: AutoModelForImageTextToText."
            )
        return mock_model

    mock_from_pretrained.side_effect = _side_effect

    model, is_adapter = load_model(
        "org/misrouted-qwen2-model",
        loader_scenario="multimodal_transformers",
    )

    assert model is mock_model
    assert is_adapter is False
    assert mock_from_pretrained.call_args.args[0] is model_loader.AutoModelForCausalLM


@patch("model_loader_under_test.PeftModel.from_pretrained")
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.resolve_base_model_reference", return_value=("base/model", "base-rev"))
def test_load_model_uses_base_revision_from_adapter_metadata(
    mock_resolve_base_model_reference,
    mock_is_adapter,
    mock_from_pretrained,
    mock_peft_from_pretrained,
):
    base_model = Mock()
    merged_model = Mock()
    peft_model = Mock()
    peft_model.merge_and_unload.return_value = merged_model
    mock_from_pretrained.return_value = base_model
    mock_peft_from_pretrained.return_value = peft_model

    model, is_adapter = load_model(
        "org/adapter",
        base_model_relation="adapter",
        revision="rev-a",
        loader_scenario="adapter_requires_base",
    )

    assert model is merged_model
    assert is_adapter is True
    assert mock_from_pretrained.call_args.kwargs["revision"] == "base-rev"
    assert mock_peft_from_pretrained.call_args.kwargs["revision"] == "rev-a"
    assert mock_resolve_base_model_reference.called
    assert mock_is_adapter.called


@patch(
    "model_loader_under_test.PeftModel.from_pretrained",
    side_effect=RuntimeError(
        "Error(s) in loading state_dict for PeftModelForCausalLM:\n"
        "\tsize mismatch for base_model.model.model.embed_tokens.weight: copying a param with shape "
        "torch.Size([32002, 4096]) from checkpoint, the shape in current model is torch.Size([32000, 4096])."
    ),
)
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.resolve_base_model_reference", return_value=("base/model", "base-rev"))
def test_load_model_classifies_adapter_checkpoint_mismatch(
    mock_resolve_base_model_reference,
    mock_is_adapter,
    mock_from_pretrained,
    mock_peft_from_pretrained,
):
    base_model = Mock()
    mock_from_pretrained.return_value = base_model

    with pytest.raises(LoaderFailure) as exc:
        load_model(
            "org/adapter",
            base_model_relation="adapter",
            revision="rev-a",
            loader_scenario="adapter_requires_base",
        )

    assert exc.value.stage == "load"
    assert exc.value.reason == "adapter_base_checkpoint_mismatch"
    assert mock_peft_from_pretrained.called
    assert mock_resolve_base_model_reference.called
    assert mock_is_adapter.called
