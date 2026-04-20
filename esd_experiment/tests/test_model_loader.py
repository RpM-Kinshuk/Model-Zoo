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
load_model = model_loader.load_model


def test_classify_loader_scenario_support_rejects_quantized_alt_format():
    failure = classify_loader_scenario_support("quantized_alt_format")

    assert isinstance(failure, LoaderFailure)
    assert failure.stage == "load"
    assert failure.reason == "unsupported_loader_scenario"


def test_classify_loader_scenario_support_accepts_standard_transformers():
    assert classify_loader_scenario_support("standard_transformers") is None


@pytest.mark.parametrize(
    "scenario",
    ["quantized_transformers_native", "multimodal_transformers"],
)
def test_classify_loader_scenario_support_accepts_current_policy_allowlist(scenario):
    assert classify_loader_scenario_support(scenario) is None


@patch("model_loader_under_test.is_adapter_model", return_value=True)
def test_load_model_raises_structured_failure_for_unresolved_adapter(mock_is_adapter):
    with patch("model_loader_under_test.resolve_base_model", side_effect=RuntimeError("missing base")):
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


@patch("model_loader_under_test.PeftModel.from_pretrained")
@patch("model_loader_under_test.hf_from_pretrained")
@patch("model_loader_under_test.is_adapter_model", return_value=True)
@patch("model_loader_under_test.resolve_base_model", return_value="base/model")
def test_load_model_forwards_revision_to_adapter_and_base_loads(
    mock_resolve_base_model,
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
    assert mock_from_pretrained.call_args.kwargs["revision"] == "rev-a"
    assert mock_peft_from_pretrained.call_args.kwargs["revision"] == "rev-a"
    assert mock_resolve_base_model.called
    assert mock_is_adapter.called
