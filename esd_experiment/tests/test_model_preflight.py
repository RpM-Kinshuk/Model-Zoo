import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / 'src' / 'model_preflight.py'
SPEC = importlib.util.spec_from_file_location('model_preflight_under_test', MODULE_PATH)
model_preflight = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(model_preflight)

PreflightDecision = model_preflight.PreflightDecision
classify_row_preflight = model_preflight.classify_row_preflight
resolve_effective_loader = model_preflight.resolve_effective_loader


def test_resolve_effective_loader_prefers_seq2seq_for_t5_configs():
    assert resolve_effective_loader({'model_type': 't5'}) == 'seq2seq'


def test_resolve_effective_loader_prefers_multimodal_for_image_text_configs():
    assert resolve_effective_loader({'architectures': ['SomeImageTextModel']}) == 'multimodal'


def test_classify_row_preflight_marks_missing_adapter_config_ineligible():
    decision = classify_row_preflight({
        'model_id': 'org/adapter-model',
        'base_model_relation': 'adapter',
        'adapter_config': '',
    })

    assert isinstance(decision, PreflightDecision)
    assert decision.eligible is False
    assert decision.reason == 'adapter_config_missing'


def test_classify_row_preflight_marks_gptq_backend_missing():
    decision = classify_row_preflight({
        'model_id': 'org/quantized-model',
        'loader_scenario': 'quantized_transformers_native',
        'backend_status': '',
    })

    assert decision.eligible is False
    assert decision.reason == 'gptq_backend_missing'


def test_classify_row_preflight_allows_standard_transformers_text_model():
    decision = classify_row_preflight({
        'model_id': 'org/text-model',
        'loader_scenario': 'standard_transformers',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
