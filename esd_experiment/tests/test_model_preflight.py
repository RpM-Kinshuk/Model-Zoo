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
    assert resolve_effective_loader({
        'loader_scenario': 'standard_transformers',
        'primary_type_bucket': 'text',
        'config_model_type': 't5',
        'config_architectures': ['T5ForConditionalGeneration'],
    }) == 'seq2seq'


def test_resolve_effective_loader_prefers_multimodal_for_image_text_configs():
    assert resolve_effective_loader({
        'loader_scenario': 'standard_transformers',
        'primary_type_bucket': 'multimodal',
        'config_model_type': 'vision-encoder-decoder',
        'config_architectures': ['SomeImageTextModel'],
    }) == 'multimodal'


def test_resolve_effective_loader_prefers_sequence_classification_for_classifier_architecture():
    assert resolve_effective_loader({
        'loader_scenario': 'standard_transformers',
        'pipeline_tag': 'text-classification',
        'Architecture': 'T5ForSequenceClassification',
    }) == 'sequence_classification'


def test_resolve_effective_loader_prefers_seq2seq_for_conditional_generation_architecture():
    assert resolve_effective_loader({
        'loader_scenario': 'multimodal_transformers',
        'Architecture': 'T5ForConditionalGeneration',
    }) == 'seq2seq'


def test_resolve_effective_loader_does_not_force_seq2seq_from_text2text_tag_alone():
    assert resolve_effective_loader({
        'loader_scenario': 'multimodal_transformers',
        'pipeline_tag': 'image-text-to-text',
        'tags': "['qwen2', 'text2text-generation', 'multimodal']",
        'Architecture': '',
    }) == 'multimodal'


def test_classify_row_preflight_allows_adapter_rows_without_curated_adapter_config():
    decision = classify_row_preflight({
        'model_id': 'org/adapter-model',
        'base_model_relation': 'adapter',
        'adapter_config': '',
        'files': [
            'README.md',
            'adapter_model.safetensors',
        ],
    })

    assert isinstance(decision, PreflightDecision)
    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'adapter_requires_base'


def test_classify_row_preflight_accepts_adapter_config_json_from_files_field():
    decision = classify_row_preflight({
        'model_id': 'org/adapter-model',
        'base_model_relation': 'adapter',
        'adapter_config': '',
        'files': [
            'README.md',
            'nested/adapter_config.json',
        ],
    })

    assert isinstance(decision, PreflightDecision)
    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'adapter_requires_base'


def test_classify_row_preflight_accepts_adapter_bin_artifact():
    decision = classify_row_preflight({
        'model_id': 'org/adapter-model',
        'base_model_relation': 'adapter',
        'adapter_config': '',
        'files': [
            'README.md',
            'adapter_model.bin',
        ],
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'adapter_requires_base'


def test_classify_row_preflight_keeps_ambiguous_adapter_rows_eligible_without_file_metadata():
    decision = classify_row_preflight({
        'model_id': 'org/adapter-model',
        'base_model_relation': 'adapter',
        'adapter_config': '',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'adapter_requires_base'


def test_classify_row_preflight_marks_gptq_backend_missing():
    decision = classify_row_preflight({
        'model_id': 'org/quantized-model',
        'loader_scenario': 'quantized_transformers_native',
        'tags': "['gptq']",
        'backend_status': '',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'gptq'


def test_classify_row_preflight_marks_awq_backend_missing_without_loader_hint():
    decision = classify_row_preflight({
        'model_id': 'org/quantized-awq-model',
        'tags': "['awq']",
        'backend_status': '',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'awq'


def test_classify_row_preflight_marks_compressed_tensors_missing_backend():
    decision = classify_row_preflight({
        'model_id': 'org/compressed-model',
        'tags': "['compressed-tensors']",
        'backend_status': '',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'compressed_tensors'


def test_classify_row_preflight_allows_compressed_tensors_with_backend():
    decision = classify_row_preflight({
        'model_id': 'org/compressed-model',
        'tags': "['compressed-tensors']",
        'backend_status': 'available',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'compressed_tensors'


def test_classify_row_preflight_allows_generic_quantized_native_without_backend_hint():
    decision = classify_row_preflight({
        'model_id': 'org/quantized-model',
        'loader_scenario': 'quantized_transformers_native',
        'backend_status': '',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'standard_causal'


def test_classify_row_preflight_marks_gguf_backend_missing_from_model_id():
    decision = classify_row_preflight({
        'model_id': 'org/model-Q4_K_M-GGUF',
        'backend_status': '',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'gguf'


def test_classify_row_preflight_marks_compressed_tensors_backend_missing(monkeypatch):
    monkeypatch.setattr(model_preflight, 'resolve_effective_loader', lambda row: 'compressed_tensors')

    decision = classify_row_preflight({
        'model_id': 'org/compressed-model',
        'backend_status': '',
        'loader_scenario': 'standard_transformers',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'compressed_tensors'


def test_classify_row_preflight_allows_gguf_row_when_backend_available():
    decision = classify_row_preflight({
        'model_id': 'org/model-GGUF',
        'loader_scenario': 'gguf',
        'backend_status': 'available',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'gguf'


def test_classify_row_preflight_blocks_gguf_row_when_backend_missing():
    decision = classify_row_preflight({
        'model_id': 'org/model-GGUF',
        'loader_scenario': 'gguf',
        'backend_status': 'missing',
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_backend'
    assert decision.effective_loader == 'gguf'


def test_classify_row_preflight_blocks_exl2_as_unsupported_alt_format():
    decision = classify_row_preflight({
        'model_id': 'org/model-6bpw-h6-exl2',
        'tags': "['exl2']",
    })

    assert decision.eligible is False
    assert decision.reason == 'unsupported_loader_scenario'
    assert decision.effective_loader == 'quantized_alt_format'


def test_classify_row_preflight_blocks_repo_marked_unavailable():
    decision = classify_row_preflight({
        'model_id': 'org/missing-model',
        'loader_scenario': 'standard_transformers',
        'Available on the hub': False,
    })

    assert decision.eligible is False
    assert decision.reason == 'repo_inaccessible'
    assert decision.effective_loader == 'standard_causal'


def test_classify_row_preflight_allows_standard_transformers_text_model():
    decision = classify_row_preflight({
        'model_id': 'org/text-model',
        'loader_scenario': 'standard_transformers',
    })

    assert decision.eligible is True
    assert decision.reason == 'eligible'
    assert decision.effective_loader == 'standard_causal'
