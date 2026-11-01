"""All-modality selection frames from the pinned public hub-stats dump."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HUB_STATS_COLUMNS = ['id', 'pipeline_tag', 'library_name', 'tags', 'downloads',
                     'safetensors', 'gguf', 'siblings', 'baseModels', 'createdAt', 'cardData']
MODALITY = {
    'text-generation': 'text', 'text2text-generation': 'text',
    'text-classification': 'text', 'token-classification': 'text',
    'question-answering': 'text', 'fill-mask': 'text', 'summarization': 'text',
    'translation': 'text', 'table-question-answering': 'text',
    'zero-shot-classification': 'text', 'multiple-choice': 'text',
    'image-text-to-text': 'vision-language', 'image-to-text': 'vision-language',
    'visual-question-answering': 'vision-language',
    'document-question-answering': 'vision-language', 'any-to-any': 'vision-language',
    'image-classification': 'vision', 'image-segmentation': 'vision',
    'object-detection': 'vision', 'zero-shot-object-detection': 'vision',
    'zero-shot-image-classification': 'vision', 'depth-estimation': 'vision',
    'image-to-image': 'image-generation', 'text-to-image': 'image-generation',
    'unconditional-image-generation': 'image-generation', 'diffusers': 'image-generation',
    'automatic-speech-recognition': 'audio', 'audio-classification': 'audio',
    'text-to-speech': 'audio', 'text-to-audio': 'audio', 'audio-to-audio': 'audio',
    'audio-text-to-text': 'audio', 'voice-activity-detection': 'audio',
    'feature-extraction': 'embedding', 'sentence-similarity': 'embedding',
    'text-to-video': 'video', 'image-to-video': 'video', 'video-to-video': 'video',
    'video-classification': 'video', 'video-text-to-text': 'video',
    'text-to-3d': '3d', 'image-to-3d': '3d', '3d': '3d',
    'robotics': 'robotics', 'robotics-action-prediction': 'robotics',
}
PILOT_HUB_PLAN = {'text:<1B': 40, 'text:1-3B': 40, 'text:3-9B': 60,
                  'text:9-35B': 30, 'text:35-75B': 10, 'vision-language': 30,
                  'vision': 20, 'image-generation': 20, 'video': 10,
                  'audio': 20, 'embedding': 20, 'decision': 20}
_FRAME_COLUMNS = ['repo_id', 'pipeline_tag', 'modality', 'is_adapter', 'system_one',
                  'is_decision', 'downloads', 'params', 'weight_format',
                  'size_stratum', 'stratum', 't0_only', 'declared_relation',
                  'declared_bases', 'created_quarter']


def load_hub_stats(path: Path) -> pd.DataFrame:
    return pq.read_table(path, columns=HUB_STATS_COLUMNS).to_pandas()


def _value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _record(value):
    value = _value(value)
    return value if isinstance(value, dict) else {}


def _items(value):
    value = _value(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else []


def _modality(row):
    task, library = row.get('pipeline_tag'), row.get('library_name')
    if task in MODALITY:
        return MODALITY[task]
    if library == 'diffusers':
        return 'image-generation'
    if library in ('sentence-transformers', 'sentence_transformers'):
        return 'embedding'
    if library == 'lerobot':
        return 'robotics'
    return 'other'


def _size_stratum(params):
    if params is None or not np.isfinite(params):
        return 'unknown'
    for bound, label in ((1e9, '<1B'), (3e9, '1-3B'), (9e9, '3-9B'), (35e9, '9-35B')):
        if params < bound:
            return label
    return '35-75B' if params <= 75e9 else '>75B'


def build_frame(df) -> pd.DataFrame:
    rows = df.to_dict('records')
    base_modalities = {row['id']: _modality(row) for row in rows}
    result, excluded = [], []
    for row in rows:
        repo = row['id']
        tags = [str(tag) for tag in _items(row.get('tags'))]
        safetensors, gguf = _record(row.get('safetensors')), _record(row.get('gguf'))
        files = [item.get('rfilename', '') if isinstance(item, dict) else str(item)
                 for item in _items(row.get('siblings'))]
        if any(name.endswith('.safetensors') for name in files) or safetensors.get('total'):
            weight_format = 'safetensors'
        elif any(name.endswith('.gguf') for name in files) or gguf.get('total'):
            weight_format = 'gguf'
        elif any(name.endswith(('.bin', '.pt', '.pth', '.ckpt')) for name in files):
            weight_format = 'bin'
        elif any(name.endswith('.onnx') for name in files):
            weight_format = 'onnx'
        else:
            excluded.append({'repo_id': repo, 'reason': 'no_weights'})
            continue
        base_info = _record(row.get('baseModels'))
        relation = base_info.get('relation') or ''
        bases = [item['id'] for item in _items(base_info.get('models'))
                 if isinstance(item, dict) and item.get('id')]
        for tag in tags:
            if tag.startswith('base_model:'):
                value = tag[len('base_model:'):]
                parts = value.split(':', 1)
                if len(parts) == 2:
                    if not relation:
                        relation = parts[0]
                    value = parts[1]
                if value and value not in bases:
                    bases.append(value)
        adapter = relation == 'adapter' or row.get('library_name') == 'peft' or any(
            tag in ('peft', 'lora', 'kohya') or tag.startswith('base_model:adapter:')
            for tag in tags
        )
        modality = _modality(row)
        if adapter:
            modality = next((base_modalities[base] for base in bases if base in base_modalities), modality)
        system_one = bool({'system-one', 'decision-model'} & set(tags))
        pipeline = row.get('pipeline_tag') or ''
        decision = system_one or 'classification' in pipeline or pipeline == 'multiple-choice'
        params = safetensors.get('total') or gguf.get('total')
        if params is None and safetensors.get('parameters'):
            counts = [value for value in safetensors['parameters'].values()
                      if isinstance(value, (int, float, np.number)) and np.isfinite(value)]
            params = sum(counts) if counts else None
        size_stratum = _size_stratum(params)
        created = pd.to_datetime(row.get('createdAt'), errors='coerce', utc=True)
        quarter = f'{created.year}Q{(created.month - 1) // 3 + 1}' if pd.notna(created) else ''
        result.append({
            'repo_id': repo, 'pipeline_tag': pipeline, 'modality': modality,
            'is_adapter': bool(adapter), 'system_one': system_one, 'is_decision': bool(decision),
            'downloads': row.get('downloads') or 0, 'params': params,
            'weight_format': weight_format, 'size_stratum': size_stratum,
            'stratum': 'decision' if system_one else f'text:{size_stratum}' if modality == 'text' else modality,
            't0_only': weight_format != 'safetensors', 'declared_relation': relation,
            'declared_bases': sorted(bases), 'created_quarter': quarter,
        })
    frame = pd.DataFrame(result, columns=_FRAME_COLUMNS).sort_values('repo_id').reset_index(drop=True)
    frame.attrs['excluded'] = excluded
    return frame


def draw_stratified(frame, n_per_stratum: dict[str, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    for stratum, requested in sorted(n_per_stratum.items()):
        if type(requested) is not int or requested < 0:
            raise ValueError('stratum sample sizes must be nonnegative integers')
        group = frame.loc[frame.stratum == stratum].sort_values('repo_id')
        n = min(requested, len(group))
        if n:
            sample = group.iloc[rng.choice(len(group), n, replace=False)].copy()
            sample['inclusion_prob'] = n / len(group)
            sample['selection_reason'] = 'stratified:' + stratum
            sample['revision'] = ''
            selected.append(sample)
    if not selected:
        return frame.iloc[:0].assign(inclusion_prob=pd.Series(dtype=float),
                                     selection_reason=pd.Series(dtype=str), revision=pd.Series(dtype=str))
    return pd.concat(selected, ignore_index=True).sort_values('repo_id').reset_index(drop=True)
