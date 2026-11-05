"""Read historical evidence names without exposing them in new controls/artifacts.

Numbered labels, old filename suffixes and old table columns are confined here.
Never rename historical files or mutate an immutable package's recorded inventory.
"""
from pathlib import Path
import re


_LEGACY_MODES = {'t0': 'metadata_only', 't0_only': 'metadata_only',
                 't1': 'row_blocks', 't2': 'full_weights', 't1_only': 'row_blocks_only'}
_LEGACY_MODEL_COLUMNS = {'fingerprint_t1': 'row_block_fingerprint',
                         'fingerprint_t2': 'full_weight_fingerprint', 'spectra_t2': 'full_weight_spectra'}
_LEGACY_FINGERPRINT_SUFFIXES = ('.t1.fingerprint.safetensors', '.t2.fingerprint.safetensors')


def canonical_mode(value):
    """Translate a historical mode; leave unrelated evidence-level names alone."""
    if isinstance(value, str):
        matched = re.fullmatch(r't1b([0-9]+)', value)
        if matched:
            return 'row_blocks_' + matched[1]
        return _LEGACY_MODES.get(value, value)
    return value


def normalize_fingerprint_metadata(metadata):
    result = dict(metadata)
    if 'tier' in result:
        result['tier'] = canonical_mode(result['tier'])
    return result


def normalize_manifest(manifest):
    result = dict(manifest)
    if 'access' in result:
        result['access'] = canonical_mode(result['access'])
    return result


def normalize_model_table(frame):
    """Expose the current columns when reading a historical database table."""
    renames = {old: new for old, new in _LEGACY_MODEL_COLUMNS.items() if old in frame}
    if any(new in frame for new in renames.values()):
        raise ValueError('Conflicting legacy and descriptive fingerprint columns')
    result = frame.rename(columns=renames)
    for column in ('access', 'coverage_status'):
        if column in result:
            result[column] = result[column].map(canonical_mode)
    return result


def normalize_view_table(frame):
    result = frame.copy()
    if 'tier' in result:
        result['tier'] = result['tier'].map(canonical_mode)
    return result


def is_baseline_fingerprint(path):
    return str(path).endswith(('.row_blocks.fingerprint.safetensors',
                               '.full_weights.fingerprint.safetensors', *_LEGACY_FINGERPRINT_SUFFIXES))


def companion_spectra_path(path):
    """Resolve the full-weight spectrum sibling in the same naming generation."""
    path = Path(path)
    for label in ('full_weights', 't2'):
        suffix = f'.{label}.fingerprint.safetensors'
        if path.name.endswith(suffix):
            return path.with_name(path.name.removesuffix(suffix) + f'.{label}.spectra.h5')
    raise ValueError('Expected a full-weight fingerprint filename')


def historical_artifact_name(name):
    """Required inventory spelling for an explicitly selected old package."""
    for current, old in (('row_blocks_128', 't1b128'), ('row_blocks_512', 't1b512'),
                         ('row_blocks', 't1'), ('full_weights', 't2')):
        name = name.replace('.' + current + '.', '.' + old + '.')
    return name


def measurement_directory(root, mode):
    """Prefer descriptive pilot directories; fall back to historical runs."""
    root = Path(root)
    current = root / mode
    if current.exists():
        return current
    legacy = {'row_blocks': 't1', 'full_weights': 't2'}.get(mode)
    return root / legacy if legacy and (root / legacy).exists() else current


def find_fingerprint(directory, safe, mode):
    """Locate an existing artifact; new producers always use descriptive names."""
    current = Path(directory) / f"{safe}.{mode}.fingerprint.safetensors"
    historical = Path(directory) / historical_artifact_name(current.name)
    return current if current.exists() or not historical.exists() else historical


def normalize_frame_table(frame):
    return frame.rename(columns={'t0_only': 'metadata_only'})


def companion_manifest_path(path):
    """Find a fingerprint's manifest using its actual current/historical suffix."""
    path = Path(path)
    matched = re.fullmatch(r'(.+)\.(?:row_blocks(?:_[0-9]+)?|full_weights|t1(?:b[0-9]+)?|t2)\.fingerprint\.safetensors', path.name)
    stem = matched[1] if matched else path.name
    return path.with_name(stem + '.manifest.json')
