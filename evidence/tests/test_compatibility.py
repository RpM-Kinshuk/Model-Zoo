"""Descriptive output names and explicitly isolated historical-format fixtures."""
from pathlib import Path

import h5py
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from evidence.cli import _parser
from evidence.fingerprint import fingerprint_model, fingerprint_tensors, load_fingerprint
from evidence.hub import HubReader


def _historical_fingerprint(source, destination, label):
    with safe_open(source, framework='numpy') as handle:
        metadata = dict(handle.metadata())
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    metadata.update(tier=label, evidence_version='2')
    save_file(tensors, destination, metadata=metadata)


def test_descriptive_measurement_controls_and_legacy_flag_rejection():
    args = _parser().parse_args(['fingerprint', 'fixture/model', '--measurement_mode', 'row_blocks'])
    assert args.tier == 'row_blocks'
    args = _parser().parse_args(['batch', '--models', 'models.csv', '--manifests', 'manifests',
                                '--out', 'out', '--measurement_mode', 'full_weights', '--also_row_block_sizes', '128,256'])
    assert args.tier == 'full_weights' and args.also_row_block_sizes == '128,256'
    for arguments in [['--tier', 't1'], ['--measurement_mode', 't1']]:
        with pytest.raises(SystemExit):
            _parser().parse_args(['fingerprint', 'fixture/model', *arguments])


def test_new_artifact_names_and_historical_full_weight_read(local_manifest, tmp_path):
    source = fingerprint_model(local_manifest, HubReader(), 'full_weights', tmp_path,
                               also_row_block_sizes=(128, 256))
    assert source.name.endswith('.full_weights.fingerprint.safetensors')
    assert list(tmp_path.glob('*.row_blocks.fingerprint.safetensors'))
    assert list(tmp_path.glob('*.row_blocks_128.fingerprint.safetensors'))
    expected = load_fingerprint(source)
    assert expected.meta['tier'] == 'full_weights'
    historical = tmp_path / 'historical.t2.fingerprint.safetensors'
    _historical_fingerprint(source, historical, 't2')
    with h5py.File(tmp_path / 'historical.t2.spectra.h5', 'w') as output:
        output.create_dataset('layers/longname', data=np.asarray(expected.names, dtype=h5py.string_dtype()))
        eigs = output.create_dataset('eigs', (len(expected.names),), dtype=h5py.vlen_dtype(np.dtype('float32')))
        for index, spectrum in enumerate(expected.spectra):
            eigs[index] = spectrum
    observed = load_fingerprint(historical)
    assert observed.meta['tier'] == 'full_weights'
    assert observed.names == expected.names
    np.testing.assert_array_equal(observed.sketch, expected.sketch)
    for left, right in zip(observed.spectra, expected.spectra):
        np.testing.assert_array_equal(left, right)


def test_historical_row_block_metadata_is_normalized(tmp_path):
    import torch
    source = fingerprint_tensors([('weight', torch.ones(16, 8))], tmp_path / 'new.safetensors',
                                 dict(repo_id='fixture/model', revision='a' * 40))
    historical = tmp_path / 'historical.t1.fingerprint.safetensors'
    _historical_fingerprint(source, historical, 't1')
    observed = load_fingerprint(historical)
    assert observed.meta['tier'] == 'row_blocks'
    assert observed.meta['evidence_version'] == '2'


@pytest.mark.parametrize('command', ['diff', 'check'])
def test_historical_fingerprint_finds_its_declared_parent_manifest(tmp_path, capsys, command):
    import json
    import torch
    from evidence.cli import main
    parent = fingerprint_tensors([('weight', torch.ones(16, 8))], tmp_path / 'parent.safetensors',
                                 dict(repo_id='fixture/parent', revision='a' * 40))
    historical = tmp_path / 'historical.t1.fingerprint.safetensors'
    _historical_fingerprint(parent, historical, 't1')
    (tmp_path / 'historical.manifest.json').write_text(json.dumps(dict(declared_bases=[str(parent)])))
    assert main([command, str(historical), '--json']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['relatedness']['verdict'] == 'uncalibrated'
