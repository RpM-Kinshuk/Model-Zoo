"""Read historical tables and packages without rewriting their stored contracts."""
import hashlib
import json
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
from safetensors.numpy import save_file

from evidence.compatibility import historical_artifact_name
from zoo.build import build_zoo, open_tables
from zoo.publish import package_inputs, publish_package


def test_historical_table_columns_are_normalized_without_rewriting(built_zoo):
    path = built_zoo / 'models.parquet'
    table = pq.read_table(path)
    legacy = {'row_block_fingerprint': 'fingerprint_t1',
              'full_weight_fingerprint': 'fingerprint_t2', 'full_weight_spectra': 'spectra_t2'}
    pq.write_table(table.rename_columns([legacy.get(name, name) for name in table.column_names]), path)
    view_path = built_zoo / 'views.parquet'
    views = pq.read_table(view_path)
    column = views.column_names.index('tier')
    tiers = [{'row_blocks': 't1', 'full_weights': 't2'}.get(value, value) for value in views['tier'].to_pylist()]
    pq.write_table(views.set_column(column, 'tier', pa.array(tiers)), view_path)
    before = [item.read_bytes() for item in (path, view_path)]
    output = open_tables(built_zoo)
    assert 'row_block_fingerprint' in output['models'] and 'fingerprint_t1' not in output['models']
    assert output['models'].row_block_fingerprint.str.len().max() > 0
    assert set(output['views'].tier) <= {'row_blocks', 'full_weights'}
    assert [item.read_bytes() for item in (path, view_path)] == before


def test_historical_run_can_publish_and_build_selected_package(package_run):
    run = package_run
    # Convert only this private fixture to historical labels/filenames. Production
    # migration never renames historical inputs or edits their fingerprint bytes.
    for path in list(run.folder.glob('*')):
        if path.name.endswith('.fingerprint.safetensors'):
            with safe_open(path, framework='numpy') as handle:
                metadata = dict(handle.metadata())
                arrays = {key: handle.get_tensor(key) for key in handle.keys()}
            mode = metadata['tier']
            metadata['tier'] = {'row_blocks': 't1', 'full_weights': 't2',
                                'row_blocks_128': 't1b128', 'row_blocks_512': 't1b512'}[mode]
            metadata['evidence_version'] = '2'
            destination = path.with_name(historical_artifact_name(path.name))
            save_file(arrays, destination, metadata=metadata)
            path.unlink()
        elif path.name.endswith('.full_weights.spectra.h5'):
            path.rename(path.with_name(historical_artifact_name(path.name)))
    manifest_path = run.folder / (run.safe + '.manifest.json')
    manifest = json.loads(manifest_path.read_text())
    manifest['evidence_version'] = '2'
    manifest_path.write_text(json.dumps(manifest))
    original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run.folder.iterdir() if path.is_file()}
    package = publish_package(run.run, run.manifest['repo_id'], run.packages, run.root)
    assert package is not None
    metadata = json.loads((package / 'package.json').read_text())
    selected, skipped = package_inputs(run.packages, metadata['conventions_id'])
    assert len(selected) == 1 and skipped == 0
    assert selected[0][1]['kind'] == 'full'
    rebuilt = build_zoo(None, None, run.root / 'rebuilt', data_root=run.root,
                        packages=run.packages, conventions_id=metadata['conventions_id'])
    tables = open_tables(rebuilt)
    assert tables['models'].n_views.iloc[0] > 0
    assert not tables['views'].empty
    assert tables['models'].row_block_fingerprint.iloc[0]
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run.folder.iterdir() if path.is_file()} == original
