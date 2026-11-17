"""Collection receipts describe retained evidence without archiving checkpoints."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from zoo.collection import write_collection_receipt


def test_collection_receipt_lists_required_files_and_honest_capabilities(tmp_path):
    root=tmp_path;stage=root/'generation';stage.mkdir()
    fp=root/'measurements'/'tiny.row_blocks.fingerprint.safetensors';fp.parent.mkdir();fp.write_bytes(b'fingerprint')
    spectra=fp.with_name('tiny.full_weights.spectra.h5');spectra.write_bytes(b'spectra')
    manifest=fp.with_name('tiny.manifest.json');manifest.write_text(json.dumps({'repo_id':'org/tiny','revision':'a'*40}))
    metadata=fp.with_name('tiny.source-metadata.json.gz');metadata.write_bytes(b'compressed-metadata')
    tables={'models':pd.DataFrame([dict(model_key='org/tiny@'+'a'*40,row_block_fingerprint=str(fp.relative_to(root)),full_weight_fingerprint='',full_weight_spectra=str(spectra.relative_to(root)),coverage_status='partial')]),
            'coverage':pd.DataFrame([dict(model_key='org/tiny@'+'a'*40,status='skipped',reason='expert_not_sampled')]),
            'scores':pd.DataFrame([dict(model_key='org/tiny@'+'a'*40)])}
    (stage/'score-coverage.json').write_text(json.dumps({'records':[dict(model_key='org/tiny@'+'a'*40,source_id='model_card',status='available',score_rows=1)]}))
    p=write_collection_receipt(stage,tables,root,manifest_paths=[manifest],provenance=[dict(kind='file',path=str(fp.relative_to(root)),sha256=hashlib.sha256(fp.read_bytes()).hexdigest())])
    d=json.loads(p.read_text());assert d['schema_version']=='1'
    assert d['reference_closure_complete']
    refs={r['path']:r for r in d['required_data_root_files']}
    assert set(refs)=={str(p.relative_to(root)) for p in [fp,spectra,manifest,metadata]}
    assert refs[str(fp.relative_to(root))]['sha256']==hashlib.sha256(fp.read_bytes()).hexdigest()
    assert not d['reference_hash_binding_complete']
    assert len(d['references_without_hashes'])==3
    m=d['models'][0];assert m['measurement_tier']=='full_spectra_measured_views'
    assert m['sampled_experts'] and m['score_rows']==1
    assert m['raw_metadata']=='retained'
    assert 'raw_tensor_methods' in d['requires_additional_inputs']
    assert not d['raw_checkpoint_archival']
    before={p:p.read_bytes() for p in [fp,spectra,manifest,metadata]}
    write_collection_receipt(stage,tables,root,manifest_paths=[manifest])
    assert all(p.read_bytes()==v for p,v in before.items())
    assert sorted(p.name for p in stage.iterdir())==['collection.json','score-coverage.json']


def test_missing_measurements_and_legacy_metadata_are_visible(tmp_path):
    stage=tmp_path/'g';stage.mkdir()
    tables={'models':pd.DataFrame([dict(model_key='org/old@'+'b'*40,row_block_fingerprint='missing.fingerprint.safetensors',full_weight_fingerprint='',full_weight_spectra='',coverage_status='not_measured')]),'scores':pd.DataFrame(),'coverage':pd.DataFrame()}
    p=write_collection_receipt(stage,tables,tmp_path)
    d=json.loads(p.read_text());assert not d['reference_closure_complete']
    assert d['required_data_root_files'][0]['status']=='missing'
    assert d['models'][0]['raw_metadata']=='not_captured'
    assert d['models'][0]['manifest']=='not_supplied'
    assert d['models'][0]['score_coverage']=='not_attempted'


def test_receipt_does_not_pull_raw_weights_or_outside_paths_into_closure(tmp_path):
    stage=tmp_path/'g';stage.mkdir();raw=tmp_path/'model.safetensors';raw.write_bytes(b'neverarchive')
    tables={'models':pd.DataFrame(),'coverage':pd.DataFrame(),'scores':pd.DataFrame()}
    p=write_collection_receipt(stage,tables,tmp_path,provenance=[dict(kind='file',path='model.safetensors',sha256='a'*64),dict(kind='file',path='../elsewhere.bin',sha256='b'*64)])
    d=json.loads(p.read_text());assert d['required_data_root_files']==[]
    assert raw.read_bytes()==b'neverarchive'


def test_model_without_manifest_cannot_claim_complete_closure(tmp_path):
    stage=tmp_path/'g';stage.mkdir()
    tables={'models':pd.DataFrame([{'model_key':'org/m@'+'a'*40}])}
    d=json.loads(write_collection_receipt(stage,tables,tmp_path).read_text())
    assert not d['reference_closure_complete']
    assert d['models'][0]['manifest']=='not_supplied'
