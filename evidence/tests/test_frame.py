import numpy as np
import pandas as pd

from evidence.frame import build_frame, draw_stratified, load_hub_stats


def test_build_frame_covers_modalities_and_metadata_only(hub_stats_sample):
    frame = build_frame(hub_stats_sample)
    assert {"text", "vision-language", "vision", "audio", "image-generation", "video", "embedding"} <= set(frame.modality)
    assert frame.system_one.any() and frame.loc[frame.system_one, "stratum"].eq("decision").all()
    assert frame.loc[frame.weight_format.isin(["gguf", "bin"]), "metadata_only"].all()
    assert frame.loc[frame.params == 7.6e9, "size_stratum"].eq("3-9B").all()

def test_draw_records_inclusion_probability(hub_stats_sample):
    frame = build_frame(hub_stats_sample)
    s = draw_stratified(frame, {"text:<1B": 2}, seed=20261028)
    assert len(s) == 2 and np.allclose(s.inclusion_prob, 2 / frame.stratum.value_counts()["text:<1B"])
    assert s.equals(draw_stratified(frame, {"text:<1B": 2}, seed=20261028))

def test_draw_caps_at_stratum_size(hub_stats_sample):
    s = draw_stratified(build_frame(hub_stats_sample), {"audio": 10_000}, seed=1)
    assert (s.inclusion_prob == 1.0).all() and set(s.modality) == {"audio"}


def test_no_weight_exclusions_and_additional_modalities_are_explicit(hub_stats_sample):
    frame = build_frame(hub_stats_sample)
    assert len(frame) == 39
    assert frame.attrs['excluded'] == [{'repo_id': 'fixture/no-weights', 'reason': 'no_weights'}]
    assert {'3d', 'robotics'} <= set(frame.modality)
    adapter = frame.set_index('repo_id').loc['fixture/diffusion-1']
    assert adapter.is_adapter and adapter.modality == 'image-generation'
    assert frame.loc[frame.repo_id == 'fixture/embedding-1', 'metadata_only'].all()


def test_parquet_loader_projects_columns(tmp_path, hub_stats_sample):
    path = tmp_path / 'models.parquet'
    sample = hub_stats_sample.assign(unneeded='ignore')
    sample.to_parquet(path, index=False)
    loaded = load_hub_stats(path)
    assert list(loaded.columns) == list(hub_stats_sample.columns)
    assert loaded.iloc[0].safetensors['total'] == 135000000


def test_null_dtype_counts_keep_parameter_size_unknown(hub_stats_sample):
    row = hub_stats_sample.iloc[[11]].copy()
    row.at[row.index[0], 'safetensors'] = {'parameters': {'BF16': None}, 'total': None}
    frame = build_frame(row)
    assert pd.isna(frame.iloc[0].params)
    assert frame.iloc[0].size_stratum == 'unknown'
