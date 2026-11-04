"""Small real files exercise the pair pass without Hub downloads or GPUs."""
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from evidence.block_cache import BlockCache
from evidence.fingerprint import fingerprint_model, load_fingerprint
from evidence.hub import HubReader
from measurement_config import safe_filename


def test_pair_delta_matches_in_run_delta_and_diff(local_manifest, parent_manifest, tmp_path):
    from evidence.pair_delta import pair_deltas, load_pair_delta
    from evidence.compare import diff_fingerprints
    child, parent = local_manifest, parent_manifest
    package = tmp_path / "package"
    a = load_fingerprint(fingerprint_model(child, HubReader(), "t1", package))
    b = load_fingerprint(fingerprint_model(parent, HubReader(), "t1", tmp_path / "parent"))
    reference = load_fingerprint(fingerprint_model(child, HubReader(), "t1", tmp_path / "reference", parent=parent))
    path = pair_deltas(child, parent, HubReader(), package)
    meta, tensors = load_pair_delta(path)
    assert meta["child"] == {key: child[key] for key in ("repo_id", "revision")}
    np.testing.assert_array_equal(tensors["rows"], reference.rows)
    for j, key in enumerate(("frac_identical", "rel_update_norm", "rank")):
        np.testing.assert_allclose(tensors["delta_scalars"][:, j], reference.delta[key], equal_nan=True)
    np.testing.assert_allclose(tensors["delta_top_sv"], reference.delta["top_sv"], equal_nan=True)
    assert meta["delta_status"] == list(reference.delta["status"])
    report = diff_fingerprints(a, b)
    for i, name in enumerate(a.names):
        if name in b.names:
            assert report["views"][name]["delta_status"] == reference.delta["status"][i]
    assert pair_deltas(child, parent, HubReader(), package) == path


def test_pair_delta_row_reads_and_sibling_parent_cache(local_manifest, parent_manifest, tmp_path):
    from evidence.pair_delta import pair_deltas
    class Reader(HubReader):
        reads = []
        def read(self, remote, start, end):
            self.reads.append(end-start)
            return super().read(remote, start, end)
    reader, cache = Reader(), BlockCache(tmp_path / "cache")
    pair_deltas(local_manifest, parent_manifest, reader, tmp_path / "a", block_cache=cache)
    first = len(reader.reads)
    reader.reads.clear()
    pair_deltas(local_manifest, parent_manifest, reader, tmp_path / "b", block_cache=cache)
    assert len(reader.reads) * 2 == first
    # The local fixture has 256-row blocks with <=64 columns, stored F32/BF16.
    assert max(reader.reads) <= 256 * 64 * 4


def test_pair_delta_incompatible_shapes_are_gaps(local_manifest, parent_manifest, tmp_path):
    from evidence.pair_delta import pair_deltas, load_pair_delta
    import copy
    other = copy.deepcopy(parent_manifest)
    other["tensors"][0]["shape"][1] //= 2
    other["tensors"][0]["end"] = other["tensors"][0]["start"] + np.prod(other["tensors"][0]["shape"]) * 4
    _, tensors = load_pair_delta(pair_deltas(local_manifest, other, HubReader(), tmp_path))
    assert np.isnan(tensors["delta_scalars"][:, :2]).any()


def test_pair_delta_rejects_unknown_format(tmp_path):
    from evidence.pair_delta import load_pair_delta
    from safetensors.numpy import save_file
    path = tmp_path / "bad.safetensors"
    save_file({}, path, metadata={"pair_delta_version": "999"})
    with pytest.raises(ValueError):
        load_pair_delta(path)
