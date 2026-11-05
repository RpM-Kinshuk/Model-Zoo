import h5py
import pandas as pd
from zoo.build import build_zoo, open_tables
from zoo.publish import publish_package, current_conventions_id
from zoo.schema import LAYERS_COLUMNS


def test_layers_rows_match_metrics_h5(package_run):
    r = package_run
    publish_package(r.run, "lab/tiny", r.packages, r.root)
    out = build_zoo(packages=r.packages, out=r.root / "builds/current", data_root=r.root)
    layers = open_tables(out)["layers"]
    assert list(layers.columns) == LAYERS_COLUMNS and len(layers) == 2
    assert layers["view_name"].tolist() == ["0.weight", "1.weight"]
    with h5py.File(r.run / "metrics/lab--tiny.h5") as h:
        assert layers.longname.tolist() == h["layers/longname"].asstr()[:].tolist()
        assert layers.h5_index.tolist() == [0, 1]
        assert layers.alpha.tolist() == h["layers/alpha"][:].tolist()


def test_build_from_packages_matches_build_from_directories(package_run):
    r = package_run
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    direct = open_tables(build_zoo(package, [package], r.root / "direct", r.root))
    packaged = open_tables(build_zoo(packages=r.packages, out=r.root / "packaged", data_root=r.root))
    for name in ("models", "views", "coverage", "lineage", "scores", "predictions"):
        pd.testing.assert_frame_equal(direct[name], packaged[name])


def test_layer_indices_resolve_through_model_spectrum_link(package_run):
    r = package_run
    package = publish_package(r.run, "lab/tiny", r.packages, r.root)
    tables = open_tables(build_zoo(packages=r.packages, out=r.root / "indexed", data_root=r.root))
    path = r.root / tables["models"].iloc[0].full_weight_spectra
    assert path == package / "esd/metrics.h5"
    with h5py.File(path) as h:
        for row in tables["layers"].itertuples():
            assert h["layers/longname"].asstr()[row.h5_index] == row.longname
