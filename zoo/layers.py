"""Per-layer fit metrics, with eigenvalues retained in their HDF5 files."""
import json

import h5py
import pandas as pd
from safetensors import safe_open

from evidence.compatibility import is_baseline_fingerprint

from .schema import LAYERS_COLUMNS, empty_table


_TEXT_COLUMNS = {"longname", "module_name", "weight_attribute", "slice", "fit_status"}


def build_layers(packages):
    """One row per runner layer record in each full package's `esd/metrics.h5`.

    `h5_index` addresses that file's aligned `/layers` and `/eigs` records;
    `view_name` links to the fingerprint view table when the identity matches.
    """
    tables = []
    for folder, package in packages:
        if package["kind"] != "full":
            continue
        names = set()
        for name in package["files"]:
            if is_baseline_fingerprint(name):
                with safe_open(folder / name, framework="numpy") as h:
                    names.update(json.loads(h.metadata()["names"]))
        with h5py.File(folder / "esd/metrics.h5") as h:
            group = h["layers"]
            count = len(group["longname"])
            if any(dataset.shape != (count,) for dataset in group.values()):
                raise ValueError("Unaligned package layer records")
            columns = {name: dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
                       for name, dataset in group.items() if name in LAYERS_COLUMNS}
        table = pd.DataFrame(columns)
        for name in _TEXT_COLUMNS & set(table.columns):
            # Missing text is null in the database, never an empty string.
            table[name] = [value if value else None for value in table[name]]
        modules = table.get("module_name", pd.Series([None] * count))
        attributes = table.get("weight_attribute", pd.Series([None] * count))
        slices = table.get("slice", pd.Series([None] * count))
        candidates = [".".join(part for part in (module, attribute) if part) + (piece or "")
                      for module, attribute, piece in zip(modules, attributes, slices)]
        table["view_name"] = [candidate if candidate in names else None for candidate in candidates]
        table["model_key"] = package["model_key"]
        table["h5_index"] = range(count)
        tables.append(table)
    if not tables:
        return empty_table(LAYERS_COLUMNS)
    return (pd.concat(tables, ignore_index=True).reindex(columns=LAYERS_COLUMNS)
            .sort_values(["model_key", "h5_index"], kind="stable").reset_index(drop=True))
