"""Per-layer fit metrics, with eigenvalues retained in their HDF5 files."""
import json

import h5py
import pandas as pd
from safetensors import safe_open

from evidence.compatibility import is_baseline_fingerprint

from .schema import LAYERS_COLUMNS, empty_table


def build_layers(packages):
    records = []
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
            for index in range(count):
                record = {name: values[index] for name, values in columns.items()}
                module, attribute = record.get("module_name", ""), record.get("weight_attribute", "")
                stored_key = ".".join(part for part in (module, attribute) if part)
                candidate = stored_key + (record.get("slice") or "")
                record.update(model_key=package["model_key"], h5_index=index,
                              view_name=candidate if candidate in names else "")
                records.append(record)
    if not records:
        return empty_table(LAYERS_COLUMNS)
    return pd.DataFrame(records).reindex(columns=LAYERS_COLUMNS).sort_values(["model_key", "h5_index"], kind="stable").reset_index(drop=True)
