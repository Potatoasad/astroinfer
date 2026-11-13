"""Small, dependency-light helpers for saving catalogs and representations.

    save_arrays / load_arrays   {group: {name: array}} <-> one HDF5 file
    write_json / read_json      JSON with numpy scalars/arrays converted
    json_clean                  make a value JSON-safe (or raise TypeError)
    type_path / resolve_type    "module:QualName" <-> class
"""
import importlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

_STRING_ATTR = "astroinfer_string"


# ============================================================
# JSON
# ============================================================

def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray) or hasattr(obj, "__array__"):
        return np.asarray(obj).tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON-serializable")


def json_clean(value):
    """``value`` round-tripped through JSON (tuples -> lists, numpy -> builtins)."""
    return json.loads(json.dumps(value, default=_json_default))


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default))


def read_json(path):
    return json.loads(Path(path).read_text())


# ============================================================
# HDF5: flat groups of arrays
# ============================================================

def save_arrays(path, groups: Mapping[str, Mapping[str, Any]]) -> None:
    """Write ``{group: {name: array}}`` to an HDF5 file (string arrays allowed)."""
    import h5py

    with h5py.File(path, "w") as f:
        for group, values in groups.items():
            g = f.create_group(group)
            for name, value in values.items():
                arr = np.asarray(value)
                if arr.dtype.kind in "OUS":
                    ds = g.create_dataset(
                        name,
                        data=np.asarray(arr, dtype=object).astype(str).astype(object),
                        dtype=h5py.string_dtype(),
                    )
                    ds.attrs[_STRING_ATTR] = True
                else:
                    g.create_dataset(name, data=arr)


def load_arrays(path) -> dict[str, dict[str, np.ndarray]]:
    import h5py

    out: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as f:
        for group, g in f.items():
            out[group] = {}
            for name, ds in g.items():
                if ds.attrs.get(_STRING_ATTR, False):
                    out[group][name] = ds.asstr()[()].astype(str)
                else:
                    out[group][name] = ds[()]
    return out


# ============================================================
# Types by import path
# ============================================================

def type_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def resolve_type(path: str, types: Iterable[type] = ()) -> type:
    """The class at ``"module:QualName"``.

    Classes passed in ``types`` are used instead of importing. A class
    matches by full path first, then by qualified name. Use this for classes
    defined in a notebook cell, since that module cannot be re-imported.
    """
    types = list(types)
    for cls in types:
        if type_path(cls) == path:
            return cls
    module_name, _, qualname = path.partition(":")
    for cls in types:
        if cls.__qualname__ == qualname:
            return cls
    try:
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError) as exc:
        raise LookupError(
            f"Cannot import {path!r}. If the class was defined in a notebook "
            f"or script, pass it on load, e.g. types=[{qualname.split('.')[-1]}]"
        ) from exc


# ============================================================
# Directories
# ============================================================

def replace_directory(target: Path, write) -> None:
    """Call ``write(tmp_dir)`` and move the result to ``target``.

    A failed write leaves any previous ``target`` untouched.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write(tmp)
        if target.exists():
            old = target.with_name(f".{target.name}.old")
            if old.exists():
                shutil.rmtree(old)
            os.replace(target, old)
            os.replace(tmp, target)
            shutil.rmtree(old)
        else:
            os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
