"""Convert measurement outputs (typed Parameters or dicts) to and from dicts of arrays."""
from dataclasses import fields

import numpy as np
import pandas as pd

from ..parameters import Parameters


def as_arrays(d) -> dict[str, np.ndarray]:
    """``{name: np.ndarray}`` from a Parameters object, dict or DataFrame."""
    if isinstance(d, Parameters):
        b = d.broadcast()
        return {f.name: np.asarray(getattr(b, f.name)) for f in fields(b)}
    if isinstance(d, pd.DataFrame):
        return {k: d[k].to_numpy() for k in d.columns}
    return {k: np.asarray(v) for k, v in dict(d).items()}


def from_arrays(arrays: dict, data_type):
    """The measurement's data type (or a plain dict) from ``{name: array}``."""
    return data_type.from_dict(arrays) if data_type is not None else dict(arrays)


def n_rows(arrays: dict) -> int:
    return len(next(iter(arrays.values())))
