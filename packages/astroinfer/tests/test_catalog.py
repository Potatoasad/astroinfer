import json
import warnings
from dataclasses import dataclass, replace
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from astroinfer import Catalog, PosteriorData, RawSamples, parameter_dataclass
from astroinfer.representations import (
    BaseFit,
    BaseFitter,
    available_kinds,
    config_hash,
    get_fitter_class,
    register,
)


@parameter_dataclass
class Params:
    x: float
    y: float


@parameter_dataclass
class Data:
    x_obs: float


def make_posterior(seed=0, n=500, observed=True, float64=False):
    rng = np.random.default_rng(seed)
    dtype = np.float64 if float64 else np.float32
    x = rng.normal(seed, 1.0, n).astype(dtype)
    y = rng.uniform(0, 1, n).astype(dtype)
    lpx = np.full(n, -np.log(10.0))
    lpy = np.zeros(n)
    return PosteriorData(
        theta=Params(x=x, y=y),
        sample_stats={"log_prior[x]": lpx, "log_prior[y]": lpy, "prior": np.exp(lpx + lpy)},
        extras={"z": x + y, "label": np.array(["a"] * n)},
        metadata={"event_name": f"ev{seed}", "snr": np.float64(seed + 8.0)},
        observed_data=Data(x_obs=float(seed)) if observed else None,
    )


# ------------------------------------------------------------ a toy representation

class MeanFit(BaseFit):
    def __init__(self, mean, fit_key):
        self.mean = np.asarray(mean)
        self._key = fit_key

    @property
    def fit_key(self):
        return self._key

    def dataproduct(self, *, N, seed, scale=1.0):
        return {"mean": self.mean * scale, "draws": np.random.default_rng(seed).normal(size=N)}

    def save(self, directory):
        np.save(directory / "mean.npy", self.mean)
        (directory / "key.txt").write_text(self._key)

    @classmethod
    def load(cls, directory, ctx):
        return cls(np.load(directory / "mean.npy"), (directory / "key.txt").read_text())


@register
@dataclass
class MeanFitter(BaseFitter):
    kind: ClassVar[str] = "mean"
    fit_type: ClassVar[type] = MeanFit
    shift: float = 0.0
    scale: float = 1.0  # data product only

    def fit_key(self, eventname, source_key):
        return config_hash({"shift": self.shift, "source": source_key})

    def fit(self, posterior, *, eventname, source_key):
        mean = np.asarray(posterior["x"]).mean() + self.shift
        return MeanFit(mean, self.fit_key(eventname, source_key))

    def dataproduct_options(self, eventname):
        return {"scale": self.scale}

    def to_dict(self):
        return {"shift": self.shift, "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


@pytest.fixture
def catalog():
    return Catalog({f"ev{i}": make_posterior(i) for i in range(3)},
                   raw=RawSamples(columns=["x", "y"], N=100))


# ------------------------------------------------------------ raw

def test_raw_dataproduct_shapes_and_marginal_prior(catalog):
    dp = catalog.dataproduct()
    assert set(dp) == {"x", "y", "prior"}
    assert dp["x"].shape == (3, 100)
    catalog.set_raw(RawSamples(columns=["x"], N=50))
    dp = catalog.dataproduct()
    np.testing.assert_allclose(dp["prior"], 0.1, rtol=1e-6)
    assert dp["x"].shape == (3, 50)


def test_raw_rejects_missing_columns():
    with pytest.raises(KeyError, match="missing columns"):
        Catalog({"a": make_posterior()}, raw=["x", "nope"])


def test_seed_depends_on_event_name_not_position(catalog):
    full = catalog.dataproduct()
    sub = catalog.dataproduct(eventnames=["ev2", "ev0"])
    np.testing.assert_array_equal(sub["x"][0], full["x"][2])
    np.testing.assert_array_equal(sub["x"][1], full["x"][0])


def test_content_hash_ignores_float_width():
    a = make_posterior(1, float64=False)
    b = replace(a, theta=Params(x=np.asarray(a.theta.x, np.float64), y=np.asarray(a.theta.y, np.float64)))
    assert a.content_hash() == b.content_hash()
    c = replace(a, theta=Params(x=np.asarray(a.theta.x) + 1, y=a.theta.y))
    assert a.content_hash() != c.content_hash()


# ------------------------------------------------------------ fitting and staleness

def test_fit_staleness_and_refit_only_stale(catalog):
    catalog.register_representation("mean", MeanFitter(shift=0.0))
    assert catalog.fit("mean") == ["ev0", "ev1", "ev2"]
    assert catalog.fit("mean") == []

    # data-product-only change: not stale
    catalog.fitter("mean").scale = 2.0
    assert catalog.stale("mean") == []
    dp = catalog.dataproduct(representation="mean", N=7)
    assert dp["draws"].shape == (3, 7)
    np.testing.assert_allclose(dp["mean"][1], 2 * catalog.get_fit("ev1", "mean").mean)

    # replacing a posterior makes just that event stale
    catalog.add_event("ev1", make_posterior(10), overwrite=True)
    assert catalog.stale("mean") == ["ev1"]
    with pytest.raises(ValueError, match="stale"):
        catalog.dataproduct(representation="mean")
    catalog.dataproduct(representation="mean", allow_stale=True)
    assert catalog.fit("mean") == ["ev1"]

    # fit-relevant config change makes everything stale
    catalog.fitter("mean").shift = 1.0
    assert catalog.stale("mean") == ["ev0", "ev1", "ev2"]


def test_add_representation_type_checked(catalog):
    catalog.register_representation("mean", MeanFitter())
    with pytest.raises(TypeError):
        catalog.add_representation("ev0", "mean", object())
    catalog.add_representation("ev0", "mean", MeanFit(1.0, "k"))
    assert catalog.is_stale("ev0", "mean")
    assert catalog.missing("mean") == ["ev1", "ev2"]


def test_fit_errors_warn_skips_event(catalog):
    class Boom(MeanFitter):
        def fit(self, posterior, *, eventname, source_key):
            if eventname == "ev1":
                raise RuntimeError("bad fit")
            return super().fit(posterior, eventname=eventname, source_key=source_key)

    catalog.register_representation("mean", Boom())
    with pytest.warns(UserWarning, match="bad fit"):
        assert catalog.fit("mean", errors="warn") == ["ev0", "ev2"]


# ------------------------------------------------------------ saving

def test_save_load_round_trip(tmp_path, catalog):
    catalog.register_representation("mean", MeanFitter(shift=0.5, scale=3.0))
    catalog.fit("mean")
    catalog.fitter("mean").shift = 0.7          # ev* fits are now stale
    catalog.fit("mean", ["ev0"])                # ev0 fresh again
    root = catalog.save(tmp_path / "cat")

    loaded = Catalog.load(root)
    assert loaded.eventnames == catalog.eventnames
    assert loaded.representations == ["raw", "mean"]
    assert loaded.fitter("mean") == catalog.fitter("mean")
    assert loaded.stale("mean") == ["ev1", "ev2"]
    for e in catalog.eventnames:
        a, b = catalog.posterior(e), loaded.posterior(e)
        np.testing.assert_array_equal(np.asarray(a.theta.x), b.theta.x)
        np.testing.assert_array_equal(np.asarray(a.extras["label"]), b.extras["label"])
        assert b.metadata == {"event_name": e, "snr": float(a.metadata["snr"])}
        assert b.observed_data == a.observed_data
        assert loaded.source_key(e) == catalog.source_key(e)
    for key in ("x", "y", "prior"):
        np.testing.assert_allclose(loaded.dataproduct()[key], catalog.dataproduct()[key])


def test_incremental_save_and_pruning(tmp_path, catalog):
    catalog.register_representation("mean", MeanFitter())
    catalog.fit("mean")
    root = catalog.save(tmp_path / "cat")
    ev0 = root / "representations/mean/events/ev0/mean.npy"
    ev1 = root / "representations/mean/events/ev1/mean.npy"
    before0, before1 = ev0.stat().st_mtime_ns, ev1.stat().st_mtime_ns

    catalog.fitter("mean").shift = 1.0
    catalog.fit("mean", ["ev1"])
    catalog.remove_event("ev2")
    catalog.save(root)

    assert ev0.stat().st_mtime_ns == before0
    assert ev1.stat().st_mtime_ns != before1
    assert not (root / "representations/mean/events/ev2").exists()
    assert not (root / "representations/raw/events/ev2").exists()
    assert json.loads((root / "catalog.json").read_text())["eventnames"] == ["ev0", "ev1"]

    catalog.remove_representation("mean")
    catalog.save(root)
    assert not (root / "representations/mean").exists()


def test_load_with_types_override(tmp_path, catalog):
    root = catalog.save(tmp_path / "cat")
    info_path = root / "representations/raw/events/ev0/posterior.json"
    info = json.loads(info_path.read_text())
    info["parameter_type"] = "__main__:Params"     # e.g. a class defined in a notebook
    info_path.write_text(json.dumps(info))
    with pytest.raises(LookupError, match="types="):
        Catalog.load(root)
    loaded = Catalog.load(root, types=[Params, Data])
    assert isinstance(loaded.posterior("ev0").theta, Params)


def test_save_refuses_foreign_directory(tmp_path, catalog):
    (tmp_path / "stuff.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        catalog.save(tmp_path)


def test_unserializable_metadata_warns(tmp_path):
    p = make_posterior(observed=False)
    p = replace(p, metadata={"ok": 1, "bad": object()})
    cat = Catalog({"a": p}, raw=["x"])
    with pytest.warns(UserWarning, match="bad"):
        root = cat.save(tmp_path / "cat")
    assert Catalog.load(root).posterior("a").metadata == {"ok": 1}


# ------------------------------------------------------------ registry and display

def test_registry():
    assert get_fitter_class("raw") is RawSamples
    assert get_fitter_class("mean") is MeanFitter
    assert {"raw", "mean"} <= set(available_kinds())
    with pytest.raises(LookupError, match="No representation kind"):
        get_fitter_class("definitely-not-a-kind")


def test_display(catalog):
    catalog.register_representation("mean", MeanFitter())
    catalog.fit("mean", ["ev0"])
    catalog.fitter("mean").shift = 3.0
    catalog.fit("mean", ["ev1"])
    html = catalog._repr_html_()
    assert "Representations" in html and "Events" in html and "MeanFitter" in html
    assert html.count(">missing</span>") == 1 and html.count(">stale</span>") == 1
    for obj in (catalog.raw, catalog.get_fit("ev0", "raw"), catalog.get_fit("ev0", "mean")):
        assert obj._repr_html_().startswith("<details")
