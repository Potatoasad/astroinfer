"""Parameter dataclasses with nested parameter-dataclass fields."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro.distributions import constraints

from astroinfer import (
    Catalog,
    LatentSpace,
    PosteriorData,
    RawSamples,
    batched_method,
    coerce,
    parameter,
    parameter_dataclass,
)
from astroinfer.detection._data import as_arrays, from_arrays


@parameter_dataclass(latex="m")
class Masses:
    m1: float = parameter(support=constraints.positive, latex="m_1")
    q: float = parameter(support=constraints.interval(0.0, 1.0))


@parameter_dataclass(latex=r"\chi")
class Spins:
    chi: float = parameter(index=("component",), support=constraints.unit_interval)


@parameter_dataclass(latex=r"\theta")
class BBH:
    masses: Masses
    spins: Spins = parameter(name="Spin magnitudes")
    z: float = parameter(support=constraints.positive)

    @batched_method
    def total(self):
        return self.masses.m1 * (1 + self.masses.q) + self.spins.chi.sum()


@parameter_dataclass
class Outer:
    """Forward reference: ``Inner`` is defined after ``Outer``."""
    inner: "Inner"
    w: float


@parameter_dataclass
class Inner:
    a: float


@parameter_dataclass(latex="b")
class Binary:
    masses: Masses
    z: float = parameter(support=constraints.positive)


FLAT = ["masses.m1", "masses.q", "spins.chi", "z"]


def make(batch=(), n_comp=2, seed=0):
    rng = np.random.default_rng(seed)
    return BBH(
        masses=Masses(
            m1=jnp.asarray(rng.uniform(5, 50, batch)),
            q=jnp.asarray(rng.uniform(0.1, 1, batch)),
        ),
        spins=Spins(chi=jnp.asarray(rng.uniform(0, 1, batch + (n_comp,)))),
        z=jnp.asarray(rng.uniform(0.1, 1, batch)),
    )


def assert_same(a, b):
    assert type(a) is type(b)
    for (pa, va), (pb, vb) in zip(a.to_flat_dict().items(), b.to_flat_dict().items()):
        assert pa == pb
        np.testing.assert_allclose(np.asarray(va), np.asarray(vb), rtol=1e-6)


# ------------------------------------------------------------ structure

def test_flat_helpers_and_type_metadata():
    assert BBH.nested_types() == {"masses": Masses, "spins": Spins}
    assert list(BBH.flat_field_specs()) == FLAT
    assert BBH.flat_field_specs()["spins.chi"].index == ("component",)
    assert BBH.parameters == ["masses", "spins", "z"]
    assert BBH.field_specs()["masses"].latex == "m"       # the child's symbol
    assert list(make().to_flat_dict()) == FLAT
    # Flat types: the flat helpers agree with the field helpers.
    assert Masses.nested_types() == {}
    assert Masses.flat_field_specs() == Masses.field_specs()


def test_tree_flatten_round_trip():
    b = make((3,))
    leaves, treedef = jax.tree_util.tree_flatten(b)
    assert len(leaves) == 4
    assert_same(jax.tree_util.tree_unflatten(treedef, leaves), b)
    doubled = jax.tree_util.tree_map(lambda v: 2 * v, b)
    np.testing.assert_allclose(doubled.masses.m1, 2 * b.masses.m1)


def test_forward_reference_resolves_lazily():
    assert Outer.nested_types() == {"inner": Inner}
    o = Outer.from_dict({"inner.a": np.array([1.0, 2.0]), "w": 3.0})
    assert o.batch_shape == (2,)
    assert o.to_dict() == {"inner": {"a": o.inner.a}, "w": 3.0}


def test_decoration_errors():
    with pytest.raises(TypeError, match="cannot set index or support"):
        @parameter_dataclass
        class BadIndex:
            masses: Masses = parameter(index=("x",))

    with pytest.raises(TypeError, match="cannot set index or support"):
        @parameter_dataclass
        class BadSupport:
            masses: Masses = parameter(support=constraints.positive)


# ------------------------------------------------------------ conversion

def test_to_dict_and_from_dict_forms():
    b = make((4,))
    d = b.to_dict()
    assert set(d) == {"masses", "spins", "z"}
    assert set(d["masses"]) == {"m1", "q"}

    assert_same(BBH.from_dict(d), b)                                  # nested dicts
    assert_same(BBH.from_dict({**d, "masses": b.masses}), b)          # child object
    assert_same(BBH.from_dict(b.to_flat_dict()), b)                   # dotted keys
    assert_same(coerce(d, BBH), b)

    with pytest.raises(KeyError, match=r"\['masses.q', 'z'\]"):
        BBH.from_dict({"masses.m1": 1.0, "spins.chi": [0.1, 0.2]})
    with pytest.raises(TypeError, match="masses: expected Masses"):
        BBH.from_dict({"masses": 1.0, "spins.chi": 0.0, "z": 1.0})


def test_from_dataframe_with_dotted_columns():
    b = make((5,), n_comp=1)
    df = pd.DataFrame({
        "masses.m1": np.asarray(b.masses.m1), "masses.q": np.asarray(b.masses.q),
        "spins.chi": np.asarray(b.spins.chi[:, 0]), "z": np.asarray(b.z),
        "extra": np.zeros(5),
    })
    out = coerce(df, BBH)
    np.testing.assert_allclose(out.masses.m1, b.masses.m1)
    assert out.batch_shape == (5,)


def test_detection_array_helpers():
    b = make((3,))
    arrays = as_arrays(b)
    assert list(arrays) == FLAT
    assert_same(from_arrays(arrays, BBH), b)


# ------------------------------------------------------------ batching

def test_broadcast_reshape_and_indexing():
    b = BBH(
        masses=Masses(m1=jnp.ones((2, 3)), q=0.5),
        spins=Spins(chi=jnp.zeros((3, 2))),
        z=1.0,
    )
    assert b.batch_shape == (2, 3)
    full = b.broadcast()
    assert full.masses.q.shape == (2, 3)
    assert full.spins.chi.shape == (2, 3, 2)
    assert b.broadcast((4, 2, 3)).z.shape == (4, 2, 3)

    r = b.reshape_batch((6,))
    assert r.batch_shape == (6,) and r.spins.chi.shape == (6, 2)

    assert b[0].batch_shape == (3,)
    assert b[:, 1:].spins.chi.shape == (2, 2, 2)
    assert b[1, np.array([True, False, True])].batch_shape == (2,)
    assert b[0, 0].batch_shape == ()

    assert b["masses.q"] == 0.5
    assert isinstance(b["masses"], Masses)
    for bad in ("masses.nope", "z.x", "nope"):
        with pytest.raises(KeyError):
            b[bad]


def test_named_axes_shared_across_children():
    @parameter_dataclass
    class A:
        h: float = parameter(index=("ifo",))

    @parameter_dataclass
    class B:
        g: float = parameter(index=("ifo",))

    @parameter_dataclass
    class AB:
        a: A
        b: B

    ok = AB(A(jnp.zeros((5, 3))), B(jnp.zeros(3)))
    assert ok._dimensions() == ((5,), {"ifo": 3})
    with pytest.raises(ValueError, match="'ifo' conflicts"):
        AB(A(jnp.zeros(3)), B(jnp.zeros(2))).batch_shape


# ------------------------------------------------------------ JAX transforms

def test_jit_vmap_grad():
    b = make((4,))
    expected = b.masses.m1 * (1 + b.masses.q) + b.spins.chi.sum(-1)

    np.testing.assert_allclose(jax.jit(lambda t: t.total())(b), expected, rtol=1e-6)
    np.testing.assert_allclose(jax.vmap(lambda t: t.total())(b), expected, rtol=1e-6)
    np.testing.assert_allclose(b.total(), expected, rtol=1e-6)
    assert_same(jax.jit(lambda t: t)(b), b)

    g = jax.grad(lambda t: t.total())(b[0])
    assert isinstance(g, BBH) and isinstance(g.masses, Masses)
    np.testing.assert_allclose(g.masses.m1, 1 + b.masses.q[0], rtol=1e-6)
    np.testing.assert_allclose(g.masses.q, b.masses.m1[0], rtol=1e-6)
    np.testing.assert_allclose(g.spins.chi, np.ones(2))
    np.testing.assert_allclose(g.z, 0.0)


def test_arithmetic_and_copy():
    a, b = make((3,), seed=1), make((3,), seed=2)
    np.testing.assert_allclose((a + b).masses.m1, a.masses.m1 + b.masses.m1)
    np.testing.assert_allclose((a - b).spins.chi, a.spins.chi - b.spins.chi)
    np.testing.assert_allclose((-a).z, -a.z)
    np.testing.assert_allclose((0.5 * a).masses.q, 0.5 * a.masses.q)
    np.testing.assert_allclose((a / 2).masses.q, a.masses.q / 2)
    assert isinstance(a - 0.1 * b, BBH)

    m = BBH.from_dict({k: np.asarray(v) for k, v in a.to_flat_dict().items()})
    c = m.copy()
    c.masses.m1[0] = -1.0
    assert m.masses.m1[0] != -1.0


# ------------------------------------------------------------ display

def test_display():
    b = make((3,))
    html = b._repr_html_()
    assert "Masses" in html and "Spin magnitudes" in html
    assert html.count("<details") >= 3                 # object, two child sections
    assert "m_1" in html and "component" in html
    md = b._repr_markdown_()
    assert "{m}_{n}" in md and r"{\chi}_{n}" in md
    assert r"\mathrm{component}" in md
    assert "masses=Masses(" in repr(b)
    assert "masses.m1" in LatentSpace(BBH)._repr_html_()


# ------------------------------------------------------------ latent space

def test_latent_round_trip_and_overrides():
    b = make((3,))
    space = LatentSpace(BBH, {"masses.m1": constraints.interval(0.0, 100.0)})
    assert list(space.supports) == FLAT
    assert set(space.children) == {"masses", "spins"}
    assert space.children["masses"].supports["m1"] is space.supports["masses.m1"]
    z = space.from_parameter(b)
    assert isinstance(z.masses, Masses)
    assert_same(space.to_parameter(z), b)
    assert_same(BBH.from_unconstrained(b.to_unconstrained()), b)
    with pytest.raises(KeyError, match="no fields"):
        LatentSpace(BBH, {"masses.nope": constraints.positive})


def test_latent_jacobian_matches_autodiff():
    space = LatentSpace.default(BBH)
    z = space.from_parameter(make())
    flat, unravel = jax.flatten_util.ravel_pytree(z)

    def forward(v):
        return jax.flatten_util.ravel_pytree(space.to_parameter(unravel(v)))[0]

    _, logdet = jnp.linalg.slogdet(jax.jacfwd(forward)(flat))
    np.testing.assert_allclose(space.log_abs_det_jacobian(z), logdet, rtol=1e-5)
    assert space.log_abs_det_jacobian(space.from_parameter(make((4,)))).shape == (4,)


# ------------------------------------------------------------ posteriors

def make_posterior(n=50, chains=2, theta=None):
    """A posterior with chains; theta defaults to a ``Binary`` (scalar leaves)."""
    rng = np.random.default_rng(n)
    if theta is None:
        theta = Binary(
            masses=Masses(m1=rng.uniform(5, 50, (chains, n)), q=rng.uniform(0.1, 1, (chains, n))),
            z=rng.uniform(0.1, 1, (chains, n)),
        )
    names = list(type(theta).flat_field_specs())
    lp = {f"log_prior[{k}]": np.full((chains, n), -1.0) for k in names}
    return PosteriorData(
        theta=theta,
        sample_stats={**lp, "prior": np.full((chains, n), np.exp(-len(names)))},
        extras={"mtot": np.asarray(theta.masses.m1 * (1 + theta.masses.q))},
        metadata={"event_name": "ev"},
        observed_data=Masses(m1=10.0, q=0.5),
        sample_dims=("chain", "draw"),
    )


BINARY = ["masses.m1", "masses.q", "z"]


def test_posterior_with_nested_theta():
    post = make_posterior()
    assert post.parameter_names == BINARY
    assert post.variable_names == BINARY + ["mtot"]
    np.testing.assert_allclose(post["masses.m1"], post.theta.masses.m1)
    assert isinstance(post["masses"], Masses)
    assert post[0].theta.batch_shape == (50,)
    assert post.isel(draw=slice(10, 20)).theta.masses.q.shape == (2, 10)

    df = post.to_dataframe()
    assert list(df.columns[:3]) == BINARY
    assert len(df) == 100 and "chain" in df

    back = PosteriorData.from_dataframe(df, Binary)
    assert isinstance(back.theta.masses, Masses)
    np.testing.assert_allclose(back.theta.masses.q, df["masses.q"])
    assert "log_prior[masses.m1]" in back.sample_stats

    table = post.summary()
    assert list(table.index) == BINARY
    expanded = make_posterior(theta=make((2, 50))).summary(expand=True)
    assert "spins.chi[1]" in expanded.index
    assert np.isfinite(table.loc["masses.m1", "r_hat"])

    np.testing.assert_allclose(post.marginal_prior(["masses.m1"]), np.exp(-1.0))
    assert post.content_hash() != make_posterior(n=51).content_hash()


def test_posterior_nested_local():
    E = 3
    post = PosteriorData(
        theta=Masses(m1=np.linspace(1, 2, 10), q=np.linspace(0, 1, 10)),
        local=make((10, E)),
    )
    table = post.summary()
    assert "local.masses.m1" in table.index and "local.spins.chi" in table.index


def test_to_arviz_nested():
    pytest.importorskip("arviz")
    idata = make_posterior(theta=make((2, 50))).to_arviz()
    assert idata["posterior"]["masses.m1"].dims == ("chain", "draw")
    assert idata["posterior"]["spins.chi"].dims == ("chain", "draw", "component")
    assert "m1" in idata["observed_data"]


def test_catalog_save_load_nested(tmp_path):
    post = make_posterior()
    catalog = Catalog({"ev": post}, raw=RawSamples(columns=["masses.m1", "z"], N=20))
    dp = catalog.dataproduct()
    assert dp["masses.m1"].shape == (1, 20)
    np.testing.assert_allclose(dp["prior"], np.exp(-2.0), rtol=1e-6)

    root = catalog.save(tmp_path / "cat")
    loaded = Catalog.load(root, types=[Binary, Masses])
    b = loaded.posterior("ev")
    assert_same(b.theta, post.theta.reshape_batch((100,)))      # the catalog stacks chains
    assert b.observed_data.to_dict() == {"m1": 10.0, "q": 0.5}
    assert loaded.source_key("ev") == catalog.source_key("ev")
    np.testing.assert_allclose(loaded.dataproduct()["masses.m1"], dp["masses.m1"])


def test_joint_likelihood_split_nested():
    from types import SimpleNamespace

    from astroinfer.inference.likelihoods import JointPopulationLikelihood

    measurement = SimpleNamespace(data_type=Masses, parameter_type=Binary)
    like = JointPopulationLikelihood(measurement, {"m1": np.ones(3), "q": np.ones(3)}, None)
    assert like.field_names == BINARY
    x = {"mu": 1.0, **{f"theta.{k}": np.full(3, i) for i, k in enumerate(BINARY)}}
    Lambda, theta = like.split(x)
    assert Lambda == {"mu": 1.0}
    np.testing.assert_allclose(theta.masses.q, 1.0)
    np.testing.assert_allclose(theta.z, 2.0)
