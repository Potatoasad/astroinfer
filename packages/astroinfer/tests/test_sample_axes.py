"""Named sample axes on PosteriorData: indexing, stacking, summary, export."""
import jax.numpy as jnp
import numpy as np
import pytest
from astroinfer import PosteriorData, parameter, parameter_dataclass


@parameter_dataclass
class Star:
    mu: float
    distance: object = parameter(index=("stars",))


@parameter_dataclass
class Point:
    x: float


C, D, S = 3, 50, 4


def make_posterior():
    rng = np.random.default_rng(0)
    theta = Star(mu=rng.normal(size=(C, D)), distance=rng.normal(size=(C, D, S)))
    return PosteriorData(
        theta=theta,
        sample_stats={
            "diverging": _diverging(),
            "log_likelihood": rng.normal(size=(C, D)),
        },
        extras={"membership": rng.uniform(size=(C, D, S))},
        sample_dims=("chain", "draw"),
    )


def _diverging():
    d = np.zeros((C, D), bool)
    d[0, 0] = True
    return d


def test_dims_and_sizes():
    post = make_posterior()
    assert post.sample_dims == ("chain", "draw")
    assert post.sizes == {"chain": C, "draw": D}
    assert post.num_samples == C * D


def test_unnamed_multi_axis_posterior_is_refused():
    with pytest.raises(ValueError, match="sample_dims"):
        PosteriorData(theta=Point(x=np.zeros((2, 3))))
    with pytest.raises(ValueError, match="every batch"):
        PosteriorData(theta=Point(x=np.zeros((2, 3))), sample_dims=("draw",))
    assert PosteriorData(theta=Point(x=np.zeros(5))).sample_dims == ("draw",)


def test_indexing_keeps_dims_in_step():
    post = make_posterior()
    first = post[0]
    assert first.sample_dims == ("draw",)
    assert first.theta.distance.shape == (D, S)
    assert first.extras["membership"].shape == (D, S)

    tail = post[:, 20:]
    assert tail.sizes == {"chain": C, "draw": D - 20}

    thinned = post[:, ::10]
    assert thinned.sizes == {"chain": C, "draw": 5}

    single = post[1, 3]
    assert single.sample_dims == ()


def test_isel_by_name():
    post = make_posterior()
    np.testing.assert_array_equal(post.isel(chain=0).theta.mu, post[0].theta.mu)
    np.testing.assert_array_equal(
        post.isel(draw=slice(20, None)).theta.mu, post[:, 20:].theta.mu
    )
    with pytest.raises(KeyError):
        post.isel(walker=0)


def test_stack_samples():
    post = make_posterior()
    flat = post.stack_samples()
    assert flat.sample_dims == ("draw",)
    assert flat.theta.mu.shape == (C * D,)
    assert flat.theta.distance.shape == (C * D, S)
    assert flat.extras["membership"].shape == (C * D, S)
    np.testing.assert_array_equal(flat.sample_stats["chain"], np.repeat(np.arange(C), D))
    # C order: chain 0's draws come first.
    np.testing.assert_allclose(flat.theta.mu[:D], post.theta.mu[0])
    assert flat.stack_samples() is flat


def test_local_follows_sample_axes():
    E = 6
    post = PosteriorData(
        theta=Point(x=np.zeros((C, D))),
        local=Point(x=np.arange(C * D * E, dtype=float).reshape(C, D, E)),
        sample_dims=("chain", "draw"),
    )
    assert post.local_dims == ("chain", "draw", "observation")
    assert post[0].local.batch_shape == (D, E)
    assert post.stack_samples().local.batch_shape == (C * D, E)
    with pytest.raises(ValueError, match="local"):
        PosteriorData(theta=Point(x=np.zeros((C, D))), local=Point(x=np.zeros((C, E))),
                      sample_dims=("chain", "draw"))


def test_to_dataframe_stacks_and_refuses_vectors():
    post = make_posterior()
    with pytest.raises(ValueError, match="one scalar per draw"):
        post.to_dataframe()

    scalar = PosteriorData(
        theta=Point(x=np.ones((C, D))),
        sample_stats={"log_likelihood": np.zeros((C, D))},
        sample_dims=("chain", "draw"),
    )
    df = scalar.to_dataframe()
    assert len(df) == C * D
    assert set(df.columns) == {"x", "log_likelihood", "chain"}


def test_summary_collapses_vector_fields():
    post = make_posterior()
    table = post.summary()
    assert list(table.index) == ["mu", "distance"]
    assert table.loc["distance", "size"] == S
    assert np.isnan(table.loc["distance", "mean"])
    assert np.isfinite(table.loc["mu", "r_hat"])
    assert table.attrs["divergences"] == 1

    expanded = post.summary(["distance"], expand=True)
    assert list(expanded.index) == [f"distance[{i}]" for i in range(S)]

    # Without chains there is no R-hat.
    assert np.isnan(post.stack_samples().summary(["mu"]).loc["mu", "r_hat"])


def test_summary_of_local_fields():
    E = 5
    rng = np.random.default_rng(1)
    post = PosteriorData(
        theta=Point(x=rng.normal(size=(C, D))),
        local=Point(x=rng.normal(size=(C, D, E))),
        sample_dims=("chain", "draw"),
    )
    table = post.summary()
    assert list(table.index) == ["x", "local.x"]
    assert table.loc["local.x", "size"] == E


def test_content_hash_sees_local():
    a = PosteriorData(theta=Point(x=jnp.zeros(4)), local=Point(x=jnp.zeros((4, 2))))
    b = PosteriorData(theta=Point(x=jnp.zeros(4)), local=Point(x=jnp.ones((4, 2))))
    assert a.content_hash() != b.content_hash()


def test_to_arviz():
    pytest.importorskip("arviz")
    idata = make_posterior().to_arviz()
    assert idata["posterior"]["distance"].dims == ("chain", "draw", "stars")
    assert idata["posterior"]["mu"].dims == ("chain", "draw")
    assert "diverging" in idata["sample_stats"]

    E = 3
    joint = PosteriorData(
        theta=Point(x=np.zeros((2, 10))),
        local=Point(x=np.zeros((2, 10, E))),
        sample_stats={"potential_energy": np.ones((2, 10))},
        sample_dims=("chain", "draw"),
    ).to_arviz()
    assert joint["posterior"]["local.x"].dims == ("chain", "draw", "observation")
    np.testing.assert_allclose(joint["sample_stats"]["lp"], -1.0)

    # A posterior without chains is given one.
    flat = PosteriorData(theta=Point(x=np.zeros(10))).to_arviz()
    assert flat["posterior"]["x"].shape == (1, 10)
