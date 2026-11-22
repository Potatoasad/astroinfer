"""Run NumPyro's NUTS or nested sampler on a model function.

``run_mcmc`` does what ``gravpop.Sampler`` does around NumPyro, with the
same keyword names and defaults, so a model run here gives the same draws as
the same model run by gravpop with the same seed and settings:

    NUTS(model, target_accept_prob=..., max_tree_depth=..., **sampler_kwargs)
    MCMC(kernel, num_warmup=..., num_samples=..., num_chains=...,
         jit_model_args=True, **mcmc_kwargs)

The run key is the second half of ``jax.random.split(PRNGKey(seed))``.
NUTS records the extra fields ``EXTRA_FIELDS`` as sample stats. With
``method="nested"`` the model goes to
``numpyro.contrib.nested_sampling.NestedSampler(model, **sampler_kwargs)``
(which needs ``jaxns``), and ``num_samples`` draws are taken from the
result with a key split from the run key.
"""
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np

METHODS = ("nuts", "nested")

EXTRA_FIELDS = (
    "potential_energy",
    "energy",
    "num_steps",
    "accept_prob",
    "diverging",
    "adapt_state.step_size",
)


def _resolve_seed(seed) -> int:
    if isinstance(seed, (int, np.integer)):
        return int(seed)
    return int(np.random.default_rng(seed).integers(2**30))


@dataclass(repr=False)
class RunResult:
    """
    The outcome of ``run_mcmc``.

    method:
        ``"nuts"`` or ``"nested"``.
    sampler:
        The NumPyro ``MCMC`` object, or the ``NestedSampler``.
    sites:
        ``{site: array}`` of every sample and deterministic site, with the
        sample axes first.
    sample_stats:
        ``{name: array}`` of the NUTS extra fields; empty for nested sampling.
    sample_dims:
        Names of the sample axes: ``("chain", "draw")`` for NUTS,
        ``("draw",)`` for nested sampling.
    seed:
        The integer seed the run used.
    """

    method: str
    sampler: Any
    sites: dict = field(default_factory=dict)
    sample_stats: dict = field(default_factory=dict)
    sample_dims: tuple = ()
    seed: int | None = None

    def get_samples(self, group_by_chain=False):
        """``{site: array}``; chains are concatenated unless ``group_by_chain``."""
        chained = self.sample_dims == ("chain", "draw")
        if group_by_chain and not chained:
            raise ValueError(f"method={self.method!r} has no chains")
        if group_by_chain or not chained:
            return dict(self.sites)
        return {
            k: np.reshape(v, (-1,) + np.shape(v)[2:]) for k, v in self.sites.items()
        }

    def __repr__(self):
        shape = dict(zip(self.sample_dims, _sample_shape(self)))
        return (
            f"RunResult(method={self.method!r}, seed={self.seed}, "
            f"sites={list(self.sites)}, sizes={shape})"
        )


def _sample_shape(result):
    for v in result.sites.values():
        return np.shape(v)[:len(result.sample_dims)]
    return ()


def _is_array_tree(value):
    """True if every leaf of ``value`` is a NumPy or JAX array.

    Plain Python numbers count as static: they are often sizes (of a plate,
    say), which must stay concrete inside the model.
    """
    return all(
        isinstance(leaf, (jax.Array, np.ndarray, np.generic))
        for leaf in jax.tree.leaves(value)
    )


def _bind_static(model, args, kwargs):
    """Bind model arguments that are not arrays (e.g. a measurement object).

    ``MCMC(jit_model_args=True)`` passes the model arguments through ``jit``,
    which only accepts pytrees of arrays. Other arguments are bound into the
    model instead, so they reach it unchanged.
    """
    static = {i: a for i, a in enumerate(args) if not _is_array_tree(a)}
    static_kw = {k: v for k, v in kwargs.items() if not _is_array_tree(v)}
    if not static and not static_kw:
        return model, args, kwargs
    n = len(args)

    def bound(*dynamic, **dynamic_kw):
        it = iter(dynamic)
        full = [static[i] if i in static else next(it) for i in range(n)]
        return model(*full, **dynamic_kw, **static_kw)

    dynamic = tuple(a for i, a in enumerate(args) if i not in static)
    dynamic_kw = {k: v for k, v in kwargs.items() if k not in static_kw}
    return bound, dynamic, dynamic_kw


def run_mcmc(
    model,
    *args,
    method="nuts",
    seed=None,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    target_accept_prob=0.7,
    max_tree_depth=6,
    summary=True,
    sampler_kwargs=None,
    mcmc_kwargs=None,
    **model_kwargs,
):
    """
    Sample the posterior of a NumPyro model.

    ``args`` and ``model_kwargs`` are passed to the model. ``seed`` is an
    int, or None for a random one. NUTS settings beyond
    ``target_accept_prob`` and ``max_tree_depth`` (``dense_mass``,
    ``init_strategy``, ...) go in ``sampler_kwargs``; ``MCMC`` settings
    (``chain_method``, ``progress_bar``, ...) go in ``mcmc_kwargs``. For
    ``method="nested"``, ``sampler_kwargs`` is passed to ``NestedSampler``
    and the NUTS and MCMC settings are ignored. ``summary=True`` prints the
    sampler's summary.

    Returns a ``RunResult``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    model, args, model_kwargs = _bind_static(model, args, model_kwargs)
    seed = _resolve_seed(seed)
    sampler_kwargs = dict(sampler_kwargs or {})
    _, rng_key = jax.random.split(jax.random.PRNGKey(seed))

    if method == "nested":
        try:
            from numpyro.contrib.nested_sampling import NestedSampler
        except ImportError as exc:
            raise ImportError(
                "Nested sampling requires jaxns; install it with `pip install jaxns`"
            ) from exc
        rng_key, rng_key_samples = jax.random.split(rng_key)
        ns = NestedSampler(model, **sampler_kwargs)
        ns.run(rng_key, *args, **model_kwargs)
        if summary:
            ns.print_summary()
        sites = ns.get_samples(rng_key_samples, num_samples=num_samples)
        return RunResult("nested", ns, dict(sites), {}, ("draw",), seed)

    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(
        model,
        **{
            "target_accept_prob": target_accept_prob,
            "max_tree_depth": max_tree_depth,
            **sampler_kwargs,
        },
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        **{"num_chains": num_chains, "jit_model_args": True, **(mcmc_kwargs or {})},
    )
    mcmc.run(rng_key, *args, extra_fields=EXTRA_FIELDS, **model_kwargs)
    if summary:
        mcmc.print_summary(exclude_deterministic=False)
    return RunResult(
        "nuts",
        mcmc,
        dict(mcmc.get_samples(group_by_chain=True)),
        dict(mcmc.get_extra_fields(group_by_chain=True)),
        ("chain", "draw"),
        seed,
    )
