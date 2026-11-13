"""The top model: binds a distribution tree to a user-chosen global order,
validates that every conditional reads only earlier variables (or a declared
external), and drives both verbs -- sample (draw chi ~ pi_Lambda,
differentiable in Lambda) and log_prob (joint density on real data).

External variables (e.g. mass_1_source, mass_ratio, redshift) are values a
model conditions on but never draws. They are supplied via `conditioned_on`
at call time, seed the initial prefix, and are exempt from the ordering
constraint (they are always available)."""
import jax
import jax.numpy as jnp
import numpy as np

from .inversion import bisection_ppf


class GenerativeModel:
    def __init__(self, root, order, external=()):
        self.root = root
        self.order = list(order)
        self.external = set(external)
        self._validate()

    def _validate(self):
        pos = {v: i for i, v in enumerate(self.order)}
        missing = set(self.root.var_names) - set(self.order)
        if missing:
            raise ValueError(f"variables {sorted(missing)} are in the model "
                             f"but not in order {self.order}.")
        overlap = self.external & set(self.order)
        if overlap:
            raise ValueError(f"variables {sorted(overlap)} are declared both "
                             f"external and in order; a variable is one or the other.")
        for v in self.order:
            for d in self.root.deps(v):
                if d in self.external:
                    continue                      # external: always available
                if d not in pos:
                    raise ValueError(f"'{v}' is conditioned on unknown variable "
                                     f"'{d}' (not in order, not declared external).")
                if pos[d] >= pos[v]:
                    raise ValueError(f"'{v}' is conditioned on '{d}', which does "
                                     f"not precede it in {self.order}.")

    def _make_seeds(self, key, shape):
        if key is None:
            key = jax.random.PRNGKey(int(np.random.SeedSequence().generate_state(1)[0]))
        keys = jax.random.split(key, len(self.order))
        return {v: jax.random.uniform(k, shape) for v, k in zip(self.order, keys)}

    def sample(self, params, seeds=None, key=None, shape=None, conditioned_on=None):
        """Draw chi ~ pi_Lambda.

        params         : hyperparameter dict L.
        seeds          : {var: U(0,1) array} of shape (E,N). Supply directly for
                         common-random-number sweeps (FIXED across Lambda).
                         Highest precedence.
        key            : PRNGKey used to generate seeds when `seeds` is None.
        shape          : (E,N) batch shape for generated seeds. Required when
                         both `seeds` and generation-from-key are used, i.e.
                         whenever `seeds` is None.
        conditioned_on : {var: array} of EXTERNAL values (e.g. real m1,q,z) the
                         model conditions on but does not draw. Seeds the prefix.

        NOTE ON SWEEPS: passing neither `seeds` nor a fixed `key` draws fresh
        uniforms each call, which BREAKS the common-random-number property and
        reintroduces Monte-Carlo jitter across a Lambda-grid. For a sweep, pass
        a fixed `seeds` (best) or a fixed `key`.
        """
        cond = dict(conditioned_on) if conditioned_on else {}

        missing_ext = self.external - set(cond)
        if missing_ext:
            raise ValueError(f"declared external variables {sorted(missing_ext)} "
                             f"were not supplied in conditioned_on.")

        if seeds is None:
            if shape is None:
                raise ValueError("provide either `seeds` or a `shape` to generate them.")
            seeds = self._make_seeds(key, shape)

        # shape to check external arrays against: the seed batch shape
        ref_shape = shape if shape is not None else next(iter(seeds.values())).shape
        for name, arr in cond.items():
            try:
                jnp.broadcast_shapes(jnp.asarray(arr).shape, ref_shape)
            except ValueError:
                raise ValueError(
                    f"external '{name}' has shape {jnp.asarray(arr).shape}, which does "
                    f"not broadcast against the sample shape {ref_shape}.")

        x = cond
        for var in self.order:
            closed_form = self.root.cond_ppf(var, seeds[var], x, params)
            if closed_form is not None:
                # Closed-form quantile (e.g. TruncatedNormal, Uniform): a plain
                # differentiable function of (u, x, L) already -- no bisection,
                # no custom_vjp needed, and noticeably cheaper (see
                # generative_spin_model.py's benchmark). Falls through to
                # bisection below only for distributions without one (Mixture).
                drawn = closed_form
            else:
                a, b = self.root.bounds(var)
                diff = {"L": params, "x": x}

                def cdf_fn(val, diff, var=var):
                    return self.root.cond_cdf(var, val, diff["x"], diff["L"])

                drawn = bisection_ppf(cdf_fn, diff, seeds[var], a, b)
            x = {**x, var: drawn}
        return x

    def log_prob(self, data, params):
        """Joint log-density of the drawn variables on real data, conditioned on
        whatever external columns `data` also carries. Returns (E, N)."""
        x = {k: data[k] for k in self.external if k in data}
        total = 0.0
        for var in self.order:
            val = data[var]
            a, b = self.root.bounds(var)
            if np.isfinite(a) and np.isfinite(b):
                # Out-of-support values get log p = -inf. Score them at a safe
                # interior point first: log(0) would have an infinite
                # derivative, and inf * 0 cotangent = NaN gradients for the
                # whole batch (e.g. importance draws beyond the population).
                inside = (val >= a) & (val <= b)
                val = jnp.where(inside, val, 0.5 * (a + b))
                logp = jnp.where(inside, jnp.log(self.root.cond_pdf(var, val, x, params)), -jnp.inf)
            else:
                logp = jnp.log(self.root.cond_pdf(var, val, x, params))
            total = total + logp
            x = {**x, var: val}
        return total