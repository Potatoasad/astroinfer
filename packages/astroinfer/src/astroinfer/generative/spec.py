"""Parameter specs. A distribution parameter (mu, sigma, a, b, ...) is one of:
  - a constant (float),
  - a hyperparameter name (str), read from L,
  - a Conditional, computed from earlier-drawn variables x and hyperparams L.

Conditioning is a property of a PARAMETER, not a new distribution class."""
from collections.abc import Callable

import jax

Array = jax.Array
ConditionedOn = dict[str, Array]   # x: variables drawn so far
Params = dict[str, Array]          # L: hyperparameters (Lambda)


class Conditional:
    """fn(x, L) -> Array. `on` names the drawn variables fn reads, and is the
    single source of truth for ordering validation. If a lambda reads a key
    not in `on`, and that key precedes anyway, it is harmless; if it does not
    precede, it fails loudly with a KeyError at sample time."""
    def __init__(self, fn: Callable[[ConditionedOn, Params], Array],
                 on: str | list[str]):
        self.fn = fn
        self.on = [on] if isinstance(on, str) else list(on)

    def __call__(self, x, L):
        return self.fn(x, L)


def conditional(fn, on) -> Conditional:
    return Conditional(fn, on)


type ParamSpec = float | str | Conditional


def resolve(spec: ParamSpec, x: ConditionedOn, L: Params) -> Array:
    if isinstance(spec, Conditional):
        return spec(x, L)
    if isinstance(spec, str):
        return L[spec]
    return spec


def spec_deps(spec: ParamSpec) -> list[str]:
    return spec.on if isinstance(spec, Conditional) else []