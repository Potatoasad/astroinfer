# Installation

astroinfer is not on PyPI yet. Install it from git, together with `astroinfer-gw`, the
gravitational-wave models and mock catalogs:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
       "astroinfer-gw @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-gw"
```

or, with pip:

```bash
pip install "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
            "astroinfer-gw @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-gw"
```

If you do not need the gravitational-wave models, leave out the second line. `astroinfer` is
listed explicitly in both cases so that it also comes from git.

astroinfer samples with [gravpop](https://git.ligo.org/asad.hussain/gravpop), which is not on
PyPI either. astroinfer's own requirements point at gravpop's git repository, so it is installed
along with astroinfer and needs nothing in your `pyproject.toml`.

JAX uses 32-bit floats by default. Every example in these docs switches on 64-bit floats at the
start, before any arrays are created, and you will usually want to do the same.

## Truncated Gaussian mixtures

`astroinfer-tgmm` represents catalog posteriors as truncated Gaussian mixtures. Most users do
not need it. It fits with
[truncatedgaussianmixtures](https://github.com/Potatoasad/truncatedgaussianmixtures), which runs
on Julia and also comes from git, so name it in the same command:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
       "astroinfer-tgmm @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-tgmm" \
       "truncatedgaussianmixtures @ git+https://github.com/Potatoasad/truncatedgaussianmixtures@jax-backend"
```

uv records the git location of `truncatedgaussianmixtures` in your project's
`[tool.uv.sources]`; with pip, replace `uv add` with `pip install`. The first TGMM fit installs
Julia into the environment, which takes a few minutes once.

## Working on astroinfer itself

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/). One
environment has every package installed in editable mode, together with marimo, matplotlib and
pytest:

```bash
uv sync
uv run pytest -m "not julia"   # tests that do not start Julia
uv run pytest                  # all tests
scripts/run_tutorials.sh       # run every tutorial as a script
```
