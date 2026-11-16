# Installation

astroinfer is not on PyPI yet. Install it from git, naming the package's subdirectory:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer"
```

astroinfer samples with [gravpop](https://git.ligo.org/asad.hussain/gravpop), which is not on
PyPI either, so your project's `pyproject.toml` has to say where it comes from:

```toml
[tool.uv.sources]
gravpop = { git = "https://git.ligo.org/asad.hussain/gravpop", rev = "simplify-composite" }
```

The extension packages are installed the same way. List `astroinfer` as well, so that it also
comes from git:

```bash
uv add "astroinfer @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer" \
       "astroinfer-tgmm @ git+https://github.com/Potatoasad/astroinfer#subdirectory=packages/astroinfer-tgmm"
```

`astroinfer-tgmm` fits with
[truncatedgaussianmixtures](https://github.com/Potatoasad/truncatedgaussianmixtures), which
needs its own source entry:

```toml
truncatedgaussianmixtures = { git = "https://github.com/Potatoasad/truncatedgaussianmixtures", rev = "jax-backend" }
```

The first TGMM fit installs Julia into the environment, which takes a few minutes once.

JAX uses 32-bit floats by default. Every example in these docs switches on 64-bit floats at the
start, before any arrays are created, and you will usually want to do the same.

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
