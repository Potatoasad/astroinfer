"""Convert the gravitational-wave notebooks into executed Jupyter notebooks for the documentation.

    uv run --with nbformat --with nbclient --with ipykernel python scripts/export_gw_examples.py

writes docs/examples/gw/<name>.ipynb for every notebooks/gw/<digit>*.py. Unlike the tutorials
(scripts/export_tutorials.py), these notebooks take a long time to run, and 02 needs Julia, so
they are executed here, once, with their outputs stored, and the notebooks are committed. Each
one opens with the page frontmatter `execute: {skip: true}`, so `jupyter-book build --execute`
shows the stored outputs instead of running the notebook again.

The conversion is the one used for the tutorials (see that script and
scripts/marimo_docs_shim.py). The notebooks are executed from the repository root, as marimo
runs them, and `mo.notebook_dir()` is notebooks/gw, so they read and write notebooks/data/ as
usual. Notebook 01 reloads notebooks/data/gw_catalog if it exists instead of rerunning PE; delete
it first to export a fresh run. Output written to stderr (progress bars, warnings) is dropped,
since the docs do not show it. A notebook whose execution fails is not written, and the script
carries on with the others and exits with an error.

Options:

    NOTEBOOK ...     export only these, given by number (01), name (01_gw_mock_catalog) or
                     path to a marimo notebook (any marimo notebook, which is useful for testing)
    --skip NOTEBOOK  leave this one out, for example --skip 02 on a machine without Julia; the
                     existing docs/examples/gw/02_*.ipynb is kept
    --timeout HOURS  the longest a single cell may run (default 6)
    --output DIR     write to DIR instead of docs/examples/gw
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marimo_docs_shim  # noqa: E402
from export_tutorials import ROOT, convert, export  # noqa: E402

NOTEBOOKS = ROOT / "notebooks" / "gw"
OUTPUT = ROOT / "docs" / "examples" / "gw"
FRONTMATTER = """\
# Executed by scripts/export_gw_examples.py; the docs build shows the stored outputs.
execute:
  skip: true
"""


def select(names: list[str]) -> list[Path]:
    """The marimo notebooks that `names` refer to (all of notebooks/gw if `names` is empty)."""
    available = sorted(NOTEBOOKS.glob("[0-9]*.py"))
    if not names:
        return available
    paths = []
    for name in names:
        if name.endswith(".py") and Path(name).is_file():
            paths.append(Path(name).resolve())
            continue
        matches = [path for path in available if path.stem == name or path.stem[:2] == name]
        if len(matches) != 1:
            sys.exit(f"{name!r} does not name one of {[path.stem for path in available]}")
        paths.append(matches[0])
    return paths


def execute(path: Path, output: Path, timeout: float) -> None:
    marimo_docs_shim.set_notebook_dir(path.parent)
    converted = convert(export(path), path, prefix=f"gw{path.stem[:2]}", frontmatter=FRONTMATTER)
    # Reading the JSON joins the cell sources, which marimo writes as lists of lines.
    notebook = nbformat.reads(json.dumps(converted), as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = [out for out in cell.outputs
                            if not (out.output_type == "stream" and out.name == "stderr")]
    nbformat.validate(notebook)
    # Write next to the target and then rename, so a failed run leaves the old notebook in place.
    target = output / f"{path.stem}.ipynb"
    with tempfile.NamedTemporaryFile("w", dir=output, suffix=".ipynb", delete=False) as file:
        nbformat.write(notebook, file)
    Path(file.name).chmod(0o644)
    Path(file.name).replace(target)
    if (path.parent / "images").is_dir():
        shutil.copytree(path.parent / "images", output / "images", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("notebooks", nargs="*", metavar="NOTEBOOK")
    parser.add_argument("--skip", action="append", default=[], metavar="NOTEBOOK")
    parser.add_argument("--timeout", type=float, default=6.0, metavar="HOURS")
    parser.add_argument("--output", type=Path, default=OUTPUT, metavar="DIR")
    args = parser.parse_args()

    skipped = set(select(args.skip)) if args.skip else set()
    paths = [path for path in select(args.notebooks) if path not in skipped]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    failed = []
    for path in paths:
        print(f"executing {path.relative_to(ROOT)}", flush=True)
        start = time.perf_counter()
        try:
            execute(path, output, timeout=args.timeout * 3600)
        except CellExecutionError as error:
            print(f"failed: {path.relative_to(ROOT)}\n{error}", file=sys.stderr, flush=True)
            failed.append(path)
            continue
        minutes = (time.perf_counter() - start) / 60
        print(f"wrote {output / f'{path.stem}.ipynb'} ({minutes:.1f} min)", flush=True)
    if failed:
        sys.exit(f"{len(failed)} notebook(s) failed: {', '.join(path.stem for path in failed)}")


if __name__ == "__main__":
    main()
