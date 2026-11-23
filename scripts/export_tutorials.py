"""Convert the marimo tutorials into Jupyter notebooks for the documentation.

    uv run --with nbformat python scripts/export_tutorials.py

writes docs/examples/<name>.ipynb for every tutorials/<digit>*.py. The notebooks have no
outputs; Jupyter Book executes them when it builds the site, like every other page. They are
generated files and are not committed. (The gravitational-wave notebooks in docs/examples/gw
are different: scripts/export_gw_examples.py executes them ahead of time and they are committed.)

Each notebook gets a hidden first cell that installs scripts/marimo_docs_shim.py as `marimo`.
Cells whose code is hidden in marimo and that only show static text (`mo.md`, `mo.callout`,
`mo.accordion`) or an image (`mo.image`) become markdown cells; tutorials/images/ is copied
next to the notebooks. Exercise cells are emptied back to their placeholder.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
OUTPUT = ROOT / "docs" / "examples"
REPOSITORY = "https://github.com/Potatoasad/astroinfer/blob/main"
EXERCISE_PLACEHOLDER = "# your code here"

sys.path.insert(0, str(ROOT / "scripts"))
import marimo_docs_shim  # noqa: E402

# Jupyter Book executes the tutorials from docs/examples; scripts/export_gw_examples.py executes
# its notebooks from the repository root. The setup finds the root from either. {notebook_dir} is
# the folder of the marimo notebook, relative to the root.
SETUP = """\
import sys
from pathlib import Path

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / "scripts" / "marimo_docs_shim.py").is_file())
sys.path[:0] = [str(ROOT / "{notebook_dir}"), str(ROOT / "scripts")]

import marimo_docs_shim
marimo_docs_shim.set_notebook_dir(ROOT / "{notebook_dir}")
sys.modules["marimo"] = marimo_docs_shim

# marimo shows a figure only when a cell returns it. Stop Jupyter from also showing every
# figure a cell creates, and close them once the cell has run.
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import flush_figures
get_ipython().run_line_magic("matplotlib", "inline")
try:
    get_ipython().events.unregister("post_execute", flush_figures)
except ValueError:
    pass
get_ipython().events.register("post_execute", lambda: plt.close("all"))

# Show astroinfer objects with their HTML display (as Jupyter does), not the short markdown one.
get_ipython().display_formatter.formatters["text/markdown"].enabled = False
# Open every object display, so readers see the contents without having to click.
import astroinfer.display
astroinfer.display.set_options(expanded=True)
"""


def code_cell(source: str, tags: list[str], id: str) -> dict:
    return {"cell_type": "code", "id": id, "metadata": {"tags": tags}, "source": source,
            "outputs": [], "execution_count": None}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def static_markdown(source: str) -> str | None:
    """The markdown that a hidden marimo cell shows, if it does not depend on other cells."""
    try:
        value = eval(compile(source, "<cell>", "eval"), {"mo": marimo_docs_shim})
    except Exception:
        return None
    return value.text if isinstance(value, marimo_docs_shim._Markdown) else None


def convert(notebook: dict, path: Path, prefix: str | None = None,
            frontmatter: str | None = None) -> dict:
    """Turn marimo's export of `path` into the notebook shown in the docs.

    `prefix` starts every cell id (the first two characters of the file name by default), and
    `frontmatter`, if given, is YAML that goes at the top of the first cell as the page's MyST
    frontmatter.
    """
    relative = path.relative_to(ROOT).as_posix()
    prefix = path.stem[:2] if prefix is None else prefix
    note = (f"This page was generated from [`{relative}`]({REPOSITORY}/{relative}), a marimo "
            "notebook. Sliders and other controls are shown at their default values. To change "
            f"them, run the notebook with `uv run marimo edit {relative}`.")

    cells = []
    for cell in notebook["cells"]:
        # marimo's cell ids repeat across notebooks, and Jupyter Book needs them to be unique.
        cell["id"] = f"{prefix}-{cell.get('id', len(cells))}"
        source = "".join(cell["source"])
        hidden = cell["metadata"].get("marimo", {}).get("config", {}).get("hide_code", False)
        if cell["cell_type"] == "code" and hidden:
            text = static_markdown(source)
            if text is not None:
                cells.append({**markdown_cell(text), "id": cell["id"]})
                continue
        if cell["cell_type"] == "code" and source.lstrip().startswith(EXERCISE_PLACEHOLDER):
            cell["source"] = EXERCISE_PLACEHOLDER
        cells.append(cell)

    # The page must open with the title cell, or Jupyter Book shows the title twice. The source
    # note goes directly under the title. The hidden docs setup and marimo's setup cell (the
    # imports), which marimo puts first, follow the introduction.
    first = next(c for c in cells if c["cell_type"] == "markdown")
    title, _, rest = "".join(first["source"]).partition("\n")
    first["source"] = f"{title}\n\n{note}\n{rest}"
    if frontmatter:
        first["source"] = f"---\n{frontmatter.strip()}\n---\n{first['source']}"
    setup = [c for c in cells if c["metadata"].get("marimo", {}).get("name") == "setup"]
    for cell in setup:
        cells.remove(cell)
    setup_code = SETUP.format(notebook_dir=path.parent.relative_to(ROOT).as_posix())
    docs_setup = code_cell(setup_code, ["remove-cell"], f"{prefix}-docs-setup")
    cells.remove(first)
    cells = [first, docs_setup, *setup, *cells]

    notebook["cells"] = cells
    notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                          "language": "python"}
    return notebook


def export(path: Path) -> dict:
    """The marimo notebook at `path` as a Jupyter notebook (a dict), without outputs."""
    with tempfile.TemporaryDirectory() as tmp:
        exported = Path(tmp) / "notebook.ipynb"
        subprocess.run([sys.executable, "-m", "marimo", "export", "ipynb", str(path),
                        "-o", str(exported), "--sort", "top-down", "-f"],
                       check=True, capture_output=True)
        return json.loads(exported.read_text())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    marimo_docs_shim.set_notebook_dir(TUTORIALS)
    tutorials = sorted(TUTORIALS.glob("[0-9]*.py"))
    # Remove notebooks of tutorials that were renamed or deleted. The glob is not recursive, so
    # the pre-executed notebooks in docs/examples/gw (scripts/export_gw_examples.py) are kept.
    for old in OUTPUT.glob("*.ipynb"):
        if old.stem not in {path.stem for path in tutorials}:
            old.unlink()
    if (TUTORIALS / "images").is_dir():
        shutil.copytree(TUTORIALS / "images", OUTPUT / "images", dirs_exist_ok=True)
    for path in tutorials:
        target = OUTPUT / f"{path.stem}.ipynb"
        target.write_text(json.dumps(convert(export(path), path), indent=1) + "\n")
        print("wrote", target.relative_to(ROOT))


if __name__ == "__main__":
    main()
