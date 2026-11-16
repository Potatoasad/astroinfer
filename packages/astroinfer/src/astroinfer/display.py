"""Rich HTML / Markdown rendering shared by every displayable object.

Nothing here knows about JAX, parameters or inference: it only turns
LaTeX strings and arrays into notebook-friendly HTML. Works in Marimo,
with a MathJax fallback for Jupyter.
"""
import sys
from html import escape
from math import prod

import numpy as np

# ============================================================
# Options
# ============================================================

_OPTIONS = {"expanded": False}


def set_options(*, expanded: bool | None = None) -> None:
    """Set how objects are displayed in notebooks.

    expanded:
        If True, the display of every object starts open, showing its contents,
        instead of as a single line that has to be clicked. The documentation
        uses this. Sections inside a display keep their own defaults.
    """
    if expanded is not None:
        _OPTIONS["expanded"] = bool(expanded)


def details_open() -> str:
    """The attribute that opens an object's top-level <details>, if requested."""
    return " open" if _OPTIONS["expanded"] else ""


# ============================================================
# Shared rendering utilities
# ============================================================

def inline_math(latex: str) -> str:
    """Render inline LaTeX in Marimo, with a Jupyter (MathJax) fallback.

    Uses marimo only when actually running inside a marimo notebook, so
    Jupyter output is the same whether or not marimo is installed.
    """
    mo = sys.modules.get("marimo")
    if mo is None or not mo.running_in_notebook():
        return (
            '<span class="math">'
            + r'\(' + escape(latex) + r'\)'
            + '</span>'
        )

    rendered = mo.md(f"${latex}$").text.strip()

    # Marimo Markdown may wrap inline math in a paragraph.
    if rendered.startswith("<p>") and rendered.endswith("</p>"):
        rendered = rendered[3:-4]

    return f'<span style="display:inline">{rendered}</span>'


def inline_row(*parts: str, separated: bool = False) -> str:
    """One flex row, wrapping only when necessary."""
    border = (
        "border-top:1px solid #8884;padding-top:0.4em;"
        if separated else ""
    )

    return (
        '<div style="display:flex;align-items:baseline;'
        'flex-wrap:wrap;gap:0.4em;margin:0.3em 0;'
        + border + '">'
        + "".join(parts)
        + "</div>"
    )


def array_preview(value, max_full=8, max_last=3):
    """Small array preview that preserves dimensional nesting."""
    try:
        shape = tuple(np.shape(value))

        if not shape:
            return np.array2string(
                np.asarray(value), precision=4
            )

        if prod(shape) <= max_full:
            limits = shape
        else:
            limits = tuple(
                min(n, max_last if i == len(shape) - 1 else 1)
                for i, n in enumerate(shape)
            )

        # Copy only the small corner being displayed.
        corner = np.asarray(
            value[tuple(slice(0, n) for n in limits)]
        )

        def nested(x, axis):
            if axis == len(shape):
                return np.array2string(
                    np.asarray(x), precision=4
                )

            items = [
                nested(sub, axis + 1)
                for sub in x
            ]

            if limits[axis] < shape[axis]:
                items.append("...")

            return "[" + ", ".join(items) + "]"

        return nested(corner, 0)

    except Exception:  # noqa: BLE001 - best-effort preview of arbitrary objects
        return f"<array shape={np.shape(value)}>"


def array_node(value) -> str:
    """Collapsible preview with the native array repr underneath."""
    preview = escape(array_preview(value))

    try:
        full = escape(repr(value))
    except Exception:  # noqa: BLE001 - best-effort preview of arbitrary objects
        full = "&lt;unavailable&gt;"

    return (
        '<details style="display:inline-block;'
        'max-width:100%;vertical-align:middle">'

        '<summary style="cursor:pointer">'
        '<code>' + preview + '</code>'
        '</summary>'

        '<pre style="'
        'margin:0.4em 0;'
        'padding:0.5em;'
        'max-height:250px;'
        'max-width:80ch;'
        'overflow:auto;'
        'white-space:pre;'
        'background:rgba(128,128,128,0.07);'
        'border-radius:5px;'
        'font-size:0.85em">'
        + full +
        '</pre></details>'
    )


def shape_badge(value) -> str:
    return (
        '<span style="opacity:0.7;font-size:0.9em">'
        'shape=<code>'
        + escape(str(np.shape(value))) +
        '</code></span>'
    )


def parameter_row(name: str, latex: str, value) -> str:
    """Display a named parameter and its collapsible array."""
    return inline_row(
        '<strong>' + escape(name) + ':</strong>',
        inline_math(latex + " ="),
        array_node(value),
        shape_badge(value),
    )


def latex_name(name: str) -> str:
    r"""Axis or field name as upright LaTeX, e.g. ``\mathrm{n\_flips}``."""
    return r"\mathrm{" + name.replace("_", r"\_") + "}"


# ============================================================
# Building blocks for class displays
# ============================================================

_CELL = "padding:0.15em 0.7em 0.15em 0;text-align:left;vertical-align:top"


def code(value) -> str:
    return "<code>" + escape(str(value)) + "</code>"


def section(title: str, body: str, *, opened: bool = False, count=None) -> str:
    """A collapsible sub-section."""
    badge = (f' <span style="opacity:0.65">({escape(str(count))})</span>'
             if count is not None else "")
    return (
        "<details" + (" open" if opened else "") + ' style="margin:0.4em 0">'
        '<summary style="cursor:pointer"><strong>' + escape(title) + "</strong>"
        + badge + "</summary>"
        '<div style="padding:0.3em 0 0.3em 1.2em">' + body + "</div></details>"
    )


def kv_table(rows) -> str:
    """Two-column table of ``{name: value}``. Values are shown as code
    (strings as-is, others by repr); wrap ready-made HTML in ``html_value``."""
    out = []
    for name, value in dict(rows).items():
        cell = value.html if isinstance(value, HTMLValue) else code(
            value if isinstance(value, str) else repr(value))
        out.append(f'<tr><td style="{_CELL};opacity:0.8">{escape(str(name))}</td>'
                   f'<td style="{_CELL}">{cell}</td></tr>')
    return '<table style="border-collapse:collapse;margin:0.2em 0">' + "".join(out) + "</table>"


def html_table(header, rows) -> str:
    """A small table; cells are HTML strings."""
    head = "".join(f'<th style="{_CELL};border-bottom:1px solid #8886">{h}</th>' for h in header)
    body = "".join("<tr>" + "".join(f'<td style="{_CELL}">{c}</td>' for c in row) + "</tr>"
                   for row in rows)
    return ('<table style="border-collapse:collapse;margin:0.3em 0;font-size:0.95em">'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


_BADGE_COLORS = {"ok": "46,160,67", "warn": "210,140,20", "bad": "210,60,60", "none": "128,128,128"}


def badge(text: str, kind: str = "none") -> str:
    rgb = _BADGE_COLORS[kind]
    return (f'<span style="background:rgba({rgb},0.16);color:rgb({rgb});border-radius:4px;'
            f'padding:0 0.4em;font-size:0.85em;white-space:nowrap">{escape(text)}</span>')


def method_row(name: str, signature_latex: str, description: str) -> str:
    return (
        '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.45em;margin:0.25em 0">'
        + code(name) + inline_math(signature_latex)
        + '<span style="opacity:0.7;font-size:0.9em">' + escape(description) + "</span></div>"
    )


def math_block(lines) -> str:
    """One equation per line, left aligned."""
    return "".join(f'<div style="margin:0.15em 0">{inline_math(line)}</div>' for line in lines)


class HTMLValue:
    """Wrap an HTML string so ``kv_table`` shows it unescaped."""

    def __init__(self, html: str):
        self.html = html


def html_value(html: str) -> HTMLValue:
    return HTMLValue(html)


# ============================================================
# Shared rich display base class
# ============================================================

class RichDisplay:
    """
    Mixin for collapsible mathematical HTML representations.

    Subclasses may override:
        _display_title() -> str
        _display_math()  -> str | None
        _display_body()  -> str (HTML)

    All HTML rendering is handled here.
    """

    def _display_title(self) -> str:
        return getattr(
            type(self),
            "_display_name",
            type(self).__name__,
        )

    def _display_math(self) -> str | None:
        return None

    def _display_body(self) -> str:
        return ""

    def _repr_html_(self) -> str:
        latex = self._display_math()

        header = (
            "<strong>"
            + escape(self._display_title())
            + ":</strong>"
        )

        if latex:
            header += inline_math(latex)

        return (
            '<details' + details_open() + ' style="font-family:inherit;line-height:1.5">'

            '<summary style="cursor:pointer;padding:0.25em 0">'
            '<span style="display:inline-flex;'
            'align-items:baseline;gap:0.4em">'
            + header +
            '</span></summary>'

            '<div style="padding:0.4em 0 0.4em 1.2em">'
            + self._display_body() +
            '</div>'

            '</details>'
        )

    def _repr_markdown_(self) -> str:
        latex = self._display_math()
        title = self._display_title()

        if latex:
            return f"**{title}:** ${latex}$"

        return f"**{title}**"
