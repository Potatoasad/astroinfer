"""A stand-in for marimo used when the tutorials are executed for the documentation.

The tutorials are marimo notebooks. For the docs, scripts/export_tutorials.py converts them to
Jupyter notebooks, and the first (hidden) cell of each one installs this module as `marimo`.
Jupyter Book then runs them like every other page. Each function here returns an object that
Jupyter displays as HTML. Markdown is converted with marimo's own renderer, and its math
elements are replaced by the `\\( \\)` and `\\[ \\]` delimiters that the docs typeset (Jupyter Book
does not render markdown outputs). Interactive elements show their label and default value, and
`.value` returns that default.

Objects made from static text also keep their markdown in `.text`. The export script uses it to
turn hidden cells that only show text into markdown cells, with `mo.callout` and `mo.accordion`
becoming MyST admonitions and dropdowns.
"""
from __future__ import annotations

import html
import inspect
import re
from pathlib import Path

# Imported before this module replaces `marimo` in sys.modules.
import marimo as _marimo

_NOTEBOOK_DIR = Path.cwd()


def set_notebook_dir(path) -> None:
    global _NOTEBOOK_DIR
    _NOTEBOOK_DIR = Path(path).resolve()


def notebook_dir() -> Path:
    return _NOTEBOOK_DIR


def running_in_notebook() -> bool:
    # astroinfer.display checks this to decide whether to emit marimo's math elements.
    return False


def _display(obj) -> None:
    from IPython.display import display as ipython_display

    ipython_display(obj)


def _as_markdown(text: str) -> str:
    return inspect.cleandoc(text)


_TEX = re.compile(r'<marimo-tex[^>]*>\|\|([(\[])(.*?)\|\|[)\]]</marimo-tex>', re.DOTALL)


def _to_html(markdown: str) -> str:
    def delimit(match):
        open_, body = match.groups()
        # The same markup astroinfer's own displays use, which the docs typeset.
        math = rf"\({body}\)" if open_ == "(" else rf"\[{body}\]"
        return f'<span class="math">{math}</span>'

    return _TEX.sub(delimit, _marimo.md(markdown).text)


class _Markdown:
    def __init__(self, text: str):
        self.text = _as_markdown(text)

    def _repr_html_(self):
        return _to_html(self.text)


class Html:
    def __init__(self, html: str):
        self.text = html

    def _repr_html_(self):
        return self.text


def md(text: str) -> _Markdown:
    return _Markdown(text)


class _Stack:
    def __init__(self, items):
        self.items = list(items)

    def _ipython_display_(self):
        for item in self.items:
            _display(md(item) if isinstance(item, str) else item)


def vstack(items, **kwargs) -> _Stack:
    return _Stack(items)


def hstack(items, **kwargs) -> _Stack:
    return _Stack(items)


_ADMONITIONS = {"info": "note", "neutral": "note", "success": "tip", "warn": "warning",
                "danger": "danger"}


def _body(value) -> str:
    return value.text if isinstance(value, _Markdown) else str(value)


class _Callout(_Markdown):
    def __init__(self, value, kind: str):
        self.body = _body(value)
        self.text = f":::{{{_ADMONITIONS.get(kind, 'note')}}}\n{self.body}\n:::"

    def _repr_html_(self):
        return ('<div style="border-left:4px solid #2a78d6;padding:0.25em 1em;margin:0.5em 0">'
                + _to_html(self.body) + "</div>")


class _Accordion(_Markdown):
    def __init__(self, items: dict):
        self.items = {label: _body(value) for label, value in items.items()}
        self.text = "\n\n".join(f":::{{dropdown}} {label}\n{body}\n:::"
                                 for label, body in self.items.items())

    def _repr_html_(self):
        return "".join(f"<details><summary>{html.escape(label)}</summary>{_to_html(body)}</details>"
                       for label, body in self.items.items())


def callout(value, kind: str = "neutral") -> _Callout:
    return _Callout(value, kind)


def accordion(items: dict, **kwargs) -> _Accordion:
    return _Accordion(items)


class _Element:
    """An interactive element, shown with its label and default value."""

    def __init__(self, value, label: str = ""):
        self.value = value
        self.label = label

    def _repr_html_(self):
        label = f"{self.label}: " if self.label else ""
        return _to_html(f"{label}`{self.value}`")


class _Dictionary(dict):
    @property
    def value(self):
        return {name: element.value for name, element in self.items()}

    def _ipython_display_(self):
        for element in self.values():
            _display(element)


class ui:
    @staticmethod
    def slider(start, stop, step=None, value=None, label="", **kwargs):
        return _Element(start if value is None else value, label)

    @staticmethod
    def dropdown(options, value=None, label="", **kwargs):
        options = list(options)
        return _Element(options[0] if value is None else value, label)

    @staticmethod
    def checkbox(label="", value=False, **kwargs):
        return _Element(value, label)

    @staticmethod
    def dictionary(elements: dict, **kwargs):
        return _Dictionary(elements)


class status:
    @staticmethod
    def progress_bar(iterable, **kwargs):
        return iterable
