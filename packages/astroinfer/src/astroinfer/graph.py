"""Directed acyclic graphs of models and distributions.

    model_graph(model, *args, expand=False, **kwargs)   the DAG of a NumPyro model
    dist.graph()                                        the DAG of a ConditionalDistribution

Both return a ``Graph``: nodes, edges and groups (plates, nested parameter
types, mixture components). It draws itself as an SVG in notebooks, with no
dependency on graphviz; ``to_dot()`` gives graphviz DOT source and
``to_graphviz()`` a ``graphviz.Digraph`` when graphviz is installed.

Node kinds:
    latent         a random quantity
    observed       an observed sample site (shaded)
    condition      an input of a distribution (e.g. Lambda)
    given          a condition bound with ``.condition``
    fixed          a fixed field
    deterministic  a ``numpyro.deterministic`` site
    factor         a ``numpyro.factor`` / ``ai.factor`` term
"""
import re
from dataclasses import dataclass, field
from html import escape

from .display import RichDisplay, details_open

# ============================================================
# Graph data
# ============================================================


@dataclass
class Node:
    id: str
    label: str            # LaTeX, drawn as text (see _tex_text)
    kind: str = "latent"
    group: str | None = None


@dataclass
class Group:
    id: str
    label: str
    parent: str | None = None


@dataclass
class Graph(RichDisplay):
    """Nodes, edges ``(from, to)`` and nested groups of a DAG."""

    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)
    title: str = "Graph"

    def add_node(self, id, label, kind="latent", group=None):
        if id not in self.nodes:
            self.nodes[id] = Node(id, label, kind, group)
        return self.nodes[id]

    def add_edge(self, a, b):
        if a != b and (a, b) not in self.edges:
            self.edges.append((a, b))

    def add_group(self, id, label, parent=None):
        if id not in self.groups:
            self.groups[id] = Group(id, label, parent)
        return self.groups[id]

    def __repr__(self):
        return (f"Graph({self.title!r}, nodes={list(self.nodes)}, "
                f"edges={self.edges})")

    # ========================================================
    # Export
    # ========================================================

    def to_dot(self) -> str:
        """Graphviz DOT source (graphviz is not needed to produce it)."""
        shape = {"latent": "ellipse", "observed": "ellipse", "condition": "box",
                 "given": "box", "fixed": "box", "deterministic": "ellipse",
                 "factor": "square"}
        lines = [f'digraph "{self.title}" {{', "  rankdir=TB;"]

        def node_line(n, indent):
            attrs = [f'label="{_tex_text(n.label)}"', f"shape={shape.get(n.kind, 'ellipse')}"]
            if n.kind == "observed":
                attrs.append("style=filled, fillcolor=lightgrey")
            if n.kind in ("fixed", "deterministic", "given"):
                attrs.append("style=dashed")
            if n.kind == "factor":
                attrs += ["style=filled", "fillcolor=black", "width=0.15", "fixedsize=true",
                          f'xlabel="{_tex_text(n.label)}"', 'label=""']
            return f'{indent}"{n.id}" [{", ".join(attrs)}];'

        def emit(group_id, indent):
            for g in self.groups.values():
                if g.parent == group_id:
                    lines.append(f'{indent}subgraph "cluster_{g.id}" {{')
                    lines.append(f'{indent}  label="{_tex_text(g.label)}";')
                    emit(g.id, indent + "  ")
                    lines.append(f"{indent}}}")
            for n in self.nodes.values():
                if n.group == group_id:
                    lines.append(node_line(n, indent))

        emit(None, "  ")
        lines += [f'  "{a}" -> "{b}";' for a, b in self.edges]
        lines.append("}")
        return "\n".join(lines)

    def to_graphviz(self):
        """A ``graphviz.Source`` of ``to_dot()``; needs the graphviz package."""
        try:
            import graphviz
        except ImportError as exc:
            raise ImportError(
                "to_graphviz needs the graphviz package (and its 'dot' program); "
                "to_svg() and the notebook display do not."
            ) from exc
        return graphviz.Source(self.to_dot())

    def to_svg(self) -> str:
        return _render_svg(self)

    # ========================================================
    # Display
    # ========================================================

    def _repr_html_(self):
        return (
            "<details" + details_open() + ' style="font-family:inherit;line-height:1.5">'
            '<summary style="cursor:pointer;padding:0.25em 0"><strong>'
            + escape(self.title) + "</strong></summary>"
            '<div style="padding:0.4em 0 0.4em 1.2em;overflow-x:auto">'
            + self.to_svg() + "</div></details>"
        )


# ============================================================
# LaTeX labels as plain text
# ============================================================

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "varpi": "ϖ", "varphi": "φ", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω", "star": "⋆", "mid": "|", "ell": "ℓ", "infty": "∞",
    "ldots": "…", "cdots": "…", "cos": "cos", "sin": "sin", "log": "log", "ln": "ln", "exp": "exp",
}
_SUB = str.maketrans("0123456789+-=()aehijklmnoprstuvx",
                     "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _tex_text(tex: str) -> str:
    """Simple LaTeX (``m_1``, ``\\hat\\chi``, ``\\log_{10}L``) as unicode text."""
    s = str(tex).strip("$ ")
    s = re.sub(r"\\(?:mathrm|mathcal|mathbf|text|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:hat|widehat)\s*\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\(?:hat|widehat)\s*(\\?[A-Za-z]+|.)",
               lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1), m.group(1)), s)
    def script(match, table, mark):
        body = match.group(1) or match.group(2)
        out = body.translate(table)
        if any(o == c and c.isalnum() for o, c in zip(out, body)):
            return mark + body          # no unicode form for some character
        return out

    s = re.sub(r"_\{([^{}]*)\}|_(.)", lambda m: script(m, _SUB, "_"), s)
    s = re.sub(r"\^\{([^{}]*)\}|\^(.)", lambda m: script(m, _SUP, "^"), s)
    s = re.sub(r"\\[,;:! ]", " ", s).replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# Layout and SVG rendering
# ============================================================

_CHAR_W, _NODE_H, _GAP_X, _GAP_Y, _PAD = 7.2, 30, 26, 56, 12


def _layers(graph):
    """Longest-path layering, then barycenter ordering within layers."""
    ids = list(graph.nodes)
    preds = {i: [a for a, b in graph.edges if b == i and a in graph.nodes] for i in ids}
    layer = {}

    def depth(i, seen=()):
        if i not in layer:
            layer[i] = 0 if i in seen else max(
                (depth(p, seen + (i,)) + 1 for p in preds[i]), default=0)
        return layer[i]

    for i in ids:
        depth(i)
    rows = [[] for _ in range(max(layer.values(), default=-1) + 1)]
    for i in ids:
        rows[layer[i]].append(i)

    def chain(i):  # group ancestry, used to keep group members together
        out, g = [], graph.nodes[i].group
        while g is not None:
            out.insert(0, g)
            g = graph.groups[g].parent if g in graph.groups else None
        return tuple(out)

    first_seen = {}
    for i in ids:
        for g in chain(i):
            first_seen.setdefault(g, len(first_seen))
    for _ in range(2):
        pos = {i: k for row in rows for k, i in enumerate(row)}
        for row in rows[1:]:
            def key(i):
                ps = [pos[p] for p in preds[i] if p in pos]
                return (tuple(first_seen[g] for g in chain(i)),
                        sum(ps) / len(ps) if ps else pos[i])
            row.sort(key=key)
        rows[0].sort(key=lambda i: tuple(first_seen[g] for g in chain(i)))
    return rows, chain


def _render_svg(graph):
    if not graph.nodes:
        return "<em>empty graph</em>"
    rows, chain = _layers(graph)
    width = {i: max(46, _CHAR_W * len(_tex_text(n.label)) + 22)
             for i, n in graph.nodes.items()}
    for i, n in graph.nodes.items():
        if n.kind == "factor":
            width[i] = 14
    depth = {g: 0 for g in graph.groups}
    for i in graph.nodes:
        for k, g in enumerate(chain(i)):
            depth[g] = max(depth.get(g, 0), len(chain(i)) - k)
    row_w = [sum(width[i] for i in r) + _GAP_X * (len(r) - 1) for r in rows]
    total_w = max(row_w) + 2 * _PAD + 40
    xy = {}
    for y, (row, rw) in enumerate(zip(rows, row_w)):
        x = (total_w - rw) / 2
        for i in row:
            xy[i] = (x + width[i] / 2, _PAD + 24 + y * (_NODE_H + _GAP_Y) + _NODE_H / 2)
            x += width[i] + _GAP_X
    total_h = _PAD + 24 + len(rows) * (_NODE_H + _GAP_Y) - _GAP_Y + _PAD + 8
    parts = []

    # groups: bounding boxes of their members, outer groups padded more
    for g in sorted(graph.groups.values(), key=lambda g: -depth.get(g.id, 0)):
        members = [i for i in graph.nodes if g.id in chain(i)]
        if not members:
            continue
        pad = 8 + 7 * depth.get(g.id, 0)
        x0 = min(xy[i][0] - width[i] / 2 for i in members) - pad
        x1 = max(xy[i][0] + width[i] / 2 for i in members) + pad
        y0 = min(xy[i][1] for i in members) - _NODE_H / 2 - pad - 12
        y1 = max(xy[i][1] for i in members) + _NODE_H / 2 + pad
        parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" '
            'rx="8" fill="#418FC5" fill-opacity="0.06" stroke="#418FC5" '
            'stroke-opacity="0.55"/>'
            f'<text x="{x0 + 7:.1f}" y="{y0 + 13:.1f}" font-size="11" fill="currentColor" '
            f'opacity="0.75">{escape(_tex_text(g.label))}</text>'
        )

    # edges: straight lines with arrowheads, ending at the node boundary
    for a, b in graph.edges:
        if a not in xy or b not in xy:
            continue
        (x0, y0), (x1, y1) = xy[a], xy[b]
        h = _NODE_H / 2 if graph.nodes[b].kind != "factor" else 7
        h0 = _NODE_H / 2 if graph.nodes[a].kind != "factor" else 7
        sy = 1 if y1 >= y0 else -1
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0 + sy * h0:.1f}" x2="{x1:.1f}" '
            f'y2="{y1 - sy * (h + 2):.1f}" stroke="currentColor" stroke-opacity="0.7" '
            'stroke-width="1.2" marker-end="url(#ai-arrow)"/>'
        )

    # nodes
    for i, n in graph.nodes.items():
        x, y = xy[i]
        w = width[i]
        text = escape(_tex_text(n.label))
        if n.kind == "factor":
            parts.append(
                f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" '
                'fill="currentColor"/>'
                f'<text x="{x + 12:.1f}" y="{y + 4:.1f}" font-size="12" '
                f'fill="currentColor">{text}</text>'
            )
            continue
        style = {
            "latent": 'fill="none" stroke="currentColor"',
            "observed": 'fill="#C9D2DE" fill-opacity="0.55" stroke="currentColor"',
            "condition": 'fill="#E0A348" fill-opacity="0.18" stroke="#E0A348"',
            "given": 'fill="none" stroke="#E0A348" stroke-dasharray="4 3"',
            "fixed": 'fill="none" stroke="currentColor" stroke-opacity="0.6" '
                     'stroke-dasharray="4 3"',
            "deterministic": 'fill="none" stroke="currentColor" stroke-dasharray="2 2"',
        }.get(n.kind, 'fill="none" stroke="currentColor"')
        rx = _NODE_H / 2 if n.kind in ("latent", "observed", "deterministic") else 4
        parts.append(
            f'<rect x="{x - w / 2:.1f}" y="{y - _NODE_H / 2:.1f}" width="{w:.1f}" '
            f'height="{_NODE_H}" rx="{rx}" {style} stroke-width="1.2"/>'
            f'<text x="{x:.1f}" y="{y + 4.5:.1f}" font-size="13" text-anchor="middle" '
            f'fill="currentColor">{text}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" '
        f'height="{total_h:.0f}" viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
        'style="font-family:inherit;color:inherit;max-width:100%">'
        '<defs><marker id="ai-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="currentColor" fill-opacity="0.75"/>'
        "</marker></defs>" + "".join(parts) + "</svg>"
    )


# ============================================================
# Graphs of distributions
# ============================================================


def _type_tex(t, default):
    return getattr(t, "_latex", None) or getattr(t, "__name__", None) or default


def _schema_items(schema, symbol):
    """``[(key, label)]`` for the arguments of a schema."""
    if schema.kind == "none":
        return []
    if schema.kind == "single":
        t = schema.types[0]
        return [("0", _type_tex(t, symbol) if isinstance(t, type) else symbol)]
    if schema.kind == "tuple":
        return [(str(k), _type_tex(t, f"{symbol}_{k}")) for k, t in enumerate(schema.types)]
    return [(name, name) for name in schema.names]


def condition_nodes(dist, graph, kind="condition", group=None):
    """Add one node per condition of ``dist``; return ``{key: node id}``."""
    ids = {}
    for key, label in _schema_items(dist._condition_schema(), dist._condition_symbol):
        ids[key] = graph.add_node(f"cond:{key}", label, kind, group).id
    return ids


def distribution_graph(dist):
    """Default graph of a ConditionalDistribution: conditions -> outputs."""
    g = Graph(title=f"Graph of {getattr(dist, 'display_name', type(dist).__name__)}")
    conds = condition_nodes(dist, g)
    for key, label in _schema_items(dist._output_schema(), dist._output_symbol):
        out = g.add_node(f"out:{key}", label).id
        for c in conds.values():
            g.add_edge(c, out)
    return g


def structured_graph(dist):
    """Fields, fixed values, conditions and field dependencies."""
    from .distributions.structured import Block, CallableLeaf, ConditionalLeaf

    data_type = dist.data_type
    g = Graph(title=f"Graph of StructuredDistribution[{data_type.__name__}]")
    conds = condition_nodes(dist, g)
    specs = data_type.flat_field_specs()

    def group_of(path):
        parts = path.split(".")[:-1]
        parent, t = None, data_type
        for k in range(len(parts)):
            gid = ".".join(parts[: k + 1])
            t = t.nested_types().get(parts[k], t) if hasattr(t, "nested_types") else t
            g.add_group(gid, f"{parts[k]}: {getattr(t, '__name__', parts[k])}", parent)
            parent = gid
        return parent

    def label(path):
        spec = specs.get(path)
        return (spec.latex if spec is not None and spec.latex else path.rsplit(".", 1)[-1])

    for path in dist.order:
        record = dist.field_at(path)
        node = g.add_node(path, label(path), "latent", group_of(path))
        if isinstance(record, (CallableLeaf, Block)):
            for c in conds.values():
                g.add_edge(c, node.id)
        if isinstance(record, ConditionalLeaf):
            for dep in dist.dependencies(path)[0]:
                g.add_edge(dep, node.id)
    values = dist.fixed_values()
    for path in dist.fixed_paths():
        v = values.get(path)
        text = label(path)
        try:
            import numpy as np
            if np.size(v) == 1:
                text += f" = {float(np.asarray(v).ravel()[0]):g}"
        except (TypeError, ValueError):
            pass
        g.add_node(path, text, "fixed", group_of(path))
    return g


def mixture_graph(dist):
    """Conditions -> weights and components -> the mixed output."""
    g = Graph(title="Graph of MixtureDistribution")
    conds = condition_nodes(dist, g)
    out_symbol = _schema_items(dist._output_schema(), dist._output_symbol)
    out = g.add_node("out", out_symbol[0][1] if out_symbol else "x").id
    fn = dist.weights if dist.logits is None else dist.logits
    w = g.add_node("weights", "w(c)" if callable(fn) else "w", "latent").id
    g.add_edge(w, out)
    if callable(fn):
        for c in conds.values():
            g.add_edge(c, w)
    g.add_group("components", "components")
    for k, comp in enumerate(dist.components):
        node = g.add_node(f"component:{k}", f"p_{k + 1}", "latent", "components").id
        g.add_edge(node, out)
        if comp._condition_schema().kind != "none":
            for c in conds.values():
                g.add_edge(c, node)
    return g


def product_graph(dist):
    """Conditions -> each independent component."""
    g = Graph(title="Graph of ProductDistribution")
    conds = condition_nodes(dist, g)
    for name, comp in dist._items():
        items = _schema_items(comp._output_schema(), comp._output_symbol)
        node = g.add_node(f"out:{name}", items[0][1] if items else str(name)).id
        if comp._condition_schema().kind != "none":
            for c in conds.values():
                g.add_edge(c, node)
    return g


def bound_graph(dist):
    """The parent's graph, with the bound conditions marked as given."""
    g = dist.parent.graph()
    remaining = {key for key, _ in _schema_items(dist._condition_schema(), "")}
    parent_items = _schema_items(dist.parent._condition_schema(), "")
    for key, _ in parent_items:
        node = g.nodes.get(f"cond:{key}")
        if node is not None and (dist._condition_schema().kind != "mapping"
                                 or key not in remaining):
            node.kind = "given"
    g.title = g.title + " (conditioned)"
    return g


# ============================================================
# Graph of a NumPyro model
# ============================================================


def _name_tex(name):
    """A site or field name as LaTeX: Greek names become symbols (``alpha`` -> alpha)."""
    head, _, tail = name.rpartition(".")
    base, _, sub = tail.partition("_")
    sym = "\\" + base if base in _GREEK and len(base) > 1 else base
    return sym + (f"_{{{sub}}}" if sub else "")


def model_graph(model, *args, expand=False, **kwargs):
    """
    The DAG of a NumPyro model written with ``ai.sample``, ``ai.factor`` and
    plain ``numpyro`` sites.

    Dependencies come from ``numpyro.infer.inspect.get_model_relations``,
    which traces the model and follows which sites each log density depends
    on. By default every object drawn with ``ai.sample`` is one node (its
    leaf and block sites and fixed fields collapsed); ``expand=True`` shows
    one node per site, grouped by object. Plates are drawn as groups.
    """
    from numpyro import handlers
    from numpyro.infer.inspect import get_model_relations

    relations = get_model_relations(model, model_args=args, model_kwargs=kwargs)
    trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace(*args, **kwargs)
    observed = set(relations.get("observed", ()))
    g = Graph(title=f"Graph of {getattr(model, '__name__', 'model')}")

    roots = {}      # root name -> output type
    site_root = {}  # site name -> root name
    for name, site in trace.items():
        meta = (site.get("infer") or {}).get("astroinfer") if site["type"] == "sample" else None
        if meta:
            roots.setdefault(meta["root"], meta.get("type"))
            site_root[name] = meta["root"]
    for name, site in trace.items():  # fixed fields: deterministic "root.path" sites
        if site["type"] == "deterministic" and name not in site_root:
            head = next((r for r in roots if name.startswith(r + ".")), None)
            if head is not None:
                site_root[name] = head

    plates, letters = {}, {}  # site -> innermost plate; plate -> index letter
    for name, site in trace.items():
        for frame in site.get("cond_indep_stack", ()) or ():
            plates.setdefault(name, frame.name)
            if frame.name not in letters:
                letters[frame.name] = "ijkl"[len(letters) % 4]
                g.add_group(f"plate:{frame.name}",
                            rf"{letters[frame.name]} = 1\ldots {frame.size}\ ({frame.name})")

    def indexed(label, plate):
        """``label`` with the plate's index letter, e.g. theta -> theta_i."""
        if plate is None:
            return label
        i = letters[plate]
        m = re.fullmatch(r"(.*)_\{(.*)\}", label) or re.fullmatch(r"(.*)_(.)", label)
        return f"{m.group(1)}_{{{m.group(2)},{i}}}" if m else f"{{{label}}}_{i}"

    def root_label(root):
        t = roots.get(root)
        return getattr(t, "_latex", None) or _name_tex(root)

    def field_label(root, name):
        path = name[len(root) + 1:] if name.startswith(root + ".") else name
        t = roots.get(root)
        spec = t.flat_field_specs().get(path) if hasattr(t, "flat_field_specs") else None
        return spec.latex if spec is not None and spec.latex else _name_tex(path)

    def kind_of(name, site):
        if site["type"] == "deterministic":
            return "fixed" if name in site_root else "deterministic"
        if type(site["fn"]).__name__ == "Unit":
            return "factor"
        return "observed" if (name in observed or site.get("is_observed")) else "latent"

    node_of = {}
    for name, site in trace.items():
        if site["type"] not in ("sample", "deterministic"):
            continue
        kind = kind_of(name, site)
        plate = plates.get(name)
        plate_group = f"plate:{plate}" if plate else None
        root = site_root.get(name)
        if root is not None and not expand:
            node = g.add_node(f"root:{root}", indexed(root_label(root), plate), kind,
                              plate_group)
            if kind == "observed":
                node.kind = "observed"
            elif node.kind == "fixed" and kind == "latent":
                node.kind = "latent"
            node_of[name] = node.id
        elif root is not None:
            t = roots.get(root)
            type_name = t.__name__ if isinstance(t, type) else root
            g.add_group(f"root:{root}", f"{indexed(root_label(root), plate)}: {type_name}",
                        plate_group)
            label = indexed(field_label(root, name), plate)
            if kind == "fixed":
                value = site["value"]
                try:
                    import numpy as np
                    flat = np.unique(np.asarray(value))
                    if flat.size == 1:
                        label += f" = {float(flat[0]):g}"
                except (TypeError, ValueError):
                    pass
            node_of[name] = g.add_node(name, label, kind, f"root:{root}").id
        else:
            label = _name_tex(name) if kind != "factor" else name
            node_of[name] = g.add_node(name, indexed(label, plate) if kind != "factor"
                                       else label, kind, plate_group).id

    for name, deps in relations.get("sample_sample", {}).items():
        for dep in deps:
            if name in node_of and dep in node_of:
                g.add_edge(node_of[dep], node_of[name])
    return g
