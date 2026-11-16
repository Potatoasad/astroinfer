"""Shared plotting style for the astroinfer tutorials.

Plain matplotlib (no LaTeX install needed). Import it at the top of a notebook:

    import _style as style
    style.apply()

Colors have fixed roles across every tutorial, so a reader learns them once:

    style.POSTERIOR  blue     posterior draws / the thing we inferred
    style.PRIOR      orange   prior draws / the population before data
    style.DATA       aqua     observed data
    style.EXTRA      violet   a fourth series (e.g. "without selection")
    style.TRUTH      navy     true values, always a dashed navy line
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# A colorblind-checked categorical order; slots are never cycled.
# BLUE is the astroinfer brand blue (assets/README.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#418FC5", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, VIOLET]

POSTERIOR, PRIOR, DATA, EXTRA = BLUE, ORANGE, AQUA, VIOLET
INK, INK_2, MUTED, GRID, SURFACE = "#17243D", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
TRUTH = INK


def apply():
    """Set the tutorial look: light surface, recessive grid, thin marks."""
    mpl.rcParams.update({
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "mathtext.fontset": "cm",
    })


def density(ax, samples, *, bins=50, color=POSTERIOR, label=None, range=None, alpha=0.25):
    """A filled, outlined density histogram (the default way to show 1-D draws)."""
    samples = np.asarray(samples).ravel()
    hist, edges = np.histogram(samples, bins=bins, range=range, density=True)
    ax.stairs(hist, edges, fill=True, color=color, alpha=alpha, linewidth=0)
    ax.stairs(hist, edges, color=color, linewidth=1.6, label=label)
    return ax


def truth(ax, value, *, label="truth", orientation="vertical"):
    """Mark a true value with a dashed ink line."""
    line = ax.axvline if orientation == "vertical" else ax.axhline
    line(float(value), color=TRUTH, linestyle="--", linewidth=1.4, label=label)
    return ax


def corner(sets, columns, *, labels=None, names=None, colors=None, truths=None,
           bins=40, ranges=None, size=2.3, max_points=3000):
    """A small corner plot: 1-D densities on the diagonal, scatter below.

    sets:    list of {column: array} (dicts or DataFrames), one per series.
    columns: the columns to show, in order.
    labels:  axis labels per column (defaults to the column names).
    names:   legend name per set.
    truths:  {column: value} drawn as dashed ink lines.
    ranges:  {column: (lo, hi)}; defaults to the pooled 0.5-99.5% range.
    """
    n = len(columns)
    labels = labels or columns
    colors = colors or SERIES
    names = names or [None] * len(sets)
    rng = np.random.default_rng(0)
    arrays = [{c: np.asarray(s[c]).ravel() for c in columns} for s in sets]
    if ranges is None:
        ranges = {}
    lims = {}
    for c in columns:
        if c in ranges:
            lims[c] = ranges[c]
        else:
            pooled = np.concatenate([a[c] for a in arrays])
            lo, hi = np.percentile(pooled, [0.5, 99.5])
            pad = 0.05 * (hi - lo or 1.0)
            lims[c] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(n, n, figsize=(size * n, size * n), squeeze=False,
                             layout="constrained")
    for i, ci in enumerate(columns):
        for j, cj in enumerate(columns):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            for a, color, name in zip(arrays, colors, names):
                if i == j:
                    density(ax, a[ci], bins=bins, color=color, range=lims[ci], label=name)
                else:
                    k = len(a[ci])
                    idx = rng.choice(k, size=min(k, max_points), replace=False)
                    ax.scatter(a[cj][idx], a[ci][idx], s=4, alpha=0.35, color=color,
                               linewidths=0, rasterized=True)
            if truths is not None:
                if ci in truths and i == j:
                    truth(ax, truths[ci], label=None)
                elif i != j and ci in truths and cj in truths:
                    ax.axvline(truths[cj], color=TRUTH, linestyle="--", linewidth=1.2)
                    ax.axhline(truths[ci], color=TRUTH, linestyle="--", linewidth=1.2)
                    ax.plot(truths[cj], truths[ci], "o", color=TRUTH, markersize=5)
            ax.set_xlim(lims[cj])
            if i != j:
                ax.set_ylim(lims[ci])
            else:
                ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            elif j != 0:
                ax.set_yticklabels([])
    if any(names):
        if truths is not None:
            axes[0, 0].plot([], [], color=TRUTH, linestyle="--", linewidth=1.4, label="truth")
        corner_legend(fig, axes)
    return fig, axes


def corner_legend(fig, axes):
    """(Re)draw a corner plot's legend from the labelled artists on its top-left panel.

    Call it again after adding entries, e.g.
    ``axes[0, 0].plot([], [], "*", label="MLE"); style.corner_legend(fig, axes)``.
    The legend is figure-level, in the empty upper-right triangle, so constrained
    layout does not squash the panels to make room for it.
    """
    for legend in list(fig.legends):
        legend.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        if axes.shape[0] > 1:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
        else:
            axes[0, 0].legend(loc="upper left")
