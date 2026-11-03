"""Collection of plot functions."""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from physprog import classfunctions as cf

# Green (best) to red (worst), one per preference region.
REGION_COLORS = ("#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027")

DIRECTION_LABELS = {
    cf.SmallerBetter: "smaller is better",
    cf.LargerBetter: "larger is better",
    cf.ValueIsBetter: "value is better",
    cf.RangeIsBetter: "range is better",
}


def _finish(fig, fname, show):
    """Save and/or show a figure."""
    if fname:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def _axis_label(func, fallback="Dependent variable"):
    label = func.name or fallback
    if func.units:
        label += f" ({func.units})"
    return label


def _plot_range(func, extra=(), pad=0.15):
    """Return (lo, hi) spanning all bounds and ``extra`` values, padded."""
    values = [*func.bounds, *extra]
    lo, hi = min(values), max(values)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _one_sided_spans(bounds, best, worst):
    """Return (left, right, region) spans for bounds ordered best to worst."""
    edges = [best, *bounds, worst]
    return [(*sorted((edges[i], edges[i + 1])), i) for i in range(len(REGION_COLORS))]


def region_spans(func, lo, hi):
    """Return the (left, right, region) spans of ``func`` within [lo, hi]."""
    if isinstance(func, cf.TwoSidedFunction):
        low, high = func.best_region
        lower = _one_sided_spans(func.lower.bounds, low, lo)[1:]
        upper = _one_sided_spans(func.upper.bounds, high, hi)[1:]
        return [*lower, (low, high, cf.AWESOME), *upper]
    if func.direction < 0:
        return _one_sided_spans(func.bounds, hi, lo)
    return _one_sided_spans(func.bounds, lo, hi)


def region_legend_handles():
    """Legend handles for the six preference regions."""
    return [
        Patch(facecolor=color, alpha=0.5, label=name)
        for color, name in zip(REGION_COLORS, cf.REGION_NAMES, strict=True)
    ]


def plot_class_function(
    func,
    *,
    ax=None,
    label_regions=True,
    markers=None,
    legend=True,
    fname=None,
    show=None,
):
    """
    Plot one soft class function with its preference regions shaded.

    Parameters
    ----------
    func : SmoothClassFunction
        A class function whose splines have been built (one- or two-sided).
    ax : matplotlib Axes, optional
        Axes to draw into. A new figure is made if omitted.
    label_regions : bool
        Write region names inside the shaded bands.
    markers : dict, optional
        ``{label: value}`` points to mark on the curve, e.g. initial/final
        designs. Labels containing "initial" are drawn hollow.
    legend : bool
        Add a legend for the markers.
    fname : str, optional
        Save the figure to this file.
    show : bool, optional
        Call ``plt.show()``. Defaults to True when no ``fname`` and no ``ax``
        are given.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    if show is None:
        show = own_fig and not fname

    markers = markers or {}
    lo, hi = _plot_range(func, extra=markers.values())
    x = np.linspace(lo, hi, 600)
    y = np.array([func.evaluate(xi) for xi in x])
    ymax = func.gis[-1] * 8

    for left, right, i in region_spans(func, lo, hi):
        ax.axvspan(left, right, color=REGION_COLORS[i], alpha=0.35, lw=0)
        if label_regions:
            # Wide bands get horizontal text; narrow ones get vertical text.
            wide = (right - left) / (hi - lo) > 0.12
            ax.text(
                (left + right) / 2,
                0.97,
                cf.REGION_NAMES[i].replace(" ", "\n") if wide else cf.REGION_NAMES[i],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                rotation=0 if wide else 90,
                fontsize=7,
                color="0.25",
            )

    ax.plot(x, y, color="k", lw=2)
    bound_values = [func.evaluate(b) for b in func.bounds]
    ax.plot(func.bounds, bound_values, "o", color="k", ms=5, zorder=3)
    if isinstance(func, cf.ValueIsBetter):
        ax.axvline(func.target, color="k", ls="--", lw=1)
        ax.text(
            func.target,
            0.03,
            " target",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
            color="0.25",
        )
    for marker_label, value in markers.items():
        hollow = "initial" in marker_label.lower()
        ax.plot(
            value,
            min(func.evaluate(value), ymax),
            "o",
            ms=9,
            mec="#2b5fa8",
            mfc="white" if hollow else "#2b5fa8",
            mew=2,
            zorder=4,
            label=marker_label,
        )

    ax.set_yscale("log")
    ax.set_ylim(func.gis[0] / 3, ymax)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(_axis_label(func))
    ax.set_ylabel("Class function value")
    direction = DIRECTION_LABELS.get(type(func), type(func).__name__)
    title = f"{func.name}: {direction}" if func.name else direction.capitalize()
    ax.set_title(title)
    ax.grid(True, which="major", color="0.85", lw=0.6)
    if markers and legend:
        ax.legend(loc="lower right", fontsize=8)

    if own_fig:
        _finish(fig, fname, show)
    return ax


def plot_class_functions(
    preferences, initials=None, finals=None, *, ncols=2, fname=None, show=None
):
    """
    Plot every soft class function in ``preferences`` in a grid.

    If ``initials``/``finals`` (dicts of dependent-variable values keyed by
    name) are given, the before/after design is marked on each curve.
    """
    soft = {
        name: func
        for name, func in preferences.items()
        if isinstance(func, cf.SmoothClassFunction)
    }
    nrows = math.ceil(len(soft) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False
    )
    for ax, (name, func) in zip(axes.flat, soft.items(), strict=False):
        if func.name is None:
            func.name = name
        markers = {}
        if initials is not None:
            markers["Initial"] = initials[name]
        if finals is not None:
            markers["Final"] = finals[name]
        plot_class_function(
            func, ax=ax, label_regions=False, markers=markers, legend=False
        )
    for ax in axes.flat[len(soft) :]:
        ax.set_visible(False)

    marker_handles = axes.flat[0].get_legend_handles_labels()[0]
    fig.legend(
        handles=[*region_legend_handles(), *marker_handles],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    if show is None:
        show = not fname
    _finish(fig, fname, show)
    return fig


def plot_optimization_results(preferences, initials, finals, *, fname=None, show=None):
    """
    Show how each dependent variable changed in terms of preference.

    ``initials`` and ``finals`` are either dicts keyed by dependent-variable
    name, or sequences in the same order as ``preferences``.
    """
    if not isinstance(initials, dict):
        initials = dict(zip(preferences, initials, strict=False))
    if not isinstance(finals, dict):
        finals = dict(zip(preferences, finals, strict=False))

    names = list(preferences)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.5))
    # Columns from worst (left) to best (right); unacceptable maps to 0.
    for i, color in enumerate(reversed(REGION_COLORS[:-1])):
        ax.axvspan(i / 5, (i + 1) / 5, color=color, alpha=0.35, lw=0)

    for row, name in enumerate(names):
        pref = preferences[name]
        y = len(names) - row - 1
        start = pref.acceptability(initials[name])
        end = pref.acceptability(finals[name])
        if not math.isclose(start, end):
            ax.annotate(
                "",
                xy=(end, y),
                xytext=(start, y),
                arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.5},
            )
        ax.plot(start, y, "o", ms=9, mfc="white", mec="k", mew=1.5)
        ax.plot(end, y, "o", ms=9, color="k")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xticks([(i + 0.5) / 5 for i in range(5)])
    ax.set_xticklabels(
        [n.replace(" ", "\n") for n in reversed(cf.REGION_NAMES[:-1])], fontsize=8
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([_axis_label(preferences[n], n) for n in reversed(names)])
    ax.set_title("Optimization results (hollow: initial, filled: final)")
    fig.tight_layout()

    if show is None:
        show = not fname
    _finish(fig, fname, show)
    return fig
