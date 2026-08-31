"""What every figure module shares: the method palette, and how a figure is saved.

Three modules draw figures -- ``figures`` (metric vs cardinality), ``decrement``
(marginal gain per generator) and ``reconstruction`` (fields and profiles). They used
to reach into ``figures`` for the palette, which made a CLI entry point the de-facto
home of shared state: ``decrement`` and ``reconstruction`` both did
``from .figures import FIGURE_EXCLUDED, STYLE``, so importing either dragged in the
whole metric-panel machinery and its argparse. The palette lives here instead, and
``figures`` imports it like everyone else.

The reason a *shared* palette matters at all is that the figures are read side by side.
A method must keep its colour and marker across every panel of every figure, or a
reader comparing the precision plot against the decrement plot has to re-learn the
legend each time. So the mapping is one dict, in one place, keyed by the same method
handles ``--methods`` takes.
"""

from __future__ import annotations

from pathlib import Path

from . import _paths  # noqa: F401  -- forces the Agg backend before pyplot is imported

import matplotlib.pyplot as plt


#: Stable per-method styling, so a method keeps its colour across every figure.
#: Families share a hue: CPG blues, mCPG greens, ADG oranges, baselines grey/red.
STYLE: dict[str, dict] = {
    "cpg_bee20":   dict(color="#1f4e9c", marker="o", ls="-",  label="CPG [BEE20]"),
    "cpg_ndee22":  dict(color="#3a7bd5", marker="s", ls="-",  label="CPG [NDEE22]"),
    "cpg_greedy":  dict(color="#7fb2f0", marker="^", ls="--", label="CPG (greedy.core)"),
    "mcpg_ndee22": dict(color="#1b7f4f", marker="o", ls="-",  label="mCPG [NDEE22]"),
    "mcpg_greedy": dict(color="#5cc98d", marker="^", ls="--", label="mCPG (greedy.core)"),
    "adg":         dict(color="#e8760a", marker="D", ls="-",  label="ADG (batch normalized)"),
    "adg_raw":     dict(color="#f0b27a", marker="d", ls=":",  label="ADG (un-normalized)"),
    "adg_momentum": dict(color="#a04000", marker="*", ls="-.",
                         label="ADG (momentum stop)"),
    "nmf_s0":      dict(color="#c0392b", marker="v", ls="-",  label="NMF (seed 0)"),
    "nmf_s1":      dict(color="#d98880", marker="v", ls=":",  label="NMF (seed 1)"),
    "nmf_s2":      dict(color="#e6b0aa", marker="v", ls=":",  label="NMF (seed 2)"),
    "orthant":     dict(color="#6c3483", marker="P", ls="--", label=r"orthant $W^+$"),
    "pod_control": dict(color="#7f8c8d", marker="x", ls="--", label="POD (control)"),
}

#: The two reference methods, kept out of the shared-axis figures.
#:
#: Both are *references*, not competitors, and both sit orders of magnitude away from the
#: methods being compared -- the orthant because it is the widest admissible cone (90 deg
#: aperture, near-total excess), POD because its error falls to machine zero past the
#: numerical rank. Plotting either forces the shared axis to span their range and squeezes
#: the curves that matter into a thin band. Their numbers stay in ``grid.csv`` and
#: ``report.txt``, where a reader can consult them without paying for them visually.
FIGURE_EXCLUDED: frozenset[str] = frozenset({"orthant", "pod_control"})


def style_for(method: str) -> dict:
    """This method's plot kwargs, with a visible fallback for an unregistered one.

    Black-with-a-dot rather than a raised ``KeyError``: a method added to ``adapters``
    and not yet to ``STYLE`` should still show up on the figure -- unmistakably
    unstyled, so it gets a colour, but never silently dropped from a comparison.
    """
    return STYLE.get(method, dict(color="black", marker=".", ls="-", label=method))


def save(fig, path: Path, *, dpi: int = 150) -> Path:
    """Write a figure and close it, returning the path for the caller's manifest.

    Closing is the point. Every figure module loops over datasets and methods, and
    ``pyplot`` keeps a reference to every unclosed figure: a full ``--separate`` run opens
    several hundred, and matplotlib starts warning at 20 before the process simply grows.
    Pairing the write with the close in one call is what stops a future caller from
    forgetting the second half.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def discard(fig) -> None:
    """Close a figure that turned out to have nothing to draw."""
    plt.close(fig)


__all__ = ["FIGURE_EXCLUDED", "STYLE", "discard", "save", "style_for"]
