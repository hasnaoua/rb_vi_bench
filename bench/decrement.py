"""Relative error decrement ``(e(n) - e(n+1)) / e(n)``, per dataset, all methods at once.

The precision curves say how much error remains. These say what the *next generator buys
you* -- which is the quantity that decides where to stop, and the one that separates
methods whose error curves look nearly identical.

The value plotted is the **fraction of the remaining error removed** by one more
generator, ``(e(n) - e(n+1)) / e(n)``. It is *positive* wherever the method improves, and
0 where the extra generator bought nothing.

Relative, not absolute. An absolute decrement ``e(n+1) - e(n)`` is dominated by wherever
the error happens to be large, so early iterations swamp late ones and two methods sitting
at different error levels cannot be compared at all -- a method with twice the error looks
twice as good per step. Dividing by ``e(n)`` asks the scale-free question instead: *what
share of what is left does this generator remove?* That is comparable across methods,
across cardinalities, and across datasets whose errors differ by decades.

The y-axis is **linear**, like every other axis in the benchmark: the plotted value is
the fraction itself, untransformed. Fractions span several orders of magnitude here, so
the small ones sit near the baseline -- read those off the CSVs. Nothing is dropped,
which matters because both 0 (the generator bought nothing) and negative values (the
method went backwards, as NMF does when refitted from scratch at each R) are results.

Two x-axes, from two different modes:

* **vs cardinality** -- ``(e(R) - e(R+1)) / e(R)`` from matched-cardinality rows. Needs
  *consecutive* R to mean "one more generator", so it is read from a sweep run with
  ``--cardinalities 1 2 3 ... N``. Gaps in R are skipped rather than divided through,
  because ``e(R+4) - e(R)`` is not a per-generator decrement.
* **vs tolerance** -- ``(e(eps_k) - e(eps_{k+1})) / e(eps_k)`` over the grid, ordered from
  loose to tight. This answers a different question: what tightening the *request* buys,
  which is the interface [BEE20] §5 argues for. The two are not interchangeable, since
  a method's R is its own response to eps.

**One incommensurability is carried over and must not be read past.** Tolerance-axis
figures put every method on a shared eps axis, but eps does not mean the same thing for
all of them: ADG's is a per-snapshot relative bound on ``S_norm``, CPG/mCPG's is one
shared absolute threshold ``eps * max_q ||theta_q||``. The cardinality axis has no such
problem -- R is R -- which is why it is the primary figure of the two.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import _paths  # noqa: F401  -- forces Agg before pyplot

import matplotlib.pyplot as plt

from . import cli, layout
from .plotting import FIGURE_EXCLUDED, discard, save, style_for
from .tabular import num as _num, rows_by_dataset_and_method


def decrements_vs_cardinality(points: list[tuple[float, float]]) -> tuple[list, list]:
    """``(e(R) - e(R+1)) / e(R)`` for consecutive R only.

    A gap in R is skipped, not interpolated: ``e(R+4) - e(R)`` is the value of four
    generators, and dividing by the gap would fabricate a per-generator figure the data
    does not contain.
    """
    xs, ys = [], []
    for (r0, e0), (r1, e1) in zip(points, points[1:]):
        if abs(r1 - r0 - 1.0) > 1e-9:
            continue
        if math.isnan(e0) or math.isnan(e1):
            continue
        if e0 <= 0:
            continue          # nothing left to remove; the fraction is undefined
        xs.append(r1)
        ys.append((e0 - e1) / e0)
    return xs, ys


def decrements_vs_tolerance(points: list[tuple[float, float]]) -> tuple[list, list]:
    """``(e(eps_k) - e(eps_{k+1})) / e(eps_k)`` walking the grid from loose to tight."""
    xs, ys = [], []
    for (t0, e0), (t1, e1) in zip(points, points[1:]):
        if math.isnan(e0) or math.isnan(e1):
            continue
        if e0 <= 0:
            continue
        xs.append(t1)
        ys.append((e0 - e1) / e0)
    return xs, ys


def _draw(ax, series, xlabel, title, xscale, mode="cardinality"):
    everything = []
    series = {m: v for m, v in series.items() if m not in FIGURE_EXCLUDED}
    for method, (xs, ys) in series.items():
        if not xs:
            continue
        style = style_for(method)
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], ls=style["ls"],
                label=style["label"], ms=4, lw=1.3, alpha=0.9)
        everything.extend(ys)
    if not everything:
        return 0
    ax.axhline(0.0, color="#444444", lw=0.8, ls="-", alpha=0.6)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\left(e(n)-e(n{+}1)\right)/e(n)$   (train)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    return 1


#: The decrement is taken on the TRAINING error, not the test error.
#:
#: This is the quantity a greedy actually drives down, and for a nested cone it is
#: monotone by construction -- so its decrement is signal end to end. The test error
#: plateaus once the cone stops generalizing further, after which differencing it yields
#: mostly round-off noise about zero and the figure says nothing about the algorithms.
#: (Generalization is what the precision figures are for.) A visible consequence worth
#: reading rather than smoothing away: NMF is refitted from scratch at every R and is not
#: hierarchical, so its curve is genuinely non-monotone here -- positive decrements are a
#: real property of the method, exactly the drawback [BEE20] §5 raises against it.
ERROR_COLUMN = "train_max_rel_err"


#: The two axes, as data rather than as two near-identical functions.
#:
#: They differ in which column indexes the x axis, which direction it runs, which
#: decrement rule applies, and the strings the reader sees -- and in nothing else.
#: Written as two functions those differences sat inside forty lines of identical figure
#: scaffolding, where a change to the scaffolding had to be made twice and a divergence
#: between the copies would surface only as a figure that looked subtly unlike its twin.
#:
#: The captions are not decoration. Neither axis means what it appears to mean at a
#: glance: on the R axis a zero decrement is a generator that bought nothing rather than
#: a missing point, and on the epsilon axis the tolerances are *not commensurable across
#: methods* -- ADG's is per-snapshot, CPG's and mCPG's is a shared absolute threshold --
#: so the curves may not be read against each other horizontally.
AXES = {
    "cardinality": dict(
        x_column="R",
        reverse=False,
        invert_axis=False,
        decrements=decrements_vs_cardinality,
        xlabel="cardinality $R$",
        title="marginal decrement per added generator",
        caption=("fraction of the remaining error removed by one more generator;  "
                 "0 = it bought nothing"),
        stem="vs_cardinality",
    ),
    "tolerance": dict(
        x_column="delta",
        # Loose to tight, i.e. decreasing epsilon.
        reverse=True,
        invert_axis=True,
        decrements=decrements_vs_tolerance,
        xlabel=r"tolerance $\varepsilon$  (tightening $\rightarrow$)",
        title="decrement per tolerance step",
        caption=("epsilon is NOT commensurable across methods: per-snapshot for ADG, "
                 "shared absolute for CPG/mCPG"),
        stem="vs_tolerance",
    ),
}


def figure_for_axis(mode: str, dataset, rows_by_method, out: Path) -> Path | None:
    """One decrement figure, for whichever of the two axes ``mode`` names."""
    spec = AXES[mode]
    series = {}
    for method, rows in rows_by_method.items():
        pts = sorted({_num(r, spec["x_column"]): _num(r, ERROR_COLUMN) for r in rows}.items(),
                     reverse=spec["reverse"])
        series[method] = spec["decrements"](pts)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    if not _draw(ax, series, spec["xlabel"], f"{dataset} — {spec['title']}", "linear",
                 mode=mode):
        discard(fig)
        return None
    if spec["invert_axis"]:
        ax.invert_xaxis()
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
    fig.text(0.5, -0.02, spec["caption"], ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    return save(fig, layout.ensure(layout.decrement_dir(out, dataset)) / f"{spec['stem']}.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="marginal decrement figures")
    p.add_argument("--cardinality-results", type=Path, default=_paths.SWEEP_DENSE,
                   help="grid with CONSECUTIVE --cardinalities (for the R axis)")
    p.add_argument("--tolerance-results", type=Path, default=_paths.RESULTS,
                   help="grid with a tolerance sweep (for the epsilon axis)")
    cli.add_out(p, _paths.RESULTS / "figures", what="decrement PNGs")
    args = p.parse_args(argv)

    written = []

    card = args.cardinality_results / "grid.csv"
    if card.is_file():
        for dataset, by_method in rows_by_dataset_and_method(card, "cardinality").items():
            path = figure_for_axis("cardinality", dataset, by_method, args.out)
            if path:
                written.append(path)
    else:
        print(f"[skip] {card} not found; no cardinality-axis figures")

    tol = args.tolerance_results / "grid.csv"
    if tol.is_file():
        for dataset, by_method in rows_by_dataset_and_method(tol, "tolerance").items():
            path = figure_for_axis("tolerance", dataset, by_method, args.out)
            if path:
                written.append(path)
    else:
        print(f"[skip] {tol} not found; no tolerance-axis figures")

    for path in written:
        print(path)
    print(f"\n{len(written)} decrement figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
