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

A **symlog** y-axis is used because the fractions span several orders of magnitude *and*
must accommodate exact 0 and the occasional negative excursion (a method going backwards);
a plain log axis would silently drop exactly the points worth seeing.

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

from . import layout
from .figures import FIGURE_EXCLUDED, STYLE
from .tabular import num as _num, rows_by_dataset_and_method

RESULTS = _paths.ROOT / "results"


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


def _symlog_threshold(values) -> float:
    """Linear-region width for the symlog axis.

    Set **relative to the largest decrement present**, not to the smallest. Anchoring it
    to the smallest non-zero magnitude puts the linear band at the float-noise floor
    (~1e-16), which leaves the entire axis logarithmic over thirteen decades and turns
    every round-off wobble on a plateaued curve into a full-height excursion -- the
    figure becomes an unreadable comb.

    Two decades below the largest fraction. That band is deliberately wide, because
    much of the combing here is **real data** rather than noise and still should not
    dominate the axis: ADG admits every snapshot tied at ``theta_max`` as one batch, so
    truncating at an intermediate R can add a generator that changes nothing (an exact
    zero) followed by one that drops the error sharply, and NMF is refitted from scratch
    at every R. Collapsing everything under 1% of the largest step into a flat band keeps
    those alternations visible as texture near zero while letting the decade-scale
    structure be read.
    """
    mags = [abs(v) for v in values if v != 0 and not math.isnan(v)]
    if not mags:
        return 1e-12
    return max(mags) * 1e-2


def _draw(ax, series, xlabel, title, xscale, mode="cardinality"):
    everything = []
    series = {m: v for m, v in series.items() if m not in FIGURE_EXCLUDED}
    for method, (xs, ys) in series.items():
        if not xs:
            continue
        style = STYLE.get(method, dict(color="black", marker=".", ls="-", label=method))
        ax.plot(xs, ys, color=style["color"], marker=style["marker"], ls=style["ls"],
                label=style["label"], ms=4, lw=1.3, alpha=0.9)
        everything.extend(ys)
    if not everything:
        return 0
    ax.axhline(0.0, color="#444444", lw=0.8, ls="-", alpha=0.6)
    ax.set_yscale("symlog", linthresh=_symlog_threshold(everything))
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


def figure_vs_cardinality(dataset, rows_by_method, out: Path) -> Path | None:
    series = {}
    err_col = ERROR_COLUMN
    for method, rows in rows_by_method.items():
        pts = sorted({_num(r, "R"): _num(r, err_col) for r in rows}.items())
        series[method] = decrements_vs_cardinality(pts)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    if not _draw(ax, series, "cardinality $R$",
                 f"{dataset} — marginal decrement per added generator", "linear",
                 mode="cardinality"):
        plt.close(fig)
        return None
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
    fig.text(0.5, -0.02,
             "fraction of the remaining error removed by one more generator;  "
             "0 = it bought nothing",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    path = layout.ensure(layout.decrement_dir(out, dataset)) / "vs_cardinality.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_vs_tolerance(dataset, rows_by_method, out: Path) -> Path | None:
    series = {}
    err_col = ERROR_COLUMN
    for method, rows in rows_by_method.items():
        # Loose to tight, i.e. decreasing epsilon.
        pts = sorted({_num(r, "delta"): _num(r, err_col) for r in rows}.items(),
                     reverse=True)
        series[method] = decrements_vs_tolerance(pts)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    if not _draw(ax, series, r"tolerance $\varepsilon$  (tightening $\rightarrow$)",
                 f"{dataset} — decrement per tolerance step", "log", mode="tolerance"):
        plt.close(fig)
        return None
    ax.invert_xaxis()
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
    fig.text(0.5, -0.02,
             "epsilon is NOT commensurable across methods: per-snapshot for ADG, "
             "shared absolute for CPG/mCPG",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    path = layout.ensure(layout.decrement_dir(out, dataset)) / "vs_tolerance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="marginal decrement figures")
    p.add_argument("--cardinality-results", type=Path, default=RESULTS / "sweep_dense",
                   help="grid with CONSECUTIVE --cardinalities (for the R axis)")
    p.add_argument("--tolerance-results", type=Path, default=RESULTS,
                   help="grid with a tolerance sweep (for the epsilon axis)")
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    args = p.parse_args(argv)

    written = []

    card = args.cardinality_results / "grid.csv"
    if card.is_file():
        for dataset, by_method in rows_by_dataset_and_method(card, "cardinality").items():
            path = figure_vs_cardinality(dataset, by_method, args.out)
            if path:
                written.append(path)
    else:
        print(f"[skip] {card} not found; no cardinality-axis figures")

    tol = args.tolerance_results / "grid.csv"
    if tol.is_file():
        for dataset, by_method in rows_by_dataset_and_method(tol, "tolerance").items():
            path = figure_vs_tolerance(dataset, by_method, args.out)
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
