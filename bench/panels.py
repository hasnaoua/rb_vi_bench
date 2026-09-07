"""What a metric panel *is*, and how one is drawn.

Split out of ``figures`` for the same reason ``plotting`` was: that module is a CLI
entry point, and four separate concerns had accumulated behind its argparse -- the panel
specifications, the drawing engine, eight figure builders, and the command itself. With
the builders separated by two hundred lines of unrelated machinery, the fact that four
of them were near-duplicates was genuinely hard to see.

This module owns the first two: the panel tuples (which column, which label, which
scale, which caption), the axis rules that go with them, and ``_panel``, which turns one
of those specifications plus a series into one drawn axes. It knows nothing about files,
layout or figure composition -- ``figures`` does that, and imports from here.

Every name here is re-exported by ``figures`` so existing imports keep working.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from . import _paths  # noqa: F401  -- forces the Agg backend before pyplot is imported

import numpy as np

from .plotting import FIGURE_EXCLUDED, style_for
from .tabular import num as _num, rows_by_dataset_and_method


#: Cone-geometry panels: how much of ``span_+{all snapshots}`` a reduced cone captures,
#: how wide it opens, and how far it reaches outside. See ``metrics.cone_geometry``.
#: Both directions are shown, because neither implies the other: a cone can cover
#: ``K_full`` perfectly while extending far beyond it, or sit strictly inside while
#: missing most of it. One consequence to read correctly: for cones spanned by snapshots
#: ``excess`` is round-off (2.8e-17 to 5.5e-16) about a structural zero -- such a cone
#: cannot leave ``K_full`` -- so those curves sit flat on the axis at what is effectively
#: 0, and only mCPG's genuine excursion (up to 0.26) lifts off it.
#:
#: The last two panels answer different questions and must not be conflated. EXTENT
#: (``section_extent``) is how much space the cone encloses: cut it by a hyperplane common
#: to every method and take the mean width of the section over directions of the AMBIENT
#: space, as a fraction of the same width for ``K_full``. It rises monotonically in R --
#: ``K_R`` is a sub-cone of ``K_{R+1}``, so the section can only grow -- and 1 means "as
#: wide, on average over directions, as everything the snapshots generate". Above 1 means
#: the cone is wider than ``K_full``, which requires leaving it; mCPG and the orthant both
#: do. CONDITIONING (``aperture_mean_deg``) is a mean over edges. It says nothing about
#: enclosed space: a generator added strictly inside the cone leaves the region unchanged
#: while still moving the mean.
CONE_PANELS = (
    ("cover_mean_err",    r"mean residual, $K_{full}\to K_R$", "linear",
     "how much of the full cone is MISSED (too small)"),
    ("excess_mean_err",   r"mean residual, $K_R\to K_{full}$", "linear",
     "how much of the cone lies OUTSIDE (too large)"),
    ("cone_sym_err",      "two-sided discrepancy",             "linear",
     r"$\frac{1}{2}$(missed + excess) — 0 iff the cones coincide"),
    # The extent panel. Read it VERTICALLY -- between methods at one R -- because the
    # underlying volume changes dimension with R; the ratio makes the numbers
    # comparable, not the geometry.
    ("section_extent",    r"$w(S_R)\,/\,w(S_{full})$",          "linear",
     "EXTENT: mean width of the section, vs that of $K_{full}$"),
    ("aperture_mean_deg", "mean pairwise angle [deg]",         "linear",
     "conditioning (mean pairwise angle, NOT an extent)"),
)

#: Extra split figures, written only by ``--separate``. Kept out of ``PANELS`` so the
#: comparison layout stays a 2x2 grid.
EXTRA_SPLIT_PANELS = (
    ("test_max_rel_err_persnap", "max per-snapshot relative error", "linear",
     "precision, each snapshot vs ITS OWN norm"),
    # The training error on its own axes, under both normalizations. It appears in the
    # comparison panel only as a dotted overlay -- deliberately faint there, since the
    # subject of that figure is generalization -- but it is a quantity in its own right:
    # it is what every greedy actually minimizes, and it is monotone in R by construction
    # for a nested cone, so it is the curve that shows whether a method is converging at
    # all as opposed to converging *usefully*. See also ``figure_train_vs_test``.
    ("train_max_rel_err", "max relative projection error", "linear",
     "precision (TRAINING set)"),
    ("train_max_rel_err_persnap", "max per-snapshot relative error", "linear",
     "precision (TRAINING set), each snapshot vs ITS OWN norm"),
    # SOLVED error: the reduced saddle-point problem actually solved at each held-out
    # parameter, not the cone scored against snapshots. Empty on datasets that ship no
    # operator, load and obstacle, which is most of them -- see Dataset.supports_online.
    ("online_primal_mean_rel", "mean relative error in $u$", "linear",
     "SOLVED primal error (reduced saddle-point problem)"),
    ("online_dual_mean_rel", r"mean relative error in $\lambda$", "linear",
     "SOLVED dual error (reduced saddle-point problem)"),
)

#: The single column plotted on a log axis, and why it earns the exception.
#:
#: Every other panel is linear, because a log axis has no coordinate for zero and drops
#: such points with no marker -- which had silently removed whole series. ``gram_cond``
#: cannot hit that failure: a condition number is >= 1 by definition, and across both
#: grids its 2429 values have minimum exactly 1 and none <= 0. So nothing is lost here,
#: and what is gained is the panel itself. Conditioning spans 4.0e2 to 3.1e13 over
#: R=2..16 on Half-disks of Hertz -- eleven decades before the basis even goes singular --
#: and on a linear axis everything under about 1e12 is pressed flat against zero, leaving
#: a panel that shows one late excursion and nothing else. It is the one quantity in this
#: benchmark that is logarithmic by nature.
LOG_AXIS_EXCEPTIONS: frozenset[str] = frozenset({"gram_cond"})

#: File stem for each metric's standalone PNG. One source for BOTH split paths -- the
#: per-metric figures and the reference-baseline ones -- because they used to carry
#: separate copies and drifted: the solved-error panels were added to one and fell back to
#: raw column names in the other. A column absent here keeps its column name.
SPLIT_NAMES: dict[str, str] = {
    "test_max_rel_err": "precision",
    "test_max_rel_err_persnap": "precision_persnap",
    "train_max_rel_err": "precision_train",
    "train_max_rel_err_persnap": "precision_train_persnap",
    "gram_cond": "conditioning",
    "e_orth_mean": "orthogonality",
    "calls_total": "offline_cost",
    "cover_mean_err": "cone_missed",
    "excess_mean_err": "cone_excess",
    "cone_sym_err": "cone_two_sided",
    "section_extent": "cone_extent",
    "aperture_mean_deg": "aperture",
    "online_primal_mean_rel": "solved_primal",
    "online_dual_mean_rel": "solved_dual",
}

#: Values a column cannot meaningfully exceed, used to bound the AXIS rather than to
#: alter any datum. Nothing is dropped or transformed: points above the ceiling still
#: plot, they simply clip at the top of the axis, and the ceiling is drawn as a marked
#: line so a reader can see that is what happened.
#:
#: ``gram_cond`` is the case. Above ``1/eps`` the Gram matrix is numerically singular and
#: the returned condition number is round-off, not a measurement. On Half-disks of Hertz
#: the matrix crosses that line at R=17 for CPG and ADG, R=18 for NMF and R=19 for mCPG,
#: and the values beyond it wander between 1e16 and 1e19 with no order to them. Letting
#: them set the axis is what broke this panel: a single 4.59e19 sample at R=22 compressed
#: the entire meaningful range -- R=2..16, rising smoothly from 4.0e2 to 3.1e13 -- into a
#: flat line at zero, so the figure showed one noise spike and nothing else. The
#: conditioning worth reading is the conditioning before the basis goes singular.
NUMERICAL_CEILING: dict[str, float] = {
    "gram_cond": 1.0 / np.finfo(float).eps,
}

PANELS = (
    ("test_max_rel_err", "max relative projection error", "linear", "precision (test set)"),
    # The ONE log axis. See LOG_AXIS_EXCEPTIONS.
    ("gram_cond",        "Gram condition number",         "log",    "conditioning"),
    ("e_orth_mean",      "mean $e_{orth}$",               "linear", "orthogonality (Eq. 41)"),
    ("calls_total",      "constrained solver calls",      "linear", "offline cost"),
)


def error_column(series) -> tuple[str, str]:
    """Pick the error column a dataset can actually support.

    A source that ships no train/test split has ``test_max_rel_err`` ``nan`` throughout,
    so a test-error panel comes out blank. Fall back to the training error and say so in
    the title, rather than shipping an empty axes that reads like missing data.

    No registered dataset is in that state any more -- ``physics`` was the last one, and
    it now carries the 50/49 partition its archive ships. The fallback stays because the
    ``Dataset`` contract still permits a split-less source, and a blank precision panel
    is exactly the kind of failure that reads as a bug in the runner instead of as an
    absent split.
    """
    for _R, row in [p for pts in series.values() for p in pts]:
        if not math.isnan(_num(row, "test_max_rel_err")):
            return "test_max_rel_err", "precision (test set)"
    return "train_max_rel_err", "precision (train set — no split)"


def load_cardinality_rows(path: Path) -> dict[str, dict[str, list[tuple[float, dict]]]]:
    """``{dataset: {method: [(R, row), ...]}}`` from the matched-cardinality cells.

    Grouping and the skip filter come from ``tabular.rows_by_dataset_and_method``, so
    "which rows are usable" is decided in one place for every consumer -- ``decrement``
    already went through that helper, and two figure families drawn from different
    subsets of the same CSV would differ with nothing to signal it. What is added here is
    specific to these figures: attaching the achieved ``R``, dropping rows that have none,
    and collapsing duplicates so each curve is single-valued and monotone in x.
    """
    out: dict[str, dict[str, list[tuple[float, dict]]]] = defaultdict(lambda: defaultdict(list))
    for ds, by_method in rows_by_dataset_and_method(path, "cardinality").items():
        for m, rows in by_method.items():
            for r in rows:
                R = _num(r, "R")
                if math.isnan(R) or R <= 0:
                    continue
                out[ds][m].append((R, r))
    for ds in out:
        for m in out[ds]:
            # Several requested R can collapse to the same achieved R (a method may cap
            # at n_train); keep the first and sort so the lines are monotone in x.
            seen: dict[float, dict] = {}
            # Sort on R alone: two rows can share an achieved R (a method caps at
            # n_train, so several requested R collapse), and tuple ordering would then
            # fall through to comparing the row dicts.
            for R, row in sorted(out[ds][m], key=lambda pair: pair[0]):
                seen.setdefault(R, row)
            out[ds][m] = sorted(seen.items())
    return out


def _panel(ax, series, column, ylabel, yscale, title, *, dashed_train=False,
           only=None):
    """Draw one metric panel.

    ``only`` restricts to an explicit method set, bypassing the usual exclusion. That is
    how the reference methods get their own figure: on a shared axis their range swamps
    everything, but alone they are perfectly readable.
    """
    plotted = 0
    primary_xy: list[tuple[float, float]] = []
    primary: list[float] = []
    if only is not None:
        series = {m: p for m, p in series.items() if m in only}
    else:
        series = {m: p for m, p in series.items() if m not in FIGURE_EXCLUDED}
    for method, points in series.items():
        style = style_for(method)
        xs = [R for R, _ in points]
        ys = [_num(row, column) for _, row in points]
        # Only missing values are dropped. Zeros and negatives plot where they fall --
        # on a linear axis they are ordinary coordinates, and dropping them would hide
        # real results (NMF issues 0 constrained solves; ADG issues 0 at R=1 and R=2).
        good = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
        if not good:
            continue
        ax.plot([g[0] for g in good], [g[1] for g in good],
                color=style["color"], marker=style["marker"], ls=style["ls"],
                label=style["label"], ms=4, lw=1.4, alpha=0.9)
        primary.extend(g[1] for g in good)
        primary_xy.extend(good)
        plotted += 1
        if dashed_train:
            yt = [_num(row, "train_max_rel_err") for _, row in points]
            gt = [(x, y) for x, y in zip(xs, yt) if not math.isnan(y)]
            if gt:
                ax.plot([g[0] for g in gt], [g[1] for g in gt],
                        color=style["color"], ls=":", lw=0.9, alpha=0.45)
    ax.set_yscale(yscale)
    # Scale to the primary (test) series, with a little headroom. The dashed train
    # overlay is a reference, not the subject: it reaches numerical zero as soon as the
    # cone contains every training snapshot, and letting it drive the limits compresses
    # the curves actually being compared. Train lines clip rather than rescale the axis.
    ceiling = NUMERICAL_CEILING.get(column)
    if primary:
        usable = [v for v in primary if ceiling is None or v <= ceiling] or primary
        lo, hi = min(usable), max(usable)
        if yscale == "log":
            # Multiplicative headroom: lo - pad would be <= 0 and unrenderable.
            ax.set_ylim(max(lo, 1e-300) / 2.0, hi * 2.0)
        else:
            pad = (hi - lo) * 0.05 or (abs(hi) * 0.05 or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
    # Mark the singular region along x, not with a line at the ceiling's y. The ceiling is
    # far above the meaningful range (4.5e15 against a usable max of 3.1e13 on Half-disks
    # of Hertz), so an axhline there would either re-inflate the axis it exists to bound
    # or -- placed outside ylim with bbox_inches="tight" -- expand the canvas to reach it,
    # which is exactly how this first went wrong: a 342-billion-pixel figure. Shading the
    # cardinalities instead says the more useful thing anyway: past this R the basis is
    # numerically singular and the curve beyond is round-off.
    if ceiling is not None and primary_xy and max(y for _x, y in primary_xy) > ceiling:
        x0 = min(x for x, y in primary_xy if y > ceiling)
        ax.axvspan(x0, max(x for x, _y in primary_xy), color="#b03a2e", alpha=0.07,
                   lw=0, zorder=0)
        ax.text(x0, 0.97, f"  numerically singular (R ≥ {x0:g})", color="#b03a2e",
                fontsize=6.5, ha="left", va="top", clip_on=True,
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("cardinality $R$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    return plotted
