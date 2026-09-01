"""Metric-vs-cardinality figures from a benchmark grid.

Plots the **matched-cardinality** rows only, and that restriction is the whole point.
In tolerance mode each method reaches its own ``R``, and ``R`` is the *output*; worse,
the tolerances are not commensurable across methods -- ADG's ``epsilon`` is a
per-snapshot relative bound on ``S_norm`` while CPG/mCPG use one shared absolute
threshold ``epsilon * max_q ||theta_q||``. Putting those on a shared x-axis would draw a
comparison that does not exist. At matched cardinality every method is handed the same
``R`` and no stopping rule applies, so the curves are directly comparable.

Four comparison panels per dataset, one line per method:

* **precision** -- test (solid) and train (dashed) max relative projection error.
* **conditioning** -- Gram condition number, log scale (the one exception to the linear
  rule). Undefined below R=2, and numerically meaningless once the Gram goes singular,
  which the panel shades.
* **orthogonality** -- ``e_orth`` ([NDEE22] Eq. 41), bounded by 1; higher is a wider cone.
* **offline cost** -- total constrained-solver calls. Machine-independent, unlike
  wall-clock -- but it counts *constrained* solves only, so NMF sits at exactly 0 and
  that means "issues none", not "is free".

The ``orthant`` and ``pod_control`` references are kept off the *comparison* axes -- see
``plotting.FIGURE_EXCLUDED`` -- because they sit orders of magnitude from the methods
being compared and would cost the shared axis its resolution. They are not hidden: they
get their own figure per dataset (``reference_orthant.png``, and one panel per metric
under ``<dataset>/orthant/`` with ``--separate``) carrying every panel the comparison
figures do. Alone on their own axes there is no shared range to protect.

**Every axis is linear. No transformation is applied to any plotted value.** Values
appear at the coordinate the data puts them, which is the only way the figures can be
read against the CSVs and the report tables without a mental inverse.

The cost is real and worth stating: several of these columns span many orders of
magnitude, so on a linear axis the small values are pressed against the baseline and
differences between well-performing methods are not resolvable by eye. Those comparisons
have to be made from ``report.txt`` or the CSVs, which carry full precision.

The benefit is that nothing is silently dropped. A log axis has no coordinate for zero
and discards such points with no marker and no gap, which previously removed NMF's entire
offline-cost series (0 constrained solves in all 346 of its cells), ADG's R=1 and R=2
points (0 solves on all 9 datasets -- it seeds from a Gram-matrix argmin, and NNLS first
appears at R=3), and the 15 cells where the orthant covers ``K_full`` exactly. All of
those now plot on the zero line, where they belong.

**One documented exception: conditioning.** A condition number is >= 1 by definition, so
the zero-dropping failure that motivates the rule cannot occur for it -- across both
grids its 2429 values have minimum exactly 1. And it needs the axis: it spans eleven
decades (4.0e2 to 3.1e13 on Half-disks of Hertz) before the basis even goes singular, so
on a linear axis the whole meaningful range collapses onto zero. See
``LOG_AXIS_EXCEPTIONS``.

``test_axes_are_linear_except_the_documented_exception`` pins this, so a transformation
cannot be reintroduced on any other panel without the rule being restated deliberately,
and ``test_zero_valued_cells_are_plotted_not_dropped`` checks against the produced CSVs
that no column with exact zeros ever lands on a log axis.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from . import _paths  # noqa: F401  -- forces the Agg backend before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np

from . import cli, layout
from .plotting import FIGURE_EXCLUDED, discard, save, style_for
from .tabular import num as _num, read_rows

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
    """``{dataset: {method: [(R, row), ...]}}`` from the matched-cardinality cells."""
    out: dict[str, dict[str, list[tuple[float, dict]]]] = defaultdict(lambda: defaultdict(list))
    for r in read_rows(path):
        if r.get("mode") != "cardinality" or r.get("skip_reason"):
            continue
        R = _num(r, "R")
        if math.isnan(R) or R <= 0:
            continue
        out[r["dataset"]][r["method"]].append((R, r))
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


def figure_for_dataset(dataset: str, series, out_dir: Path) -> Path:
    err_col, err_title = error_column(series)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for ax, (column, ylabel, yscale, title) in zip(axes.ravel(), PANELS):
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        _panel(ax, series, column, ylabel, yscale, title,
               dashed_train=(column == "test_max_rel_err"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — metrics vs cardinality (matched-R mode)", fontsize=12)
    fig.text(0.5, 0.945,
             "dotted = train error;  ADG (momentum stop) overlays ADG exactly here — "
             "matched-R mode has no stopping rule",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    path = layout.ensure(layout.dataset_dir(out_dir, dataset)) / "panel.png"
    save(fig, path)
    return path


def figure_train_vs_test(dataset: str, series, out_dir: Path) -> Path | None:
    """Training and test error side by side, on one shared y-axis, per dataset.

    Every source in the merge now carries a train/test split, which makes the pair of
    curves readable as a pair for the first time. The comparison panel shows the training
    error only as a dotted overlay, because there its job is to be a reference for the
    test curve rather than a subject; that is the right call for a four-metric summary
    and the wrong one if the question is the *gap* itself.

    **The shared y-axis is the whole point of the figure.** Two panels drawn with
    independent limits each fill their own axes, so a method that generalizes badly looks
    identical to one that generalizes perfectly -- the eye compares shapes and the scales
    silently differ, sometimes by orders of magnitude. Forcing one range onto both makes
    the vertical offset between the panels *be* the generalization gap. A method whose
    two curves sit at the same height is interpolating; one whose test curve rides above
    its training curve is not, and by how much is now something you can see rather than
    something you have to read off two sets of tick labels.

    Both normalizations get a row, because they answer different questions and diverge
    exactly where snapshot magnitudes spread (see ``metrics.precision``): the shared
    denominator is comparable across methods and readable against the tolerance that
    produced it, the per-snapshot one is what ADG's tolerance actually bounds.

    Returns ``None`` for a source with no split -- there is nothing to put in the right
    column, and half a figure is worse than none.
    """
    if not any(not math.isnan(_num(row, "test_max_rel_err"))
               for pts in series.values() for _R, row in pts):
        return None

    rows = (
        ("max relative projection error", r"shared denominator  ($\max_q\|\theta_q\|$)",
         "train_max_rel_err", "test_max_rel_err"),
        ("max per-snapshot relative error", "per-snapshot  (each vs its OWN norm)",
         "train_max_rel_err_persnap", "test_max_rel_err_persnap"),
    )
    # Deliberately NOT sharey="row". Sharing the axis makes matplotlib propagate each
    # ``set_ylim`` to its partner, and ``_panel`` sets limits from the series it just
    # drew -- so the second panel silently overwrites the first, and a union computed
    # afterwards reads the same (test) range from both axes. The training curve then
    # clips out of view, which is the exact failure this figure exists to prevent. Draw
    # both independently, then union the ranges they each asked for.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    drawn = 0
    for r, (ylabel, what, train_col, test_col) in enumerate(rows):
        pair, wanted = [], []
        for c, (column, which) in enumerate(((train_col, "TRAINING"), (test_col, "TEST"))):
            ax = axes[r, c]
            if _panel(ax, series, column, ylabel if c == 0 else "", "linear",
                      f"{which} set — {what}"):
                drawn += 1
                wanted.append(ax.get_ylim())
            pair.append(ax)
        if wanted:
            lo, hi = min(w[0] for w in wanted), max(w[1] for w in wanted)
            for ax in pair:
                ax.set_ylim(lo, hi)

    if not drawn:
        discard(fig)
        return None

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — training vs held-out error", fontsize=12)
    fig.text(0.5, 0.945,
             "each row shares one y-axis, so the vertical offset between the two panels "
             "IS the generalization gap",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return save(fig, layout.ensure(layout.dataset_dir(out_dir, dataset)) / "train_vs_test.png")


#: Kept as an alias so existing imports keep working; the layout module owns it now.
_slug = layout.slug


def _write_panels(dataset: str, series, out: Path, panels, *, only=None,
                  legend: dict, dashed_train: bool) -> list[Path]:
    """One standalone PNG per panel spec, into ``out``. Returns what it wrote.

    The scaffolding both ``--separate`` paths need: resolve the error column once, then
    for each panel open an axes, draw, and either save it or throw it away when the panel
    turned out to have no curve on it. Written twice it was forty lines of identical code
    around five differences, which is exactly the shape where a fix lands in one copy.

    A panel with nothing plotted is *discarded*, not saved empty. An empty axes is the
    worst possible output here: it renders as a valid PNG of a blank grid, which reads as
    "measured, and the answer was nothing" rather than "not measured".
    """
    err_col, err_title = error_column(series)
    written: list[Path] = []
    for column, ylabel, yscale, title in panels:
        name = SPLIT_NAMES.get(column, column)
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        elif column == "test_max_rel_err_persnap" and err_col == "train_max_rel_err":
            # Same no-split fallback the shared column gets: a split-less source has a
            # nan test column throughout and the panel would come out blank.
            column = "train_max_rel_err_persnap"
            title += " — train (no split)"

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if not _panel(ax, series, column, ylabel, yscale, title, only=only,
                      dashed_train=dashed_train and column == "test_max_rel_err"):
            discard(fig)
            continue
        ax.set_title(f"{dataset} — {title}", fontsize=11)
        ax.legend(frameon=False, loc="best", **legend)
        fig.tight_layout()
        written.append(save(fig, out / f"{name}.png"))
    return written


def figures_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric, under ``<out_dir>/<dataset>/``.

    Same content as the four-panel figure, but each metric gets its own axes and its own
    file -- the form you want for dropping a single curve into a document, where a 2x2
    grid would have to be cropped.
    """
    return _write_panels(
        dataset, series, layout.ensure(layout.metrics_dir(out_dir, dataset)),
        PANELS + EXTRA_SPLIT_PANELS,
        legend=dict(fontsize=7.5, ncol=2), dashed_train=True)


def figure_cone_geometry(dataset: str, series, out_dir: Path) -> Path | None:
    """Cone-geometry panels vs cardinality, all methods on one axis.

    Separate from the metric panel because it answers a different question: those measure
    a cone against the finite snapshot set, these measure it against the whole cone the
    snapshots generate. A method can be identical on the first and very different on the
    second -- which is exactly what mCPG does.
    """
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0))
    drawn = 0
    for ax, (column, ylabel, yscale, title) in zip(axes.ravel(), CONE_PANELS):
        drawn += _panel(ax, series, column, ylabel, yscale, title)
    for ax in axes.ravel()[len(CONE_PANELS):]:
        ax.set_axis_off()
    if not drawn:
        discard(fig)
        return None

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset} — cone geometry vs cardinality", fontsize=12)
    fig.text(0.5, 0.945,
             r"$K_{full}=span_+\{$all snapshots$\}$.  Both directions shown: neither "
             r"implies the other, and lower is better in both.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = layout.ensure(layout.dataset_dir(out_dir, dataset)) / "cone_geometry.png"
    save(fig, path)
    return path


#: The ADG initialization ablation: identical algorithm, different first step.
#:
#: Stock ``adg`` opens with the PAIR of snapshots at the largest mutual angle. ``adg_k0``
#: opens from the empty cone, as [BEE20] Alg. 2 line 2 and [NDEE22] Alg. 2 line 3 both
#: do, which collapses the first selection to [BEE20] Eq. (56). Everything after that --
#: angular-defect argmax, batch admission, normalization, stopping -- is shared, so any
#: separation between the curves is the initialization and nothing else.
#:
#: They get their own figure rather than another pair of lines on the comparison axes:
#: on most datasets the two coincide exactly, and two overlaid identical curves among six
#: others reads as a rendering artefact rather than as the result it is.
ADG_INIT_METHODS: tuple[str, ...] = ("adg", "adg_k0")


def figure_adg_init(dataset: str, series, out_dir: Path) -> Path | None:
    """ADG's two initializations, side by side on every metric.

    The comparison is controlled: one variant differs from the other only in how the
    first generator is chosen, so a gap between the curves isolates that choice. Reading
    it, two things are worth knowing in advance.

    **Coinciding curves are the common outcome, not a bug.** Whenever the largest-norm
    snapshot already lies in the largest-mutual-angle pair, the two cones agree from R=2
    onward and the lines lie exactly on top of each other. The dashed ``adg_k0`` style is
    chosen so that this is visible as agreement rather than as a missing series.

    **Only ``adg_k0`` is defined at R=1.** Starting from a pair means stock ADG's
    trajectory begins at two generators, so its curve starts at R=2 while the ablation's
    starts at R=1. The offset is the point, not a gap in the data.
    """
    sub = {m: p for m, p in series.items() if m in ADG_INIT_METHODS and p}
    if len(sub) < 2:
        return None

    err_col, err_title = error_column(series)
    panels = list(PANELS + CONE_PANELS)
    nrows = -(-len(panels) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 3.75 * nrows))
    drawn = 0
    for ax, (column, ylabel, yscale, title) in zip(axes.ravel(), panels):
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        drawn += _panel(ax, sub, column, ylabel, yscale, title, only=set(sub))
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    if not drawn:
        discard(fig)
        return None

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{dataset} — ADG initialization: largest-angle pair vs $K_0=\\{{0\\}}$\n"
        "identical angular-defect rule, batch admission, normalization and stopping; "
        "only the first generator differs",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = layout.ensure(layout.dataset_dir(out_dir, dataset)) / "adg_initialization.png"
    save(fig, path)
    return path


def figures_adg_init_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric for the initialization ablation."""
    sub = {m: p for m, p in series.items() if m in ADG_INIT_METHODS and p}
    if len(sub) < 2:
        return []
    err_col, err_title = error_column(series)
    out = layout.ensure(layout.dataset_dir(out_dir, dataset) / "adg_init")
    written: list[Path] = []
    for column, ylabel, yscale, title in PANELS + CONE_PANELS:
        name = SPLIT_NAMES.get(column, column)
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if not _panel(ax, sub, column, ylabel, yscale, title, only=set(sub)):
            discard(fig)
            continue
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"{dataset} — {title}", fontsize=10)
        fig.tight_layout()
        path = out / f"{name}.png"
        save(fig, path)
        written.append(path)
    return written


def figure_reference(dataset: str, series, out_dir: Path) -> Path | None:
    """The reference methods on their own axes -- every metric, one figure.

    ``FIGURE_EXCLUDED`` keeps the orthant out of the comparison figures because its range
    swamps the shared axis: it is the widest admissible cone, so its aperture is pinned at
    90 degrees and its excess near-total, and a shared y-range spanning that leaves the
    four real curves in a thin band. Excluding it, though, meant its *evolution* in R was
    only readable as CSV columns.

    Alone on its own axes there is no such conflict, so this draws every panel the
    comparison figures carry -- the four metric panels and the four cone-geometry ones --
    for the references only. Same columns, same scales, no shared range to protect.
    """
    ref = {m: p for m, p in series.items() if m in FIGURE_EXCLUDED}
    if not any(ref.values()):
        return None

    err_col, err_title = error_column(series)
    panels = list(PANELS + CONE_PANELS)
    nrows = -(-len(panels) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 3.75 * nrows))
    drawn = 0
    for ax, (column, ylabel, yscale, title) in zip(axes.ravel(), panels):
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        drawn += _panel(ax, ref, column, ylabel, yscale, title, only=set(ref))
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    if not drawn:
        discard(fig)
        return None

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{dataset} — reference baselines, every metric vs cardinality\n"
        r"orthant = $span_+$ of canonical directions along mCPG's iteration;"
        r" at $R=\dim$ it is all of $W^+$",
        fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = layout.ensure(layout.dataset_dir(out_dir, dataset)) / "reference_orthant.png"
    save(fig, path)
    return path


def figures_reference_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric for the references, under ``<dataset>/orthant/``.

    On their own axes there is no shared range to protect, so the references get every
    panel the comparison figures carry -- which is the point of the separate directory.
    No dashed training overlay: these are reference curves, and a second line per method
    would clutter an axes whose whole job is to be readable in isolation.
    """
    ref = {m: p for m, p in series.items() if m in FIGURE_EXCLUDED}
    if not any(ref.values()):
        return []
    return _write_panels(
        dataset, ref, layout.ensure(layout.dataset_dir(out_dir, dataset) / "orthant"),
        PANELS + CONE_PANELS,
        only=set(ref), legend=dict(fontsize=8), dashed_train=False)


def figure_precision_overview(all_series, out_dir: Path) -> Path:
    """One precision panel per dataset -- the cross-dataset summary."""
    names = sorted(all_series)
    ncol = 4
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        col, _title = error_column(all_series[name])
        suffix = "" if col == "test_max_rel_err" else "  (train — no split)"
        _panel(ax, all_series[name], col, "max rel. error", "linear", name + suffix)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Precision vs cardinality, all datasets (matched-R mode)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    path = layout.ensure(layout.overview_dir(out_dir)) / "precision_all_datasets.png"
    save(fig, path)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="metric-vs-cardinality figures")
    cli.add_results(p)
    cli.add_out(p, None, what="PNGs (default: <results>/figures)")
    cli.add_separate(p, what="metric")
    p.add_argument("--no-panel", action="store_true",
                   help="skip the combined four-panel and overview figures")
    args = p.parse_args(argv)

    grid = args.results / "grid.csv"
    if not grid.is_file():
        raise SystemExit(f"{grid} not found; run `python -m bench.runner` first")

    out_dir = args.out or (args.results / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series = load_cardinality_rows(grid)
    if not all_series:
        raise SystemExit(
            f"{grid} has no matched-cardinality rows. Figures are drawn from that mode "
            "only -- tolerance-mode R is an output, and the tolerances are not "
            "commensurable across methods. Re-run with --cardinalities."
        )

    written = []
    for dataset in sorted(all_series):
        if not args.no_panel:
            written.append(figure_for_dataset(dataset, all_series[dataset], out_dir))
            for extra in (figure_cone_geometry(dataset, all_series[dataset], out_dir),
                          figure_adg_init(dataset, all_series[dataset], out_dir),
                          figure_train_vs_test(dataset, all_series[dataset], out_dir),
                          figure_reference(dataset, all_series[dataset], out_dir)):
                if extra:
                    written.append(extra)
        if args.separate:
            written.extend(figures_split(dataset, all_series[dataset], out_dir))
            written.extend(figures_reference_split(dataset, all_series[dataset], out_dir))
            written.extend(figures_adg_init_split(dataset, all_series[dataset], out_dir))
    if not args.no_panel:
        written.append(figure_precision_overview(all_series, out_dir))

    n_pts = {d: max((len(v) for v in s.values()), default=0) for d, s in all_series.items()}
    thin = [d for d, n in n_pts.items() if n < 4]
    for path in written:
        print(path)
    if thin:
        print(f"\nnote: only {min(n_pts.values())} cardinality points for {', '.join(thin)}; "
              "pass more --cardinalities for smoother curves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
