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
from pathlib import Path

from . import _paths  # noqa: F401  -- forces the Agg backend before pyplot is imported

import matplotlib.pyplot as plt

from . import cli, layout
from .plotting import FIGURE_EXCLUDED, discard, save
from .tabular import num as _num
from .panels import (  # noqa: F401  -- re-exported: tests and callers import these
    CONE_PANELS,
    EXTRA_SPLIT_PANELS,
    LOG_AXIS_EXCEPTIONS,
    NUMERICAL_CEILING,
    PANELS,
    SPLIT_NAMES,
    _panel,
    error_column,
    load_cardinality_rows,
)



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


def _write_grid_figure(dataset: str, series, subset, out_dir: Path, panels, *,
                       ncol: int, suptitle: str, filename: str,
                       rect: tuple[float, float, float, float]) -> Path | None:
    """One multi-panel figure over ``panels``, drawn for ``subset`` only.

    The grid-path counterpart to ``_write_panels``. Two builders had this sequence
    written out in full -- resolve the error column, size the grid from the panel list,
    loop applying the ``test_max_rel_err`` substitution, blank the spare axes, bail if
    nothing drew, lift the legend off the first axes, save -- differing in four
    parameters. Written twice, a change to the substitution rule or the spare-axes
    blanking lands in one copy.

    ``series`` is the full mapping and ``subset`` the methods to draw: the error column
    is resolved from the former (a dataset either has a held-out split or it does not,
    and that is a property of the dataset, not of whichever methods this figure shows),
    while ``only=`` restricts the drawing to the latter.
    """
    err_col, err_title = error_column(series)
    panels = list(panels)
    nrows = -(-len(panels) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 3.75 * nrows))
    drawn = 0
    for ax, (column, ylabel, yscale, title) in zip(axes.ravel(), panels):
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        drawn += _panel(ax, subset, column, ylabel, yscale, title, only=set(subset))
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    if not drawn:
        discard(fig)
        return None

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=rect)
    return save(fig, layout.ensure(layout.dataset_dir(out_dir, dataset)) / filename)


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
    # Rows from the panel list, never a literal: a hardcoded grid silently truncates
    # through zip() the moment CONE_PANELS outgrows it -- no error, and a figure that
    # still renders and still looks complete. The list has grown twice already.
    nrows = -(-len(CONE_PANELS) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 4.0 * nrows))
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
    return _write_grid_figure(
        dataset, series, sub, out_dir, PANELS + CONE_PANELS,
        ncol=2,
        suptitle=(f"{dataset} — ADG initialization: largest-angle pair vs "
                  f"$K_0=\\{{0\\}}$\n"
                  "identical angular-defect rule, batch admission, normalization and "
                  "stopping; only the first generator differs"),
        filename="adg_initialization.png",
        rect=(0, 0.03, 1, 0.95))


def figures_adg_init_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric for the initialization ablation.

    Through ``_write_panels`` like the other split path: this was a third hand-rolled
    copy of that loop, and had already drifted from it (title one point smaller, no
    ``loc="best"``) while missing the split-less ``persnap`` fallback the helper gained
    afterwards.
    """
    sub = {m: p for m, p in series.items() if m in ADG_INIT_METHODS and p}
    if len(sub) < 2:
        return []
    return _write_panels(
        dataset, sub, layout.ensure(layout.dataset_dir(out_dir, dataset) / "adg_init"),
        PANELS + CONE_PANELS,
        only=set(sub), legend=dict(fontsize=8), dashed_train=False)


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
    return _write_grid_figure(
        dataset, series, ref, out_dir, PANELS + CONE_PANELS,
        ncol=3,
        suptitle=(f"{dataset} — reference baselines, every metric vs cardinality\n"
                  r"orthant = $span_+$ of canonical directions along mCPG's iteration;"
                  r" at $R=\dim$ it is all of $W^+$"),
        filename="reference_orthant.png",
        rect=(0, 0.03, 1, 0.96))


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
    # Read before creating anything: the message comes from tabular.read_rows, which owns
    # it, and is re-raised as SystemExit so a missing file is a one-line CLI error rather
    # than a traceback. Previously this pre-checked and carried its own copy of the hint.
    try:
        all_series = load_cardinality_rows(grid)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    out_dir = args.out or (args.results / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

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
