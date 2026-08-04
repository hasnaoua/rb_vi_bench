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
* **conditioning** -- Gram condition number, log scale. Undefined below R=2.
* **orthogonality** -- ``e_orth`` ([NDEE22] Eq. 41), bounded by 1; higher is a wider cone.
* **offline cost** -- total constrained-solver calls. Machine-independent, unlike
  wall-clock -- but it counts *constrained* solves only, so NMF sits at exactly 0 and
  that means "issues none", not "is free".

The ``orthant`` and ``pod_control`` references are kept off the *comparison* axes -- see
``FIGURE_EXCLUDED`` -- because they sit orders of magnitude from the methods being
compared and would cost the shared axis its resolution. They are not hidden: they get
their own figure per dataset (``reference_orthant.png``, and one panel per metric under
``<dataset>/orthant/`` with ``--split``) carrying every panel the comparison figures do.
Alone on their own axes there is no shared range to protect.

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

``test_every_axis_is_linear`` pins this, so a transformation cannot be reintroduced on
one panel without the rule being restated deliberately.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from . import _paths  # noqa: F401  -- forces the Agg backend before pyplot is imported

import matplotlib.pyplot as plt

from . import layout
from .tabular import num as _num, read_rows

RESULTS = _paths.ROOT / "results"

# Stable per-method styling, so a method keeps its colour across every figure.
# Families share a hue: CPG blues, mCPG greens, ADG oranges, baselines grey/red.
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

#: Extra split figures, written only by ``--split``. Kept out of ``PANELS`` so the
#: comparison layout stays a 2x2 grid.
EXTRA_SPLIT_PANELS = (
    ("test_max_rel_err_persnap", "max per-snapshot relative error", "linear",
     "precision, each snapshot vs ITS OWN norm"),
)

PANELS = (
    ("test_max_rel_err", "max relative projection error", "linear", "precision (test set)"),
    ("gram_cond",        "Gram condition number",         "linear", "conditioning"),
    ("e_orth_mean",      "mean $e_{orth}$",               "linear", "orthogonality (Eq. 41)"),
    ("calls_total",      "constrained solver calls",      "linear", "offline cost"),
)


def error_column(series) -> tuple[str, str]:
    """Pick the error column a dataset can actually support.

    ``physics`` has **no train/test split** -- ``greedy_algos``' physics pipeline builds
    and evaluates on the whole dataset deliberately -- so its ``test_max_rel_err`` is
    ``nan`` throughout and a test-error panel comes out blank. Fall back to the training
    error and say so in the title, rather than shipping an empty axes that reads like
    missing data.
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


#: Measured in every table, never drawn.
#:
#: Both are *references*, not competitors, and both sit orders of magnitude away from the
#: methods being compared -- the orthant because it is the widest admissible cone (90 deg
#: aperture, near-total excess), POD because its error falls to machine zero past the
#: numerical rank. Plotting either forces the shared axis to span their range and squeezes
#: the curves that matter into a thin band. Their numbers stay in ``grid.csv`` and
#: ``report.txt``, where a reader can consult them without paying for them visually.
FIGURE_EXCLUDED: frozenset[str] = frozenset({"orthant", "pod_control"})

def _panel(ax, series, column, ylabel, yscale, title, *, dashed_train=False,
           only=None):
    """Draw one metric panel.

    ``only`` restricts to an explicit method set, bypassing the usual exclusion. That is
    how the reference methods get their own figure: on a shared axis their range swamps
    everything, but alone they are perfectly readable.
    """
    plotted = 0
    primary: list[float] = []
    if only is not None:
        series = {m: p for m, p in series.items() if m in only}
    else:
        series = {m: p for m, p in series.items() if m not in FIGURE_EXCLUDED}
    for method, points in series.items():
        style = STYLE.get(method, dict(color="black", marker=".", ls="-", label=method))
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
    if primary:
        lo, hi = min(primary), max(primary)
        pad = (hi - lo) * 0.05 or (abs(hi) * 0.05 or 1.0)
        ax.set_ylim(lo - pad, hi + pad)
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


#: Kept as an alias so existing imports keep working; the layout module owns it now.
_slug = layout.slug


def figures_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric, under ``<out_dir>/<dataset>/``.

    Same content as the four-panel figure, but each metric gets its own axes and its own
    file -- the form you want for dropping a single curve into a document, where a 2x2
    grid would have to be cropped.
    """
    err_col, err_title = error_column(series)
    ds_dir = layout.ensure(layout.metrics_dir(out_dir, dataset))

    written: list[Path] = []
    for column, ylabel, yscale, title in PANELS + EXTRA_SPLIT_PANELS:
        name = column
        if column == "test_max_rel_err":
            column, title = err_col, err_title
            name = "precision"
        elif column == "gram_cond":
            name = "conditioning"
        elif column == "e_orth_mean":
            name = "orthogonality"
        elif column == "calls_total":
            name = "offline_cost"
        elif column == "test_max_rel_err_persnap":
            name = "precision_persnap"
            # Same no-split fallback the shared column gets: physics has no test set, so
            # its test column is nan throughout and the panel would come out blank.
            if err_col == "train_max_rel_err":
                column = "train_max_rel_err_persnap"
                title += " — train (no split)"

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        plotted = _panel(ax, series, column, ylabel, yscale, title,
                         dashed_train=(column == "test_max_rel_err"))
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{dataset} — {title}", fontsize=11)
        ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="best")
        fig.tight_layout()
        path = ds_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


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
        plt.close(fig)
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
        plt.close(fig)
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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figures_reference_split(dataset: str, series, out_dir: Path) -> list[Path]:
    """One standalone PNG per metric for the references, under ``<dataset>/orthant/``."""
    ref = {m: p for m, p in series.items() if m in FIGURE_EXCLUDED}
    if not any(ref.values()):
        return []
    err_col, err_title = error_column(series)
    out = layout.ensure(layout.dataset_dir(out_dir, dataset) / "orthant")
    names = {"test_max_rel_err": "precision", "gram_cond": "conditioning",
             "e_orth_mean": "orthogonality", "calls_total": "offline_cost",
             "cover_mean_err": "cone_missed", "excess_mean_err": "cone_excess",
             "cone_sym_err": "cone_two_sided", "aperture_mean_deg": "aperture",
             "section_extent": "cone_extent"}
    written: list[Path] = []
    for column, ylabel, yscale, title in PANELS + CONE_PANELS:
        name = names[column]
        if column == "test_max_rel_err":
            column, title = err_col, err_title
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        if not _panel(ax, ref, column, ylabel, yscale, title, only=set(ref)):
            plt.close(fig)
            continue
        ax.set_title(f"{dataset} — {title}", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


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
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="metric-vs-cardinality figures")
    p.add_argument("--results", type=Path, default=RESULTS,
                   help="directory holding grid.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="where to write PNGs (default: <results>/figures)")
    p.add_argument("--split", action="store_true",
                   help="also write one PNG per metric under <out>/<dataset>/")
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
                          figure_reference(dataset, all_series[dataset], out_dir)):
                if extra:
                    written.append(extra)
        if args.split:
            written.extend(figures_split(dataset, all_series[dataset], out_dir))
            written.extend(figures_reference_split(dataset, all_series[dataset], out_dir))
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
