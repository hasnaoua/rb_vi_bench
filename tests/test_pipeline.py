"""The pipeline: runner, figures, reconstruction, decrement -- the artefacts themselves.

Everything downstream of a fitted cone. The runner's contract is that a cell which does
not run emits a ``skip_reason`` rather than disappearing, because "not measured here" and
"scored badly here" must never be confusable in ``grid.csv``. The figure modules' contract
is that a curve which cannot be drawn is *said* to be absent rather than silently omitted,
and that axes stay linear except for the one documented exception.

The rendering tests assert on file size rather than on pixels: a figure that raises no
exception but draws nothing still writes a valid PNG, and only its size gives it away.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench import _paths
from bench import datasets as ds_mod
from bench.adapters import METHODS
from bench.types import Dataset


def test_subsample_carries_the_field_geometry_through():
    """Subsampling drops SNAPSHOTS, never the dataset's description of its own nodes.

    ``_subsample`` rebuilds the Dataset field by field, and ``geometry`` was missing from
    that list -- so any dataset above the subsample threshold reached the reconstruction
    figures with ``geometry=None`` and was drawn as an index-ordered curve. For
    ``fem_lambda`` that silently discards the mirroring that makes the figure match
    [BEE20] Fig. 7-8, and for ``physics`` it flattens a 76x101 grid into a 7676-point
    line. No shipped dataset is currently both geometric and large enough to trip it, so
    nothing in the results was wrong -- but it fires at any smaller ``--subsample``,
    which is exactly when someone is iterating quickly and least likely to notice.
    """
    from bench.runner import _subsample

    for key in ("fem_lambda", "fem_lambda_pressure", "physics"):
        full = ds_mod.load(key)
        sub = _subsample(full, 20)
        assert sub.n_snapshots == 20, key
        assert sub.geometry is not None, f"{key}: geometry dropped by subsampling"
        assert sub.geometry.kind == full.geometry.kind, key
        # Geometry describes NODES, which subsampling does not touch.
        assert sub.dim == full.dim, key


def test_subsample_preserves_split_and_callable():
    """``_subsample`` rebuilds a Dataset through asdict, which drops callables.

    ``B_of_mu`` has to be re-attached and re-indexed onto the retained columns, or the
    inf-sup metrics would silently read the wrong parameter.
    """
    from bench.runner import _subsample

    ds = ds_mod.load("toy_bee20")
    small = _subsample(ds, 12)
    assert small.n_snapshots <= 12
    assert small.supports_infsup, "B_of_mu was lost through the rebuild"
    assert small.A is not None
    assert small.primal_snapshots.shape[1] == small.n_snapshots
    if small.train_idx is not None:
        assert small.train_idx.max() < small.n_snapshots
    if small.test_idx is not None:
        assert small.test_idx.max() < small.n_snapshots
    # The retained parameters must be the ones B_of_mu now indexes.
    assert small.B_of_mu(0).shape == ds.B_of_mu(0).shape


def test_runner_records_skips_rather_than_dropping(bumps):
    """A skipped cell and a badly scoring cell must never be confusable."""
    from bench.runner import run_cell

    row = run_cell(bumps, "nmf_s0", delta=0.2, with_infsup=False, with_determinism=False)
    assert row["skip_reason"], "NMF in tolerance mode should be skipped with a reason"
    assert "cardinality-only" in row["skip_reason"]
    assert "test_max_rel_err" not in row, "a skipped cell must not carry metric columns"


def test_runner_cell_is_complete_when_it_runs(bumps):
    from bench.runner import run_cell

    row = run_cell(bumps, "cpg_bee20", delta=0.2, with_infsup=False, with_determinism=True)
    assert not row["skip_reason"]
    for key in ("R", "train_max_rel_err", "gram_cond", "calls_total", "fit_seconds",
                "rerun_same_order"):
        assert key in row, f"missing {key}"


def test_figures_render_from_a_grid(tmp_path, bumps):
    """Figures must build from a grid CSV, and cover the no-split case.

    A source with no train/test split has an all-nan test-error column, so a test-error
    panel would come out blank and read as missing data rather than as an absent split.
    The figure module has to fall back to the training error and say so.

    No registered dataset is in that state any more -- ``physics`` was the last, and now
    carries its archive's 50/49 partition -- so the fallback is exercised against a
    synthetic split-less Dataset, which is the contract that still permits it.
    """
    from bench import figures
    from bench.runner import _write_csv, run_cell

    rows = []
    no_split = Dataset(name="nosplit", snapshots=bumps.snapshots)
    for ds in (bumps, no_split):
        for R in (2, 4, 6):
            for m in ("cpg_ndee22", "mcpg_ndee22", "adg", "pod_control"):
                rows.append(run_cell(ds, m, R=R, with_infsup=False, with_determinism=False))
    _write_csv(tmp_path / "grid.csv", rows)

    series = figures.load_cardinality_rows(tmp_path / "grid.csv")
    assert set(series) == {"bumps", "nosplit"}
    assert figures.error_column(series["bumps"])[0] == "test_max_rel_err"
    assert figures.error_column(series["nosplit"])[0] == "train_max_rel_err"

    out = tmp_path / "figs"
    assert figures.main(["--results", str(tmp_path), "--out", str(out)]) == 0
    # Everything is grouped per dataset; the only cross-dataset figure is under _overview.
    assert (out / "_overview" / "precision_all_datasets.png").stat().st_size > 5000
    for ds in ("bumps", "nosplit"):
        assert (out / ds / "panel.png").stat().st_size > 5000

    # --split: one standalone PNG per metric, under <dataset>/metrics/.
    split_out = tmp_path / "split"
    assert figures.main(["--results", str(tmp_path), "--out", str(split_out),
                         "--split", "--no-panel"]) == 0
    assert not (split_out / "_overview").exists(), "--no-panel still wrote the overview"
    for ds in ("bumps", "nosplit"):
        assert not (split_out / ds / "panel.png").exists(), "--no-panel wrote a panel"
        names = sorted(p.stem for p in (split_out / ds / "metrics").glob("*.png"))
        # The four EXTRA_SPLIT_PANELS exist only here, not in the 2x2 combined panel, so
        # that grid stays square: the per-snapshot normalization and the two training
        # curves. `nosplit` gets the training panels too -- they are the only precision
        # it can report, and `precision` there is already the training error under the
        # documented no-split fallback.
        assert names == ["conditioning", "offline_cost", "orthogonality",
                         "precision", "precision_persnap",
                         "precision_train", "precision_train_persnap"], names
        for p in (split_out / ds / "metrics").glob("*.png"):
            assert p.stat().st_size > 5000, f"{ds}/{p.name} looks empty"


def test_reconstruction_uses_the_scored_approximation(bumps):
    """The plotted reconstruction must be the one the error column actually scored.

    Cone methods and NMF are scored by NNLS projection; POD by unconstrained least
    squares, because its coefficients carry no sign constraint. If the figure module and
    the metric module disagreed, a figure would show POD a reconstruction it is not
    allowed to use -- or show a cone method one that violates its own cone.
    """
    from bench.metrics.precision import (projection_errors, reconstruct,
                                         reconstruction_errors, uses_cone_projection)

    S = bumps.train()
    for key, expect_cone in (("cpg_ndee22", True), ("nmf_s0", True), ("pod_control", False)):
        result = METHODS[key].fit(bumps, R=6)
        assert uses_cone_projection(result) is expect_cone, key
        approx = reconstruct(S, result.generators, cone=expect_cone)
        drawn = np.linalg.norm(S - approx, axis=0)
        scored = (projection_errors if expect_cone else reconstruction_errors)(
            S, result.generators)
        assert np.allclose(drawn, scored, atol=1e-9), key
        if expect_cone:
            # A cone reconstruction is a non-negative combination of non-negative
            # generators, so it cannot go negative.
            assert approx.min() >= -1e-9, key


def test_reconstruction_figures_render(tmp_path):
    """Best/worst figures build, and skip snapshots whose relative error is 0/0.

    physics carries five numerically zero dual snapshots, for which
    ||theta - Pi(theta)|| / ||theta|| is undefined; ranking must not pick them.
    """
    from bench import reconstruction

    methods = ["cpg_ndee22", "adg"]
    rc = reconstruction.main([
        "--datasets", "toy_bee20", "--methods", *methods, "pod_control", "orthant",
        "--R", "4", "--split", "--out", str(tmp_path),
    ])
    assert rc == 0

    root = tmp_path / "toy_bee20" / "reconstruction"
    assert (root / "all_methods.png").stat().st_size > 5000
    # One directory per method, each holding best.png and worst.png separately.
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == sorted(methods)
    for m in methods:
        cases = sorted(p.name for p in (root / m).glob("*.png"))
        assert cases == ["best.png", "worst.png"], (m, cases)
        for p in (root / m).glob("*.png"):
            assert p.stat().st_size > 5000, f"{m}/{p.name} looks empty"


def test_reconstruction_ranking_still_guards_undefined_ratios(bumps):
    """The nan guard in the ranking stays, as defence in depth.

    Dataset now drops zero snapshots, so the guard should never fire in practice -- but
    ``_ranking`` is also callable on arbitrary column blocks, and a 0/0 there must be nan
    rather than an arbitrary finite number that could be picked as "worst".
    """
    from bench import reconstruction

    result = METHODS["cpg_ndee22"].fit(bumps, R=4)
    cols = np.column_stack([bumps.train(), np.zeros(bumps.dim)])
    rel, _approx = reconstruction._ranking(bumps, result, cols)
    assert np.isnan(rel[-1]), "0/0 must be nan, not a finite value"
    assert int(np.nanargmax(rel)) != len(rel) - 1


def test_decrement_skips_non_consecutive_cardinalities():
    """``e(R+4) - e(R)`` is four generators' worth and must not be plotted as one.

    The metric sweep uses a sparse R grid, so a decrement figure built from it would
    silently mix per-generator steps with multi-generator ones.
    """
    from bench.decrement import decrements_vs_cardinality

    xs, ys = decrements_vs_cardinality([(1, 0.5), (2, 0.4), (3, 0.35), (7, 0.2), (8, 0.19)])
    assert xs == [2, 3, 8], xs
    # Relative: the fraction of the remaining error each extra generator removes.
    assert ys == pytest.approx([0.1 / 0.5, 0.05 / 0.4, 0.01 / 0.2])


def test_decrement_axes_are_linear():
    """The decrement plots its fractions untransformed, like every other axis here.

    This was symlog, with a linear band sized two decades below the largest step. That
    band existed to keep exact zeros on a log axis and to stop round-off on a plateaued
    curve from reading as full-height excursions. A linear axis needs neither: zero is an
    ordinary coordinate, and negative decrements -- NMF going backwards, since it is
    refitted from scratch at each R -- plot where they fall.
    """
    import inspect

    from bench import decrement

    assert not hasattr(decrement, "_symlog_threshold")
    src = inspect.getsource(decrement._draw)
    assert "symlog" not in src
    assert 'ax.set_yscale' not in src, "the y-axis is left at its linear default"


def test_decrement_figures_render(tmp_path, bumps):
    """Both axes render, into the per-dataset decrement/ folder."""
    from bench import decrement
    from bench.runner import _write_csv, run_cell

    card, tol = [], []
    for R in range(1, 7):
        for m in ("cpg_ndee22", "mcpg_ndee22", "adg"):
            card.append(run_cell(bumps, m, R=R, with_infsup=False, with_determinism=False))
    for d in (0.4, 0.2, 0.1):
        for m in ("cpg_ndee22", "mcpg_ndee22", "adg"):
            tol.append(run_cell(bumps, m, delta=d, with_infsup=False, with_determinism=False))

    (tmp_path / "c").mkdir()
    (tmp_path / "t").mkdir()
    _write_csv(tmp_path / "c" / "grid.csv", card)
    _write_csv(tmp_path / "t" / "grid.csv", tol)

    out = tmp_path / "figs"
    assert decrement.main(["--cardinality-results", str(tmp_path / "c"),
                           "--tolerance-results", str(tmp_path / "t"),
                           "--out", str(out)]) == 0
    for name in ("vs_cardinality.png", "vs_tolerance.png"):
        p = out / "bumps" / "decrement" / name
        assert p.stat().st_size > 5000, name


def test_reconstruction_renders_fields_for_grid_datasets(tmp_path):
    """physics figures must be surfaces, not curves against a component index."""
    from bench import reconstruction

    rc = reconstruction.main([
        "--datasets", "physics", "--methods", "cpg_bee20", "mcpg_ndee22",
        "--R", "4", "--split", "--out", str(tmp_path),
    ])
    assert rc == 0
    root = tmp_path / "3D_Pellet-Cladding" / "reconstruction"
    assert (root / "all_methods.png").stat().st_size > 5000
    for m in ("cpg_bee20", "mcpg_ndee22"):
        for case in ("best.png", "worst.png"):
            # A three-panel field triptych is substantially larger than a line plot.
            assert (root / m / case).stat().st_size > 20000, f"{m}/{case} looks like a curve"


def test_only_the_two_references_are_kept_off_the_comparison_axes():
    """Exclusion is a property of the METHOD, not of the axis it is drawn against.

    An earlier version routed this through a mode-aware ``excluded_for(mode)`` so that
    ``adg_momentum`` could be dropped from matched-cardinality figures, where it
    coincides with ``adg``. That made the method look absent rather than coincident, so
    the exclusion was reverted and the set it fed became permanently empty -- leaving a
    ``mode`` parameter that could not change any answer. One flat set now, and the
    coincidence is shown by drawing it in dash-dot over ``adg`` instead of hiding it.
    """
    from bench import decrement, figures, plotting, reconstruction

    assert plotting.FIGURE_EXCLUDED == frozenset({"orthant", "pod_control"})
    # Every drawing module filters through the same set; none re-derives its own, and
    # none owns it either -- it lives in `plotting`, which is why importing `decrement`
    # no longer drags in `figures`' whole metric-panel machinery.
    for mod in (decrement, figures, reconstruction):
        assert mod.FIGURE_EXCLUDED is plotting.FIGURE_EXCLUDED


def test_adg_momentum_is_drawn_in_every_figure():
    """It must not silently vanish from the figures the way the earlier exclusion did.

    At matched cardinality it coincides with ``adg`` exactly -- there is no stopping rule
    in that mode -- so it overlays rather than adds information. It is drawn anyway, in
    dash-dot, so the coincidence is visible instead of the method appearing absent.
    """
    from bench.plotting import FIGURE_EXCLUDED, STYLE

    assert "adg_momentum" not in FIGURE_EXCLUDED
    assert STYLE["adg_momentum"]["ls"] == "-.", "must be distinguishable where it overlays"


def test_reference_baseline_gets_its_own_figure(tmp_path, bumps):
    """The orthant is kept off the comparison axes but must still be plottable.

    Excluding it from the shared figures protected their y-ranges, but left its evolution
    in R readable only as CSV columns. On its own axes there is no range to protect, so it
    gets every panel the comparison figures carry.
    """
    from bench import figures, plotting
    from bench.runner import _write_csv, run_cell

    rows = []
    for R in (2, 4, 6):
        for m in ("cpg_ndee22", "adg", "orthant"):
            rows.append(run_cell(bumps, m, R=R, with_infsup=False, with_determinism=False))
    _write_csv(tmp_path / "grid.csv", rows)

    out = tmp_path / "figs"
    assert figures.main(["--results", str(tmp_path), "--out", str(out), "--split"]) == 0

    ref = out / "bumps" / "reference_orthant.png"
    assert ref.stat().st_size > 5000, "reference figure missing or empty"
    names = sorted(p.stem for p in (out / "bumps" / "orthant").glob("*.png"))
    assert names == ["aperture", "conditioning", "cone_excess", "cone_extent",
                     "cone_missed", "cone_two_sided", "offline_cost", "orthogonality",
                     "precision"], names
    # And it must still be absent from the shared comparison panel.
    assert "orthant" in plotting.FIGURE_EXCLUDED


def test_train_and_test_are_drawn_on_one_shared_axis(tmp_path, bumps, monkeypatch):
    """The shared y-range must CONTAIN both curves, not merely be equal on both panels.

    Equality is the weak property and comes for free: ``sharey="row"`` guarantees it and
    guarantees the wrong thing, because matplotlib propagates each ``set_ylim`` to its
    partner. ``_panel`` sets limits from the series it just drew, so with sharey the test
    panel overwrites the training panel and a union computed from ``get_ylim()`` reads
    the same range twice -- agreeing, and clipping the training curve straight out of
    view. That bug passes an equality check.

    So this asserts containment against the plotted data: every training and test point
    must lie inside the range its row was given. That is what makes the vertical offset
    between the panels readable as the generalization gap.
    """
    from bench import figures

    # Synthetic rows with the training error ABOVE the test error. That ordering is not
    # exotic -- `physics` reports train 0.01167 against test 0.01058 at R=8, because its
    # held-out half interpolates between training parameters and is genuinely the easier
    # set. It is also the only ordering that exposes the sharey bug: when train sits
    # inside the test range, sharing the axis clips nothing and looks correct.
    series = {
        "cpg_ndee22": [
            (float(R), {"R": str(R),
                        "train_max_rel_err": str(0.30 / R),      # larger
                        "test_max_rel_err": str(0.10 / R),       # smaller
                        "train_max_rel_err_persnap": str(0.60 / R),
                        "test_max_rel_err_persnap": str(0.20 / R)})
            for R in (2, 4, 6, 8)
        ]
    }

    # Intercept the figure on its way to disk: the limits are gone once save() closes it.
    captured = {}
    real_save = figures.save

    def capture(fig, path, **kw):
        captured["ylims"] = [ax.get_ylim() for ax in fig.axes]
        captured["ydata"] = [[ln.get_ydata() for ln in ax.lines] for ax in fig.axes]
        return real_save(fig, path, **kw)

    monkeypatch.setattr(figures, "save", capture)
    path = figures.figure_train_vs_test("bumps", series, tmp_path / "figs")
    assert path is not None and path.stat().st_size > 5000

    ylims, ydata = captured["ylims"], captured["ydata"]
    assert len(ylims) == 4, f"expected a 2x2 grid, got {len(ylims)}"

    for r, (left, right) in enumerate(((0, 1), (2, 3))):
        assert ylims[left] == ylims[right], (
            f"row {r}: panels on different scales, so their offset is not the gap")
        lo, hi = ylims[left]
        for which, ax_i in (("train", left), ("test", right)):
            for line in ydata[ax_i]:
                for y in line:
                    assert lo <= y <= hi, (
                        f"row {r} {which}: {y} falls outside the shared range "
                        f"({lo}, {hi}) -- the curve is clipped out of view")


def test_train_vs_test_returns_nothing_without_a_split(tmp_path, bumps):
    """A source with no held-out set gets no figure, rather than half of one.

    The right-hand column would be empty and the left would read as if it were the whole
    story. No registered dataset is in this state any more, but ``Dataset`` still permits
    it and a blank half-figure is exactly the kind of output that gets quoted.
    """
    from bench import figures
    from bench.runner import _write_csv, run_cell
    from bench.types import Dataset

    no_split = Dataset(name="nosplit", snapshots=bumps.snapshots)
    rows = [run_cell(no_split, m, R=R, with_infsup=False, with_determinism=False)
            for R in (2, 4, 6) for m in ("cpg_ndee22", "adg")]
    _write_csv(tmp_path / "grid.csv", rows)
    series = figures.load_cardinality_rows(tmp_path / "grid.csv")["nosplit"]

    assert figures.figure_train_vs_test("nosplit", series, tmp_path / "figs") is None


def test_training_error_gets_its_own_standalone_panels():
    """``--separate`` writes the training error under both normalizations.

    It is what every greedy actually minimizes and is monotone in R for a nested cone, so
    it shows whether a method converges at all -- a different question from whether it
    generalizes, which is what the test panels answer. In the combined figure it appears
    only as a dotted overlay, which is deliberately unreadable as a curve in its own right.
    """
    from bench.figures import EXTRA_SPLIT_PANELS, SPLIT_NAMES

    columns = {c for c, *_ in EXTRA_SPLIT_PANELS}
    assert {"train_max_rel_err", "train_max_rel_err_persnap"} <= columns
    assert SPLIT_NAMES["train_max_rel_err"] == "precision_train"
    assert SPLIT_NAMES["train_max_rel_err_persnap"] == "precision_train_persnap"
    # Distinct stems, or one would overwrite the other's PNG.
    assert len(set(SPLIT_NAMES.values())) == len(SPLIT_NAMES)


def test_axes_are_linear_except_the_documented_exception():
    """No panel transforms its values, apart from conditioning, which earns it.

    The rule exists because a log axis has no coordinate for zero and discards such cells
    with no marker -- that had silently removed NMF's entire offline-cost series, ADG's
    cheapest cardinalities, and every cell where the orthant covers K_full exactly.

    gram_cond cannot hit that failure: a condition number is >= 1 by definition. And it
    needs the axis more than any other column, spanning eleven decades before the basis
    goes singular, which on a linear axis collapses the whole meaningful range onto zero.
    The exception is a set of one, and this test is what keeps it that size.
    """
    from bench.figures import (CONE_PANELS, EXTRA_SPLIT_PANELS, LOG_AXIS_EXCEPTIONS,
                               PANELS)

    assert LOG_AXIS_EXCEPTIONS == frozenset({"gram_cond"})
    for column, _ylabel, scale, _title in PANELS + CONE_PANELS + EXTRA_SPLIT_PANELS:
        expected = "log" if column in LOG_AXIS_EXCEPTIONS else "linear"
        assert scale == expected, (column, scale)


@pytest.mark.parametrize("grid", ["grid.csv", "sweep_dense/grid.csv"])
def test_the_log_exception_column_is_strictly_positive(grid):
    """The exception is only safe while gram_cond never reaches zero. Checked, not assumed.

    If a degenerate basis ever reported a condition number of 0 -- or the column were
    redefined -- the point would vanish from the panel with no marker, which is the exact
    failure the linear rule was adopted to prevent.
    """
    from bench.figures import LOG_AXIS_EXCEPTIONS
    from bench.tabular import num, read_rows

    path = _paths.RESULTS / grid
    if not path.is_file():
        pytest.skip(f"{path} not present; run bench.runner first")

    rows = [r for r in read_rows(path) if not r.get("skip_reason")]
    for column in LOG_AXIS_EXCEPTIONS:
        vals = [num(r, column) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        assert vals, f"{column} has no values in {grid}"
        assert min(vals) > 0, f"{column} reaches {min(vals)}, unrenderable on a log axis"


@pytest.mark.parametrize("grid", ["grid.csv", "sweep_dense/grid.csv"])
def test_zero_valued_cells_are_plotted_not_dropped(grid):
    """The zeros a log axis used to discard must survive into the figure data.

    These are results, not gaps: NMF issues no constrained solves, ADG issues none at
    its two smallest cardinalities, and the orthant misses nothing of K_full. Each was
    invisible under a log scale. This asserts they are present in the CSVs and that the
    panels carrying them are on an axis that can render a zero.
    """
    from bench.figures import CONE_PANELS, EXTRA_SPLIT_PANELS, PANELS
    from bench.tabular import num, read_rows

    path = _paths.RESULTS / grid
    if not path.is_file():
        pytest.skip(f"{path} not present; run bench.runner first")

    rows = [r for r in read_rows(path) if not r.get("skip_reason")]
    scales = {c: sc for c, _y, sc, _t in PANELS + CONE_PANELS + EXTRA_SPLIT_PANELS}
    zeros = {
        (column, r["method"])
        for column in scales
        for r in rows
        if num(r, column) == 0.0
    }
    assert ("calls_total", "nmf_s0") in zeros, "NMF issues no constrained solves"
    for column, _method in zeros:
        assert scales[column] == "linear", (
            f"{column} has exact zeros but is on a {scales[column]} axis, "
            "which cannot render them"
        )


def test_grid_datasets_get_the_axial_panel(tmp_path):
    """physics gains a fourth panel; a scatter geometry has no axis to collapse."""
    from bench import reconstruction

    assert reconstruction.main([
        "--datasets", "physics", "--methods", "adg", "cpg_bee20",
        "--R", "4", "--split", "--out", str(tmp_path)]) == 0
    root = tmp_path / "3D_Pellet-Cladding" / "reconstruction"
    for m in ("adg", "cpg_bee20"):
        for case in ("best.png", "worst.png"):
            # Four panels is materially wider than three.
            assert (root / m / case).stat().st_size > 30000, f"{m}/{case}"
