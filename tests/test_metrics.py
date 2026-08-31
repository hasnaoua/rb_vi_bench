"""The metrics: each must report a real property, not an artefact of the harness.

A metric that quietly measures the wrong thing is the worst failure mode available here,
because the number it produces still looks like a result. So each test pins the metric
against a case whose answer is known independently -- a cone whose generators *are*
snapshots has exactly zero excess, a nested cone's section can only grow, a projection
error is bounded by an unconstrained one.

Covers ``metrics.cone_geometry`` (coverage, excess, aperture, section extent, two-sided
discrepancy), ``metrics.precision`` (the two normalizations and where they diverge),
``metrics.stability`` (Gram conditioning) and ``metrics.online`` (solving the reduced
saddle-point problem rather than scoring a cone against snapshots).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from bench import _paths
from bench import datasets as ds_mod
from bench import metrics
from bench.adapters import METHODS
from bench.types import Dataset
from _fixtures import basis_of


def test_selection_based_methods_stay_inside_the_snapshot_cone(bumps):
    """CPG and ADG pick snapshots as generators, so K_R subset K_full by construction."""
    from bench.metrics import cone_geometry

    for key in ("cpg_bee20", "adg"):
        r = METHODS[key].fit(bumps, R=6)
        stats = cone_geometry.reach_outside(bumps, r.generators)
        assert stats["outside_K_full_frac"] == 0.0, key
        assert stats["outside_K_full_max"] < cone_geometry.OUTSIDE_TOL, key


def test_mcpg_can_leave_the_snapshot_cone_but_never_W_plus():
    """mCPG's cone can extend beyond span_+{snapshots} -- and does, on real data.

    Its generators are nu_r = (theta_q - Upsilon_r)/||.||, and a *difference* of two
    elements of span_+{theta} need not lie in span_+{theta}. Line 9's second constraint
    only guarantees nu_r >= 0, i.e. membership of W^+ -- which is the property the method
    needs. [NDEE22] Remark 4.3's parenthetical claim of Span^+({theta_{q_n}}) is therefore
    stronger than what actually holds.

    It is data-dependent, not universal: on the synthetic bump fixture every generator
    stays inside, while on toy_bee20 a quarter of them leave and on fem_lambda most do.
    The test uses toy_bee20 for that reason, and cross-checks against LP feasibility
    rather than trusting the NNLS residual alone.
    """
    from scipy.optimize import linprog

    from bench.metrics import cone_geometry

    ds = ds_mod.load("toy_bee20")
    r = METHODS["mcpg_ndee22"].fit(ds, R=8)
    G, T = r.generators, ds.train()
    stats = cone_geometry.reach_outside(ds, G)

    assert stats["min_entry"] >= -1e-12, "mCPG generator left W^+ -- that would be a bug"
    assert stats["outside_K_full_frac"] > 0.0, "expected mCPG to leave the snapshot cone"

    # Independent confirmation: exists c >= 0 with T c = nu_r ?
    infeasible = 0
    for k in range(G.shape[1]):
        lp = linprog(np.zeros(T.shape[1]), A_eq=T, b_eq=G[:, k],
                     bounds=[(0, None)] * T.shape[1], method="highs")
        infeasible += int(not lp.success)
    assert infeasible > 0, "NNLS said outside, LP said inside -- metric is unreliable"
    assert infeasible / G.shape[1] == pytest.approx(stats["outside_K_full_frac"], abs=0.2)


def test_selection_methods_never_leave_on_any_fast_dataset():
    """The construction guarantee, checked across datasets rather than on one fixture."""
    from bench.metrics import cone_geometry

    for key in ("toy_bee20", "fem_lambda", "gaussian_synth"):
        ds = ds_mod.load(key)
        for method in ("cpg_bee20", "adg"):
            stats = cone_geometry.reach_outside(
                ds, METHODS[method].fit(ds, R=6).generators)
            assert stats["outside_K_full_frac"] == 0.0, (key, method)


def test_excess_is_zero_exactly_when_generators_are_snapshots(bumps):
    """The two directions measure different things and must not be conflated.

    A cone built from selected snapshots is a sub-cone of K_full, so *nothing* in it lies
    outside -- excess is zero. mCPG's is not, so excess must be able to be positive. If
    excess were always zero the metric would be vacuous; if it were never zero it would
    be measuring numerical noise.
    """
    from bench.metrics import cone_geometry

    ds = ds_mod.load("toy_bee20")
    for key in ("cpg_bee20", "adg"):
        e = cone_geometry.excess(ds, METHODS[key].fit(ds, R=8).generators, n_samples=24)
        assert e["excess_mean_err"] == pytest.approx(0.0, abs=1e-9), key
    m = cone_geometry.excess(ds, METHODS["mcpg_ndee22"].fit(ds, R=8).generators,
                             n_samples=24)
    assert m["excess_mean_err"] > 1e-3, "mCPG cone should extend outside K_full here"


def test_cone_hausdorff_is_two_sided(bumps):
    """The summary distance must not hide either direction.

    Reporting coverage alone lets a cone that is far too *large* look perfect, which is
    exactly the mCPG case: its coverage matches CPG's to three digits while a fifth of
    its cone volume sits outside K_full.
    """
    from bench.metrics import cone_geometry

    ds = ds_mod.load("toy_bee20")
    rows = {}
    for key in ("cpg_bee20", "mcpg_ndee22"):
        r = METHODS[key].fit(ds, R=8)
        rows[key] = cone_geometry.evaluate(ds, r, n_samples=24)

    # Nearly identical coverage ...
    assert rows["cpg_bee20"]["cover_mean_err"] == pytest.approx(
        rows["mcpg_ndee22"]["cover_mean_err"], rel=0.05)
    # ... but the two-sided distance separates them, because excess does not.
    assert rows["mcpg_ndee22"]["cone_hausdorff"] > rows["cpg_bee20"]["cone_hausdorff"]
    for key, row in rows.items():
        assert row["cone_hausdorff"] >= row["cover_max_err"] - 1e-12, key
        assert row["cone_hausdorff"] >= row["excess_max_err"] - 1e-12, key


def test_cone_geometry_skips_pathologically_large_cones(bumps):
    """A cone with thousands of generators must be skipped, not silently reported as 0.

    Each sampled statistic costs an NNLS against a dim x R matrix, so the work explodes
    while the answer stops meaning anything -- such a cone is essentially W^+ already.
    The orthant baseline forces this: in tolerance mode on physics (dim 7676) it needs
    R = 5001 at delta = 0.5. That R is the informative result and is still reported; only
    the O(R) geometry is dropped, and the skip is recorded so a reader cannot mistake an
    absent coverage for a perfect one.
    """
    from bench.metrics import cone_geometry

    big = METHODS["orthant"].fit(bumps, R=bumps.dim)
    assert big.R <= cone_geometry.MAX_R_FOR_SAMPLING
    row = cone_geometry.evaluate(bumps, big, n_samples=8)
    assert "cover_mean_err" in row, "small cones must still be measured"

    # Force the guard with a synthetic oversized cone.
    class _Fake:
        R = cone_geometry.MAX_R_FOR_SAMPLING + 1
        generators = np.eye(3)

    skipped = cone_geometry.evaluate(bumps, _Fake(), n_samples=8)
    assert skipped == {"cone_geometry_skipped_R": float(_Fake.R)}
    assert "cover_mean_err" not in skipped, "a skip must not look like zero error"


def test_per_snapshot_column_is_reported_and_differs_from_the_shared_one():
    """Both normalizations must be present, and they must actually differ.

    The shared column divides every snapshot by the largest snapshot norm; the
    per-snapshot one divides each by its own. Where magnitudes spread, a small snapshot
    can be almost entirely unrepresented while the shared column still reads well --
    which is exactly the convention ADG's tolerance bounds. If the two columns ever
    coincided everywhere, the second would be redundant.
    """
    ds = ds_mod.load("physics")          # snapshot norms span ~600x
    row = metrics.precision.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=8))
    for key in ("train_max_rel_err", "train_max_rel_err_persnap",
                "train_mean_rel_err_persnap"):
        assert key in row, key
    assert row["train_max_rel_err_persnap"] > 10 * row["train_max_rel_err"], (
        "on physics the shared column should massively understate the worst snapshot")

    # And the ranking between methods can invert between the two conventions.
    adg = metrics.precision.evaluate(ds, METHODS["adg"].fit(ds, R=8))
    cpg = row
    assert adg["train_max_rel_err"] > cpg["train_max_rel_err"], "shared: ADG behind"
    assert adg["train_max_rel_err_persnap"] < cpg["train_max_rel_err_persnap"], (
        "per-snapshot: ADG should lead, since that is the bound it optimizes")


def test_per_snapshot_ratio_never_divides_by_zero(bumps):
    """A zero-norm column contributes nothing rather than an invented ratio."""
    cols = np.column_stack([bumps.train(), np.zeros(bumps.dim)])
    errs = np.concatenate([np.full(bumps.train().shape[1], 0.5), [0.0]])
    ps = metrics.precision.per_snapshot_rel_errors(errs, cols)
    assert len(ps) == bumps.train().shape[1], "zero column should be dropped, not 0/0"
    assert np.all(np.isfinite(ps))


def test_coverage_improves_with_cardinality(bumps):
    """More generators cannot capture less of K_full, for a nested cone."""
    from bench.metrics import cone_geometry

    prev = None
    for R in (2, 4, 8, 12):
        r = METHODS["cpg_bee20"].fit(bumps, R=R)
        cov = cone_geometry.coverage(bumps, r.generators, n_samples=24)["cover_mean_err"]
        if prev is not None:
            assert cov <= prev + 1e-9, f"coverage worsened from R={R//2} to R={R}"
        prev = cov


def test_cone_geometry_is_reproducible(bumps):
    """The Monte-Carlo coverage estimate must be seeded, or no run is comparable."""
    from bench.metrics import cone_geometry

    r = METHODS["cpg_bee20"].fit(bumps, R=6)
    a = cone_geometry.coverage(bumps, r.generators, n_samples=24)
    b = cone_geometry.coverage(bumps, r.generators, n_samples=24)
    assert a == b


def test_section_extent_never_decreases_when_a_generator_is_added(bumps):
    """K_R is a sub-cone of K_{R+1}, so the section can only grow. Non-negotiable.

    The previous metric measured the section's (R-1)-volume, which FELL as R rose. That
    is impossible for the object being measured, and it happened because the dimension of
    the measurement moved with R -- the statistic was reporting on its own yardstick. Here
    the Gaussian directions are cached from a fixed seed, so the same g are used at every
    R and monotonicity holds per realization: Monte-Carlo noise cannot manufacture a
    decrease, and this test would catch it if the cache were dropped.
    """
    from bench.metrics.cone_geometry import section_extent

    for method in ("cpg_ndee22", "mcpg_ndee22", "adg", "orthant"):
        seen = []
        for R in range(2, 11):
            res = METHODS[method].fit(bumps, R=R)
            seen.append(section_extent(bumps, res.generators)["section_extent"])
        for lo, hi in zip(seen, seen[1:]):
            assert hi >= lo - 1e-12, (method, seen)


@pytest.mark.parametrize("grid", ["sweep_dense/grid.csv"])
def test_extent_rises_for_nested_cones_and_only_nmf_is_exempt(grid):
    """The monotonicity claim, checked against the whole sweep rather than one dataset.

    It is a statement about NESTED cones: a greedy appends to what it has, so K_R is a
    sub-cone of K_{R+1}. NMF is the one method here that does not nest -- it is refitted
    from scratch at each R and its atoms are optimized, not accumulated -- so its cones
    are unrelated across R and its extent may fall. Asserting monotonicity for it would
    be asserting something false; asserting it for the others is what catches the bug
    this metric was rewritten to fix.
    """
    from bench.tabular import num, read_rows

    path = _paths.RESULTS / grid
    if not path.is_file():
        pytest.skip(f"{path} not present; run bench.runner first")

    series: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in read_rows(path):
        if r.get("skip_reason") or r.get("mode") != "cardinality":
            continue
        v, R = num(r, "section_extent"), num(r, "R")
        if not (math.isnan(v) or math.isnan(R)):
            series[(r["dataset"], r["method"])][int(R)] = v

    nested, checked = [], 0
    for (dataset, method), by_R in series.items():
        if method.startswith("nmf"):
            continue
        pts = [by_R[k] for k in sorted(by_R)]
        for lo, hi in zip(pts, pts[1:]):
            checked += 1
            if hi < lo - 1e-9:
                nested.append((dataset, method, lo, hi))
    assert checked > 500, f"only {checked} pairs; the sweep looks too sparse to mean much"
    assert not nested, f"extent fell for a nested cone: {nested[:5]}"


def test_section_extent_is_measured_in_the_dimension_of_the_space():
    """The generators are R rays in R^m, not a basis -- m sets the dimension, not R.

    Two consequences are checked. Padding the ambient space with coordinates the cone
    does not occupy changes the measurement, because the mean is over directions of the
    whole space. And a cone with more generators than the space has dimensions is handled
    normally, where anything keyed to R as a dimension would be nonsense.
    """
    from bench.metrics.cone_geometry import _gaussian_directions

    assert _gaussian_directions(37, 64, 0).shape == (37, 64), "sampled in ambient m"

    rng = np.random.default_rng(5)
    G = np.abs(rng.normal(size=(6, 4))) + 1e-2
    ds_small = Dataset(name="s", snapshots=G)
    from bench.metrics.cone_geometry import section_extent
    assert section_extent(ds_small, ds_small.snapshots)["section_extent"] == pytest.approx(1.0)

    # R > m: 8 rays in 5 dimensions. A measure that took R-1 as its dimension could not
    # even be posed here; this is an ordinary cone with redundant generators.
    G2 = np.abs(rng.normal(size=(5, 8))) + 1e-2
    ds2 = Dataset(name="r", snapshots=G2)
    out = section_extent(ds2, G2[:, :6])["section_extent"]
    assert 0.0 < out <= 1.0 + 1e-9, out


def test_section_extent_reaches_one_for_the_full_cone_and_exceeds_it_only_by_leaving():
    """1 means "as wide as K_full". Above 1 requires generators outside it.

    This is the reading rule for the panel, so it is pinned rather than left to the
    docstring. Measuring K_full against itself must give exactly 1. Canonical axes, which
    are not non-negative combinations of the snapshots, must exceed it.
    """
    from bench.metrics.cone_geometry import section_extent

    ds = ds_mod.load("fem_lambda")
    train = ds.train()
    assert section_extent(ds, train)["section_extent"] == pytest.approx(1.0)
    # A strict sub-cone cannot be wider than the whole.
    assert section_extent(ds, train[:, :5])["section_extent"] <= 1.0 + 1e-9
    axes = np.eye(ds.dim)[:, :5]
    assert section_extent(ds, axes)["section_extent"] > 1.0


def test_section_extent_is_invariant_to_generator_scaling():
    """Vertices are g_i / <g_i, u>, so rescaling a generator cannot move the section.

    The methods disagree about normalization -- ADG normalizes, CPG selects raw snapshots,
    NMF's atoms carry arbitrary scale -- and a width metric that responded to that would
    be comparing conventions rather than cones.
    """
    from bench.metrics.cone_geometry import section_extent

    rng = np.random.default_rng(3)
    G = np.abs(rng.normal(size=(60, 6))) + 1e-3
    ds = Dataset(name="s", snapshots=G)
    ref = section_extent(ds, G)["section_extent"]
    for _ in range(5):
        scaled = G * rng.uniform(1e-3, 1e3, size=G.shape[1])
        assert section_extent(ds, scaled)["section_extent"] == pytest.approx(ref, rel=1e-9)


def test_section_extent_declines_to_measure_an_unbounded_section():
    """A hyperplane that does not cut the cone transversally bounds no region.

    POD is the case that matters: mixed-sign modes make <g, u> zero or negative and the
    section runs to infinity. nan is correct; a finite number would be fabricated.
    """
    from bench.metrics.cone_geometry import section_extent

    ds = ds_mod.load("fem_lambda")
    mixed = np.random.default_rng(4).normal(size=(ds.dim, 4))
    assert math.isnan(section_extent(ds, mixed)["section_extent"])


def test_two_sided_discrepancy_averages_both_directions():
    """It must combine the two directions, not report whichever happens to be larger.

    The panel exists to compare methods, and max() cannot: over the dense sweep it returns
    the MISSED mass in 88% of CPG and ADG cells and the EXCESS mass in 76-95% of mCPG, NMF
    and orthant cells, so the curve silently changes which quantity it plots from one
    method to the next. It also throws the smaller term away, scoring a cone that is both
    somewhat too small and hugely too large identically to one that is only too large.

    The average is built from the MEAN residuals, so this panel is exactly the average of
    the two panels beside it. The old max used the max-over-samples residuals while those
    panels plot the mean-over-samples ones, so it could not be read against them.
    """
    from bench import metrics
    from bench.figures import CONE_PANELS

    columns = [c for c, _y, _s, _t in CONE_PANELS]
    assert "cone_sym_err" in columns and "cone_hausdorff" not in columns

    ds = ds_mod.load("fem_lambda")
    for method in ("cpg_bee20", "mcpg_ndee22", "adg"):
        res = METHODS[method].fit(ds, R=6)
        row = metrics.cone_geometry.evaluate(ds, res)
        assert row["cone_sym_err"] == pytest.approx(
            0.5 * (row["cover_mean_err"] + row["excess_mean_err"]))
        # The strict Hausdorff stays available, and stays a DIFFERENT number.
        assert row["cone_hausdorff"] == pytest.approx(
            max(row["cover_max_err"], row["excess_max_err"]))


def test_two_sided_discrepancy_vanishes_only_when_both_directions_do():
    """Zero must mean the cones coincide, not that one direction happens to vanish.

    This is the property the max was chosen for originally, and averaging keeps it: both
    terms are non-negative, so their mean is zero exactly when each is. A cone spanned by
    every training snapshot misses nothing and exceeds nothing; a strict sub-cone misses
    something even though its excess is still zero, and must not read as coincident.
    """
    from bench import metrics

    ds = ds_mod.load("fem_lambda")
    full = metrics.cone_geometry.evaluate(
        ds, basis_of(ds, ds.train()))
    assert full["cone_sym_err"] == pytest.approx(0.0, abs=1e-9)

    sub = metrics.cone_geometry.evaluate(ds, basis_of(ds, ds.train()[:, :4]))
    assert sub["excess_mean_err"] == pytest.approx(0.0, abs=1e-9), "a sub-cone cannot exceed"
    assert sub["cover_mean_err"] > 1e-3, "but it does miss"
    assert sub["cone_sym_err"] > 0.0, "so it must not read as coincident"


def test_conditioning_axis_is_bounded_at_numerical_singularity():
    """Past 1/eps the condition number is round-off, and must not set the axis.

    On Half-disks of Hertz the Gram goes numerically singular at R=17 for CPG and ADG.
    Beyond that the values wander between 1e16 and 1e19 with no order, and one 4.59e19
    sample at R=22 was compressing the meaningful range -- R=2..16, rising smoothly from
    4.0e2 to 3.1e13 -- into a flat line at zero. The ceiling bounds the AXIS only: the
    points are still plotted and still in the CSV.
    """
    from bench.figures import NUMERICAL_CEILING

    assert NUMERICAL_CEILING["gram_cond"] == pytest.approx(1.0 / np.finfo(float).eps)
    # Every other panel is unbounded; a ceiling is a claim about numerics, not a style.
    from bench.figures import CONE_PANELS, EXTRA_SPLIT_PANELS, PANELS
    for column, _y, _s, _t in PANELS + CONE_PANELS + EXTRA_SPLIT_PANELS:
        if column != "gram_cond":
            assert column not in NUMERICAL_CEILING, column


def test_general_solve_matches_the_reference_when_B_is_identity():
    """The generalized solve must BE rb_online.solve_reduced wherever that one applies.

    solve_reduced hardcodes B_hat = Xi.T @ V, i.e. K = I. That is right for toy_bee20 and
    unusable for obstacle_ndee22, where the multiplier lives on 40 collocation points and
    the displacement on ~200 nodes -- Xi.T @ V does not even have compatible shapes. So a
    generalized version is necessary, and the risk it introduces is drift from the
    reference. This pins them together on the dataset where both can run.
    """
    from rb_online import solve_reduced

    from bench.metrics.online import primal_basis, solve_reduced_general

    ds = ds_mod.load("toy_bee20")
    V = primal_basis(ds)
    Xi = METHODS["cpg_bee20"].fit(ds, R=6).generators
    B = ds.B_of_mu(0)
    assert np.array_equal(B, np.eye(B.shape[0])), "this dataset's B must be the identity"

    for q in np.asarray(ds.test_idx, int)[:6]:
        f, gap = ds.rhs_of_mu(int(q)), ds.gap_of_mu(int(q))
        u_ref, lam_ref = solve_reduced(ds.A, f, gap, V, Xi)
        u_gen, lam_gen = solve_reduced_general(ds.A, f, gap, V, Xi, B)
        assert np.allclose(u_gen, u_ref, rtol=1e-9, atol=1e-12)
        assert np.allclose(lam_gen, lam_ref, rtol=1e-9, atol=1e-12)


def test_online_metric_is_reported_only_where_the_problem_is_available():
    """Solving needs the operator, load and obstacle -- not every source ships them.

    A dataset with only a snapshot matrix must report NOTHING here, rather than a number
    resting on an invented right-hand side. The two that can solve must actually do so.
    """
    from bench import metrics

    for key in ("toy_bee20", "obstacle_ndee22"):
        ds = ds_mod.load(key)
        assert ds.supports_online, key
        row = metrics.online.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=5))
        assert row["online_primal_mean_rel"] > 0.0
        assert row["online_dual_mean_rel"] > 0.0

    for key in ("gaussian_synth", "fem_lambda", "hertz_2d"):
        ds = ds_mod.load(key)
        assert not ds.supports_online, key
        assert metrics.online.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=5)) == {}


def test_solved_error_is_not_the_projection_error():
    """The two are different functionals of the same cone, which is why this metric exists.

    The reduced solve minimizes an energy over W_R^+; it does not project the true
    multiplier onto it. If solved error merely tracked projection error the metric would
    be redundant, so the distinction is asserted rather than assumed.
    """
    from bench import metrics

    ds = ds_mod.load("toy_bee20")
    for R in (4, 6, 8):
        res = METHODS["cpg_bee20"].fit(ds, R=R)
        proj = metrics.precision.evaluate(ds, res)["test_max_rel_err"]
        solved = metrics.online.evaluate(ds, res)["online_dual_mean_rel"]
        assert not math.isclose(proj, solved, rel_tol=1e-3), (R, proj, solved)


def test_full_cone_reference_is_not_advertised_as_a_lower_bound():
    """Methods do come in under it, so it must not be read as a floor.

    Enlarging the cone can overshoot: the solve minimizes over W_R^+ rather than
    projecting onto it. Sweeping R over both supported datasets, a good fraction of
    comparisons land below the full-cone reference -- so the column is named
    ``fullcone``, not ``floor``, and this test keeps that honest.
    """
    from bench import metrics

    below = total = 0
    for key in ("toy_bee20", "obstacle_ndee22"):
        ds = ds_mod.load(key)
        for R in (7, 9, 11):
            for m in ("cpg_bee20", "mcpg_ndee22", "nmf_s0"):
                row = metrics.online.evaluate(ds, METHODS[m].fit(ds, R=R))
                if not row:
                    continue
                total += 1
                below += row["online_primal_mean_rel"] < row["online_primal_fullcone"]
    assert total > 0
    assert below > 0, "if nothing ever beats it, the naming should be revisited"
    assert not any("floor" in k for k in row), "the column must not be called a floor"
