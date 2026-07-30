"""Tests for the benchmark harness itself.

These do **not** re-test the algorithms -- each source repository has its own suite for
that, and ``repos/rb_vi_shared/tests/test_equivalence.py`` already covers the shared
library. What is tested here is the layer this monorepo adds, where the bugs would be
silent and would corrupt every number in the output:

* the orientation and tolerance conversions between the two implementation families,
* that a metric reports a real property rather than an artefact of the harness,
* that skips are recorded rather than dropped.

The riskiest thing in the harness is the convention normalization. A transpose bug
would not raise -- both families accept a 2-D array of either shape -- it would quietly
benchmark ``dim`` snapshots of length ``n`` instead of the reverse. So the first tests
pin orientation from both directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import datasets as ds_mod
from bench import metrics
from bench.adapters import CROSS_FAMILY_PAIRS, METHODS
from bench.instrument import count_solver_calls, summarize
from bench.types import Dataset


# A small, cheap, structurally realistic dataset: non-negative bumps whose support
# moves with the parameter, matching what a contact multiplier looks like. Shared by
# most tests so the suite stays fast.
def _bumps(dim=30, n=18, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, dim)
    cols = []
    for _ in range(n):
        c, w, a = rng.uniform(0.25, 0.75), rng.uniform(0.08, 0.25), rng.uniform(0.5, 2.0)
        cols.append(a * np.clip(1.0 - ((x - c) / w) ** 2, 0.0, None))
    S = np.column_stack(cols)
    idx = np.arange(n)
    test_idx = idx[::4]
    return Dataset(name="bumps", snapshots=S,
                   train_idx=np.setdiff1d(idx, test_idx), test_idx=test_idx)


@pytest.fixture(scope="module")
def bumps() -> Dataset:
    return _bumps()


# ---------------------------------------------------------------------------
# Conventions: the harness's real job
# ---------------------------------------------------------------------------

def test_dataset_is_column_oriented(bumps):
    """Canonical form is (dim, n_snapshots). A transpose bug would not raise."""
    assert bumps.snapshots.shape == (30, 18)
    assert bumps.dim == 30 and bumps.n_snapshots == 18
    assert bumps.train().shape[0] == 30
    assert bumps.test().shape[0] == 30


@pytest.mark.parametrize("key", [k for k, m in METHODS.items() if m.supports_tolerance])
def test_generators_have_snapshot_dimension(bumps, key):
    """Every adapter must return (dim, R), whatever its native orientation.

    This is the test that catches a missing transpose in the ``greedy.core`` adapters:
    those return ``(R, dim)`` natively, so a forgotten ``.T`` yields generators of
    length ``n_train`` instead of ``dim``.
    """
    result = METHODS[key].fit(bumps, delta=0.2)
    assert result.generators.shape[0] == bumps.dim, (
        f"{key}: generators have leading dim {result.generators.shape[0]}, "
        f"expected snapshot dimension {bumps.dim} -- orientation bug")
    assert result.generators.shape[1] == result.R
    assert result.R > 0


def test_bee20_absolute_tolerance_matches_relative_ones(bumps):
    """The [BEE20] adapter's absolute/relative conversion is the one that must hold.

    [BEE20] Eq. (58) is absolute, [NDEE22] Eq. (13) relative. The adapter multiplies the
    canonical ``delta`` by ``max_q ||theta_q||``, which is exactly what makes the two
    criteria the same inequality. If that conversion were dropped, the [BEE20] method
    would silently stop at a wildly different R -- so this pins it at several tolerances.
    """
    for delta in (0.5, 0.3, 0.15, 0.05):
        a = METHODS["cpg_bee20"].fit(bumps, delta=delta)
        b = METHODS["cpg_ndee22"].fit(bumps, delta=delta)
        assert a.R == b.R, f"delta={delta}: R {a.R} vs {b.R} -- tolerance conversion wrong"
        assert a.selected_indices == b.selected_indices


def test_matched_cardinality_is_exact(bumps):
    """Matched-R mode must produce exactly R generators, or comparisons are meaningless."""
    for key in ("cpg_bee20", "cpg_ndee22", "mcpg_ndee22", "cpg_greedy", "mcpg_greedy",
                "adg", "nmf_s0", "pod_control"):
        result = METHODS[key].fit(bumps, R=6)
        assert result.R == 6, f"{key}: asked for R=6, got {result.R}"


# ---------------------------------------------------------------------------
# Structural properties the metrics must detect
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [k for k, m in METHODS.items() if m.cone_method])
def test_cone_methods_preserve_nonnegativity(bumps, key):
    """A non-negative combination of non-negative generators cannot go negative.

    This is the property the entire cone construction exists to preserve ([BEE20] §5).
    For a genuine cone method a violation is a bug, not a result.
    """
    result = METHODS[key].fit(bumps, R=6)
    assert result.generators.min() >= -1e-9, f"{key}: generator left W^+"
    viol = metrics.precision.nonnegativity_violation(
        bumps.train(), result.generators, cone=True)
    assert viol["max_violation"] <= 1e-9, f"{key}: reduced multiplier went negative"


def test_pod_control_actually_violates_nonnegativity(bumps):
    """The negative control must fail the check it exists to fail.

    If POD passed, the dataset could not discriminate between a cone method and an
    unconstrained one, and every non-negativity column in the grid would be vacuous.
    """
    result = METHODS["pod_control"].fit(bumps, R=6)
    viol = metrics.precision.nonnegativity_violation(
        bumps.train(), result.generators, cone=False)
    assert viol["max_violation"] > 1e-6, (
        "POD did not violate non-negativity; the negative control is not controlling")


def test_pod_beats_cone_methods_on_unconstrained_error(bumps):
    """POD is the error floor: no cone of equal cardinality can beat it.

    POD is optimal in the least-squares sense at each cardinality, so a cone method
    scoring *better* would mean the two are not being measured on comparable objects.
    """
    pod = METHODS["pod_control"].fit(bumps, R=6)
    floor = metrics.precision.reconstruction_errors(bumps.train(), pod.generators).max()
    cone = METHODS["cpg_bee20"].fit(bumps, R=6)
    cone_err = metrics.precision.projection_errors(bumps.train(), cone.generators).max()
    assert floor <= cone_err + 1e-9


def test_cone_projection_error_is_monotone_in_R(bumps):
    """Nested cones: adding a generator cannot increase the projection error.

    [BEE20] §5 calls the cones hierarchical, which is one of its arguments for CPG over
    NMF. It is also what makes truncating a fitted basis a valid way to reach a smaller R.
    """
    errs = []
    for R in (2, 4, 6, 8):
        result = METHODS["cpg_bee20"].fit(bumps, R=R)
        errs.append(metrics.precision.projection_errors(
            bumps.train(), result.generators).max())
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])), errs


def test_mcpg_is_better_conditioned_than_cpg(bumps):
    """[NDEE22] claims C5/C6 -- the reason mCPG exists.

    Gram conditioning should improve and ``e_orth`` should rise (generators closer to
    orthogonal to the existing cone). Both are empirical claims about the paper's own
    test case, so this checks the direction on this dataset rather than a magnitude.
    """
    cpg = METHODS["cpg_ndee22"].fit(bumps, delta=0.15)
    mcpg = METHODS["mcpg_ndee22"].fit(bumps, delta=0.15)
    c_cond = metrics.stability.gram_conditioning(cpg.generators)["gram_cond"]
    m_cond = metrics.stability.gram_conditioning(mcpg.generators)["gram_cond"]
    c_orth = metrics.stability.orthogonality_defect(cpg.generators)["e_orth_mean"]
    m_orth = metrics.stability.orthogonality_defect(mcpg.generators)["e_orth_mean"]
    assert m_cond <= c_cond, f"mCPG worse conditioned: {m_cond:.3g} vs {c_cond:.3g}"
    assert m_orth >= c_orth - 1e-12, f"e_orth: mCPG {m_orth:.3f} vs CPG {c_orth:.3f}"


def test_e_orth_respects_its_bound(bumps):
    """[NDEE22] Eq. (41) bounds ``e_orth <= 1``; a value above it means unnormalized input."""
    for key in ("cpg_ndee22", "mcpg_ndee22", "cpg_bee20", "adg"):
        result = METHODS[key].fit(bumps, delta=0.15)
        stats = metrics.stability.orthogonality_defect(result.generators)
        if not np.isnan(stats["e_orth_mean"]):
            assert stats["e_orth_mean"] <= 1.0 + 1e-9, key


# ---------------------------------------------------------------------------
# Batch Normalized Angular-Defect Greedy -- the spec's own invariants
# ---------------------------------------------------------------------------

def test_adg_default_is_the_normalized_form(bumps):
    """The algorithm is defined on S_norm, so the registered `adg` must use it.

    The un-normalized variant is a different algorithm: its stopping rule is one
    absolute threshold `epsilon * max_q ||x_q||` shared by every snapshot, and its
    selection argmax is no longer equivalent to the spec's, because `e_K = sin(theta_K)`
    only holds on unit vectors.
    """
    result = METHODS["adg"].fit(bumps, delta=0.2)
    assert "normalize_snapshots=True" in result.notes
    raw = METHODS["adg_raw"].fit(bumps, delta=0.2)
    assert "normalize_snapshots=False" in raw.notes


def test_adg_terminates_at_the_requested_tolerance(bumps):
    """Output guarantee: r_p* = max_{v in S_norm} e_K(v) <= epsilon.

    On S_norm the error is measured per snapshot against its own unit norm, so this is
    a genuine per-snapshot bound -- unlike a shared absolute threshold, under which a
    small-magnitude snapshot can stay badly represented while the max looks fine.
    """
    from bench.metrics.precision import projection_errors

    S = bumps.train()
    S_norm = S / np.linalg.norm(S, axis=0)
    for eps in (0.5, 0.2, 0.05):
        result = METHODS["adg"].fit(bumps, delta=eps)
        worst = projection_errors(S_norm, result.generators).max()
        assert worst <= eps + 1e-9, (
            f"epsilon={eps}: r_p* = {worst:.4e} exceeds the tolerance")


def test_adg_initializes_with_the_widest_angle_pair(bumps):
    """W_0 is the pair realizing argmax_{i<j} sin(theta(x_i, x_j)).

    For non-negative snapshots every pairwise angle lies in [0, pi/2], where sine is
    increasing, so the widest angle is the smallest cosine.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    rows = np.ascontiguousarray(bumps.train().T)
    model = AngularDefectGreedy(snapshots=rows, epsilon=0.2, normalize_snapshots=True)
    model.compute_phases()

    U = rows / np.linalg.norm(rows, axis=1)[:, None]
    C = U @ U.T
    np.fill_diagonal(C, np.inf)
    i, j = np.unravel_index(int(np.argmin(C)), C.shape)
    assert set(model.initial_pair) == {int(i), int(j)}
    assert model.selected_indices[:2] == list(model.initial_pair)


def test_adg_admits_the_whole_tied_batch(bumps):
    """Each round adds EVERY snapshot attaining theta_max, not just one.

    That is what makes it a batch method, and what the Theorem 3.5 certificate rests on.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    rows = np.ascontiguousarray(bumps.train().T)
    model = AngularDefectGreedy(snapshots=rows, epsilon=0.05, normalize_snapshots=True)
    model.compute_phases()
    assert model.batch_size_history, "no enrichment round ran"
    # 2 from the initial pair, plus one batch per round.
    assert len(model.selected_indices) == 2 + sum(model.batch_size_history)
    cert = model.verify_angular_defect_certificate()
    assert cert["checked"] > 0
    assert cert["violations"] == 0, (
        f"Theorem 3.5 violated in rounds {cert['violation_rounds']} "
        f"(max slack {cert['max_violation']:.3e})")


def test_adg_collapses_coincident_rays():
    """S_norm is a SET: snapshots on one positive ray contribute a single generator.

    This binds precisely because the method is batched. Coincident rays necessarily tie
    at theta_max, so without the set semantics a k-fold repeated direction would enter
    the cone as k generators that add nothing to span_+, inflating n* and making the
    Gram matrix singular.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    base = _bumps(dim=24, n=6, seed=3).snapshots          # (dim, n)
    # Three copies of one direction at different magnitudes: one ray, one generator.
    ray = base[:, 0]
    cols = [ray, 2.0 * ray, 7.5 * ray] + [base[:, k] for k in range(1, 6)]
    rows = np.ascontiguousarray(np.column_stack(cols).T)

    model = AngularDefectGreedy(snapshots=rows, epsilon=1e-3, normalize_snapshots=True)
    model.compute_phases()

    G = np.asarray(model.basis_matrix, float).T           # -> (dim, R)
    U = G / np.linalg.norm(G, axis=0)
    C = np.abs(U.T @ U)
    np.fill_diagonal(C, 0.0)
    assert C.max() < 1.0 - 1e-9, (
        f"cone contains duplicate rays (max off-diagonal cosine {C.max():.12f})")


# ---------------------------------------------------------------------------
# Determinism, instrumentation, agreement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [k for k, m in METHODS.items()
                                 if m.deterministic and m.supports_tolerance])
def test_deterministic_methods_are_deterministic(bumps, key):
    """A drift here means an unseeded RNG, which would make every run unreproducible."""
    row = metrics.stability.determinism(METHODS[key], bumps, delta=0.2)
    assert row["rerun_same_R"] == 1.0
    assert row["rerun_same_order"] == 1.0
    assert row["rerun_generator_drift"] == pytest.approx(0.0, abs=1e-12)


def test_nmf_is_seed_dependent(bumps):
    """NMF's non-determinism is a documented drawback ([BEE20] §5), so it must show up.

    If the seeds agreed, the three NMF rows in the grid would be redundant and the
    across-seed spread reported as stability would be meaningless.
    """
    a = METHODS["nmf_s0"].fit(bumps, R=6)
    b = METHODS["nmf_s1"].fit(bumps, R=6)
    assert not np.allclose(a.generators, b.generators), "NMF seeds gave identical atoms"


def test_instrumentation_counts_nnls(bumps):
    """The counters must see solvers bound at module import, not via scipy's namespace."""
    with count_solver_calls() as counts:
        METHODS["cpg_bee20"].fit(bumps, delta=0.2)
    assert summarize(counts).get("nnls", 0) > 0


def test_instrumentation_restores_originals(bumps):
    """A leaked patch would make every later timing wrong."""
    import rb_vi_common.cone_projection as cp

    before = cp.nnls
    with count_solver_calls():
        pass
    assert cp.nnls is before


def test_mcpg_costs_a_constrained_solve_per_generator(bumps):
    """mCPG's extra cost is [NDEE22] Alg. 2 line 9; CPG has no such solve.

    This pins the performance axis to something structural rather than to wall-clock:
    if the ``minimize`` count for mCPG were zero, line 9 would not be running.
    """
    cpg = METHODS["cpg_ndee22"].fit(bumps, delta=0.2)
    mcpg = METHODS["mcpg_ndee22"].fit(bumps, delta=0.2)
    assert cpg.solver_calls.get("minimize", 0) == 0
    assert mcpg.solver_calls.get("minimize", 0) > 0


@pytest.mark.parametrize("pair", CROSS_FAMILY_PAIRS)
def test_cross_family_implementations_agree(bumps, pair):
    """The claim that justifies keeping duplicate transcriptions.

    Checked at the ``set`` level -- the criterion that makes two cones interchangeable
    downstream -- because generator scaling and tie-breaking are legitimately free.
    """
    a_key, b_key = pair
    a = METHODS[a_key].fit(bumps, delta=0.2)
    b = METHODS[b_key].fit(bumps, delta=0.2)
    row = metrics.agreement.compare(a, b)
    assert row["same_R"] == 1.0, f"{pair}: R {a.R} vs {b.R}"
    assert metrics.agreement.verdict(row, pair).startswith("equivalent"), (
        f"{pair}: verdict={metrics.agreement.verdict(row, pair)} "
        f"set_diff={row['set_max_diff']:.3e}")


def test_agreement_detects_genuine_difference(bumps):
    """The agreement metric must be able to fail, or its passes mean nothing.

    POD is the right probe: it is not a cone at all, so it must differ both
    geometrically *and* in what a cone projection onto it achieves.
    """
    cpg = METHODS["cpg_ndee22"].fit(bumps, R=6)
    pod = METHODS["pod_control"].fit(bumps, R=6)
    row = metrics.agreement.compare(cpg, pod, dataset=bumps)
    assert row["set_max_diff"] > metrics.agreement.SOLVER_ATOL
    assert metrics.agreement.verdict(row, ("cpg_ndee22", "pod_control")).startswith("divergent")


def test_geometric_divergence_is_separated_from_accuracy_loss(bumps):
    """A different-but-equally-good cone must not be reported as a disagreement.

    mCPG's generators are residuals built on earlier generators, so the [UNSPECIFIED]
    line-9 solver choice compounds with R and the two implementations build visibly
    different cones. They achieve the same accuracy, and the verdict has to say so --
    otherwise the grid reads as though one implementation were defective.
    """
    a = METHODS["mcpg_ndee22"].fit(bumps, delta=0.05)
    b = METHODS["mcpg_greedy"].fit(bumps, delta=0.05)
    row = metrics.agreement.compare(a, b, dataset=bumps)
    v = metrics.agreement.verdict(row, ("mcpg_ndee22", "mcpg_greedy"))
    assert v in ("equivalent", "equivalent-within-solver-tol",
                 "same-cone-different-order", "different-cone-same-accuracy"), v
    if row["set_max_diff"] > metrics.agreement.SOLVER_ATOL:
        assert row["max_err_gap"] <= metrics.agreement.ACCURACY_RTOL, (
            "mCPG implementations differ in achieved accuracy, not just in cone shape")


def test_two_exact_cones_report_no_accuracy_gap(bumps):
    """Two cones that both reproduce the training set exactly must show a zero gap.

    At R = n_train the cone contains every training snapshot, so both errors sit at the
    NNLS solver's own accuracy -- 2.5e-7 against 6.8e-9 was observed on membrane_2d,
    both far past exact but a factor of 37 apart. A relative comparison of those reports
    a 97% gap, which would label a genuine agreement a divergence.
    """
    n = bumps.train().shape[1]
    a = METHODS["mcpg_ndee22"].fit(bumps, R=n)
    b = METHODS["mcpg_greedy"].fit(bumps, R=n)
    row = metrics.agreement.compare(a, b, dataset=bumps)
    assert row["train_err_a"] < metrics.agreement.ACCURACY_FLOOR
    assert row["train_err_b"] < metrics.agreement.ACCURACY_FLOOR
    assert row["train_err_gap"] == 0.0, (
        "two exact cones reported an accuracy gap; the floor is below NNLS accuracy")


def test_accuracy_gap_is_absent_without_a_dataset(bumps):
    """``compare`` must stay usable standalone, and must not fake an accuracy verdict."""
    a = METHODS["cpg_ndee22"].fit(bumps, R=6)
    b = METHODS["cpg_greedy"].fit(bumps, R=6)
    row = metrics.agreement.compare(a, b)
    assert "max_err_gap" not in row


# ---------------------------------------------------------------------------
# Dataset and runner plumbing
# ---------------------------------------------------------------------------

def test_dataset_rejects_negative_snapshots():
    """A negative multiplier means a broken HF solve, not a dataset to benchmark on."""
    S = np.abs(np.random.default_rng(0).random((10, 5)))
    S[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        Dataset(name="bad", snapshots=S)


def test_dataset_clips_roundoff_negatives():
    """Solver noise at 1e-18 must not be treated as a sign error.

    ``fem_lambda`` and ``physics`` both arrive with entries at ~1e-12 and ~1e-70.
    """
    S = np.abs(np.random.default_rng(0).random((10, 5)))
    S[0, 0] = -1e-16
    ds = Dataset(name="ok", snapshots=S)
    assert ds.snapshots.min() >= 0.0


def test_no_split_reports_nan_not_zero(bumps):
    """A missing test set must not read as a perfect generalization score."""
    ds = Dataset(name="nosplit", snapshots=bumps.snapshots)
    assert ds.test() is None
    row = metrics.precision.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=5))
    assert np.isnan(row["test_max_rel_err"])


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

    ``physics`` has no train/test split by design, so its test-error column is all nan;
    a test-error panel would come out blank and read as missing data rather than as an
    absent split. The figure module has to fall back to the training error and say so.
    """
    import csv

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
        assert names == ["conditioning", "offline_cost", "orthogonality", "precision"], names
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

    physics carries two numerically zero dual snapshots, for which
    ||theta - Pi(theta)|| / ||theta|| is undefined; ranking must not pick them.
    """
    from bench import reconstruction

    methods = ["cpg_ndee22", "adg", "pod_control"]
    rc = reconstruction.main([
        "--datasets", "toy_bee20", "--methods", *methods,
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


def test_reconstruction_ranking_ignores_zero_norm_snapshots(bumps):
    """A zero snapshot has an undefined relative error and must never be 'worst'."""
    from bench import reconstruction

    S = np.column_stack([bumps.train(), np.zeros(bumps.dim)])
    ds = Dataset(name="withzero", snapshots=S)
    result = METHODS["cpg_ndee22"].fit(ds, R=4)
    rel, _approx = reconstruction._ranking(ds, result, ds.train())
    assert np.isnan(rel[-1]), "zero-norm snapshot should have nan relative error"
    assert int(np.nanargmax(rel)) != len(rel) - 1


def test_decrement_skips_non_consecutive_cardinalities():
    """``e(R+4) - e(R)`` is four generators' worth and must not be plotted as one.

    The metric sweep uses a sparse R grid, so a decrement figure built from it would
    silently mix per-generator steps with multi-generator ones.
    """
    from bench.decrement import decrements_vs_cardinality

    xs, ys = decrements_vs_cardinality([(1, 0.5), (2, 0.4), (3, 0.35), (7, 0.2), (8, 0.19)])
    assert xs == [2, 3, 8], xs
    assert ys == pytest.approx([-0.1, -0.05, -0.01])


def test_decrement_symlog_band_is_relative_to_the_largest_step():
    """Anchoring the linear band to the smallest step puts it at the float-noise floor."""
    from bench.decrement import _symlog_threshold

    assert _symlog_threshold([-1e-2, -1e-3, -1e-16, 0.0]) == pytest.approx(1e-4)
    assert _symlog_threshold([]) > 0
    assert _symlog_threshold([0.0, 0.0]) > 0


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


def test_default_methods_are_one_canonical_version_per_algorithm():
    """Reported outputs carry one implementation per algorithm, from its own paper.

    CPG from [BEE20] Algorithm 2, mCPG from [NDEE22] Algorithm 2, ADG in its normalized
    (standard) form, one NMF seed, and the POD control. The duplicates must stay
    *registered* -- they are the input to the agreement metric, which is what makes the
    merge's retained transcriptions checkable rather than merely asserted.
    """
    from bench.adapters import DEFAULT_METHODS

    assert set(DEFAULT_METHODS) <= set(METHODS)
    assert DEFAULT_METHODS == ("cpg_bee20", "mcpg_ndee22", "adg", "nmf_s0", "pod_control")

    # One CPG, one mCPG, one ADG, one NMF.
    for prefix, expected in (("cpg_", 1), ("mcpg_", 1), ("adg", 1), ("nmf_", 1)):
        n = sum(1 for k in DEFAULT_METHODS
                if k.startswith(prefix) and not (prefix == "cpg_" and k.startswith("mcpg_")))
        assert n == expected, f"{prefix}: {n} in the default set"

    # ADG must be the normalized form, never the non-standard one.
    assert "adg_raw" not in DEFAULT_METHODS

    # The duplicates the agreement metric needs are still reachable.
    for a, b in CROSS_FAMILY_PAIRS:
        assert a in METHODS and b in METHODS


def test_agreement_still_runs_outside_the_default_set(bumps):
    """agreement.csv is built from CROSS_FAMILY_PAIRS, not from --methods.

    Narrowing the reported grid must not silently switch off the cross-implementation
    check, which is the reason the duplicate transcriptions are kept at all.
    """
    from bench.adapters import DEFAULT_METHODS
    from bench.runner import run_agreement

    assert any(b not in DEFAULT_METHODS for _a, b in CROSS_FAMILY_PAIRS)
    rows = run_agreement(bumps, delta=0.2)
    assert len(rows) == len(CROSS_FAMILY_PAIRS)
    assert all(not r["skip_reason"] for r in rows)


def test_registry_is_self_consistent():
    """Registry keys must match the labels the adapters actually return."""
    for key, method in METHODS.items():
        assert method.key == key
        assert method.family in ("rb_vi_common", "greedy.core", "baseline")
    for a, b in CROSS_FAMILY_PAIRS:
        assert a in METHODS and b in METHODS
        assert METHODS[a].family != METHODS[b].family, (
            f"{a}/{b} are not a cross-family pair")


@pytest.mark.parametrize("key", ds_mod.HEAVY)
def test_heavy_datasets_load_when_cvxopt_is_present(key):
    """The 2-D sources are advertised, so they must build -- given their dependency.

    Both import ``cvxopt`` at module scope. Skipping when it is absent keeps the suite
    green on a bare install, but a *different* failure must still fail the test rather
    than be swallowed as "optional".
    """
    pytest.importorskip("cvxopt", reason="heavy-tier datasets need greedy_algos[qp]")
    ds = ds_mod.load(key)
    assert ds.snapshots.min() >= 0.0
    assert ds.dim > 1 and ds.n_snapshots > 1
    assert np.all(np.isfinite(ds.snapshots))


@pytest.mark.parametrize("key", ds_mod.FAST)
def test_fast_datasets_load_and_are_valid(key):
    """Every advertised fast dataset must build and satisfy the Dataset contract."""
    ds = ds_mod.load(key)
    assert ds.snapshots.min() >= 0.0
    assert ds.dim > 0 and ds.n_snapshots > 1
    assert np.all(np.isfinite(ds.snapshots))
    assert ds.scale > 0.0
    if ds.supports_infsup:
        assert ds.A is not None and ds.B_of_mu(0).ndim == 2
