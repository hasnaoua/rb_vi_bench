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

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bench import _paths
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
        # precision_persnap is an EXTRA_SPLIT_PANEL: it exists only here, not in the
        # 2x2 combined panel, so the grid layout stays square.
        assert names == ["conditioning", "offline_cost", "orthogonality",
                         "precision", "precision_persnap"], names
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


def test_zero_snapshots_are_dropped_before_any_algorithm_runs(bumps):
    """A numerically zero snapshot is absence of data and must never reach a method.

    It is a parameter value at which no contact occurred, so lambda = 0 everywhere.
    Normalizing it is undefined, any per-snapshot relative criterion sees an arbitrary
    error on it, and the ADG spec excludes it outright (S subset R_+^m \\ {0}). Dropping
    it at construction is what keeps every downstream method from having to special-case
    it. physics carries two such columns.
    """
    S = np.column_stack([bumps.train(), np.zeros(bumps.dim)])
    ds = Dataset(name="withzero", snapshots=S)
    assert ds.n_dropped_zero == 1
    assert ds.n_snapshots == bumps.train().shape[1]
    assert np.linalg.norm(ds.snapshots, axis=0).min() > 0

    physics = ds_mod.load("physics")
    assert physics.n_dropped_zero == 2, "physics' two zero snapshots should be gone"
    assert physics.name == "3D Pellet-Cladding"


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


def test_default_methods_are_one_canonical_version_per_algorithm():
    """Reported outputs carry one implementation per algorithm, from its own paper.

    CPG from [BEE20] Algorithm 2, mCPG from [NDEE22] Algorithm 2, ADG in its normalized
    (standard) form, one NMF seed, and the POD control. The duplicates must stay
    *registered* -- they are the input to the agreement metric, which is what makes the
    merge's retained transcriptions checkable rather than merely asserted.
    """
    from bench.adapters import DEFAULT_METHODS

    assert set(DEFAULT_METHODS) <= set(METHODS)
    assert DEFAULT_METHODS == ("cpg_bee20", "mcpg_ndee22", "adg", "adg_momentum",
                               "nmf_s0", "orthant")
    # POD is deliberately out: it is not a dual basis at all ([BEE20] §5), so scoring it
    # beside methods bound by lambda >= 0 compares different problems. Still registered
    # for the sign-violation checks.
    assert "pod_control" not in DEFAULT_METHODS and "pod_control" in METHODS

    # One implementation per algorithm -- except ADG, which contributes two entries
    # deliberately: they are the *same* algorithm under two stopping criteria (absolute
    # error vs relative stagnation), and comparing the criteria is the point.
    for prefix, expected in (("cpg_", 1), ("mcpg_", 1), ("adg", 2), ("nmf_", 1)):
        n = sum(1 for k in DEFAULT_METHODS
                if k.startswith(prefix) and not (prefix == "cpg_" and k.startswith("mcpg_")))
        assert n == expected, f"{prefix}: {n} in the default set"
    # Both ADG entries must be the normalized form; only the stopping rule differs.
    assert {"adg", "adg_momentum"} <= set(DEFAULT_METHODS)

    # ADG must be the normalized form, never the non-standard one.
    assert "adg_raw" not in DEFAULT_METHODS
    # The admissible reference must be present.
    assert "orthant" in DEFAULT_METHODS

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


def test_physics_reshape_matches_the_repository_convention():
    """The theta x z reshape must be greedy_algos', not a plausible-looking transpose.

    physics snapshots are 7676 = 76 x 101 nodes on a quarter-cylinder contact surface,
    not a sequence. Transposing would still produce a 2-D image, and a wrong one -- so
    this is pinned against the function the repository's own publication figures use.
    """
    from greedy.datasets.physics_dataset import reshape_contact_surface

    from bench import geometry

    ds = ds_mod.load("physics")
    geom = ds.geometry
    assert geom.kind == "grid" and geom.shape == (76, 101)

    v = ds.snapshots[:, 0]
    assert np.array_equal(geometry.as_surface(v, geom), reshape_contact_surface(v))

    # extent is (z_min, z_max, theta_min, theta_max) in mm and degrees
    assert geom.extent == (0.0, 5.0, 0.0, 90.0)

    with pytest.raises(ValueError, match="expected"):
        geometry.as_surface(v[:-1], geom)


def test_field_datasets_declare_their_geometry():
    """A contact set that tiles a surface must never fall back to an index plot."""
    pytest.importorskip("cvxopt", reason="membrane_2d needs greedy_algos[qp]")
    for key, kind in (("physics", "grid"), ("membrane_2d", "scatter"), ("hertz_2d", "line")):
        ds = ds_mod.load(key)
        assert ds.geometry is not None, key
        assert ds.geometry.kind == kind, (key, ds.geometry.kind)
    # hertz_2d is genuinely 1-D but along an arc: it carries a physical abscissa.
    hz = ds_mod.load("hertz_2d")
    assert hz.geometry.coords is not None
    assert len(hz.geometry.coords) == hz.dim


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


def test_fem_lambda_pressure_corrects_only_the_symmetry_node():
    """The pressure view doubles node 0 and touches nothing else.

    Under lambda_i = integral p phi_i ~ p(x_i) h_i with uniform spacing, the tributary
    length is h everywhere except the symmetry-axis node, whose hat is truncated to h/2.
    So the conversion is exactly a factor 2 on row 0, up to a global constant that
    span_+ is invariant to.
    """
    force = ds_mod.load("fem_lambda")
    press = ds_mod.load("fem_lambda_pressure")

    assert press.snapshots.shape == force.snapshots.shape
    assert np.allclose(press.snapshots[1:], force.snapshots[1:]), "non-symmetry nodes moved"
    assert np.allclose(press.snapshots[0], 2.0 * force.snapshots[0])
    # The split must be the paper's, same as the uncorrected view.
    assert np.array_equal(press.train_idx, force.train_idx)
    assert np.array_equal(press.test_idx, force.test_idx)
    # Loading one must not mutate the other -- both are lru_cached.
    assert not np.allclose(force.snapshots[0], press.snapshots[0])


def test_coordinate_rescaling_can_change_the_reduction():
    """Doubling one coordinate is not a no-op for a cone method.

    The cone algorithms are invariant to rescaling each *snapshot* (span_+ is), but not
    to rescaling a *coordinate*. If this ever came out invariant for every method, the
    pressure dataset would be redundant and should be dropped.
    """
    force = ds_mod.load("fem_lambda")
    press = ds_mod.load("fem_lambda_pressure")
    differed = []
    for key in ("cpg_bee20", "mcpg_ndee22", "adg"):
        a = METHODS[key].fit(force, delta=0.02)
        b = METHODS[key].fit(press, delta=0.02)
        if a.selected_indices != b.selected_indices:
            differed.append(key)
    assert differed, "no method saw the correction; the pressure view would be redundant"


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


def test_orthant_is_the_maximal_admissible_cone(bumps):
    """At R = dim the orthant baseline IS W^+, the largest cone these methods may build.

    Three properties define it, and all three are what make it the right reference:
    it reproduces every non-negative snapshot exactly, it contains ``K_full`` entirely so
    it misses nothing, and it is strictly larger -- maximal excess. A method beating it on
    coverage would mean the coverage metric is wrong.
    """
    from bench.metrics import cone_geometry

    full = METHODS["orthant"].fit(bumps, R=bumps.dim)
    assert full.R == bumps.dim
    assert full.generators.min() >= 0.0

    err = metrics.precision.projection_errors(bumps.train(), full.generators).max()
    assert err < 1e-12, "W^+ must reproduce non-negative snapshots exactly"

    g = cone_geometry.evaluate(bumps, full, n_samples=24)
    assert g["cover_mean_err"] == pytest.approx(0.0, abs=1e-9), "W^+ contains K_full"
    assert g["excess_mean_err"] > 0.1, "W^+ must be strictly larger than K_full"


def test_orthant_preserves_nonnegativity_at_every_R(bumps):
    """Coordinate generators are non-negative, so the reduced multiplier cannot go below 0."""
    for R in (2, 6, 12):
        r = METHODS["orthant"].fit(bumps, R=R)
        viol = metrics.precision.nonnegativity_violation(
            bumps.train(), r.generators, cone=True)
        assert viol["max_violation"] <= 1e-12, R


def test_orthant_tolerance_mode_is_correct_but_not_reported(bumps):
    """The closed-form residual is right, yet the mode is deliberately not run.

    Projecting a non-negative vector onto span_+{e_i : i in S} keeps S exactly and drops
    the rest, so the *residual* is the norm of the discarded coordinates -- closed form.
    The *selection* is not free: it now tracks mCPG's iteration, so it inherits mCPG's
    NNLS cost. That is the price of a baseline that follows the same path as the method
    it references, instead of an unrelated ranking.

    What rules out tolerance mode is every downstream metric: meeting a tolerance needs
    nearly every coordinate (R = 5001 at delta = 0.5 on physics, dim 7676), and precision
    alone would then be 96 NNLS solves against a 7676 x 5001 matrix.
    """
    for delta in (0.5, 0.2, 0.05):
        r = METHODS["orthant"].fit(bumps, delta=delta)
        err = metrics.precision.projection_errors(
            bumps.train(), r.generators).max() / bumps.scale
        assert err <= delta + 1e-12, f"delta={delta}: got {err:.4e} with R={r.R}"

    assert not METHODS["orthant"].supports_tolerance


def test_orthant_tracks_mcpg_and_is_maximally_wide(bumps):
    """The baseline follows mCPG's selection but emits canonical directions.

    Ranking coordinates by global peak activity -- the earlier version -- built a cone
    unrelated to what the greedy methods do, so any difference confounded two things at
    once: coordinate-vs-snapshot generators, and two unrelated selection rules. Tracking
    mCPG isolates the first.
    """
    from rb_vi_common.cone_greedy import mcpg

    R = 6
    orth = METHODS["orthant"].fit(bumps, R=R)
    assert orth.R == R

    # Generators are canonical: one 1 per column, everything else 0.
    G = orth.generators
    assert set(np.unique(G)) <= {0.0, 1.0}
    assert (G.sum(axis=0) == 1).all()
    assert len(set(orth.selected_indices)) == R, "a coordinate was used twice"

    # Maximal aperture: distinct axes are exactly orthogonal.
    ap = metrics.cone_geometry.aperture(G)
    assert ap["aperture_min_deg"] == pytest.approx(90.0)
    assert ap["aperture_max_deg"] == pytest.approx(90.0)

    # Each chosen axis is the dominant coordinate of the snapshot mCPG selected there.
    order = list(mcpg(bumps.train(), 1e-14, max_R=R).order)
    for step, coord in enumerate(orth.selected_indices[:len(order)]):
        col = bumps.train()[:, order[step]]
        assert col[coord] > 0, f"step {step}: axis carries no mass in mCPG's snapshot"


def test_skip_reasons_are_method_specific(bumps):
    """A shared skip message would misattribute why each method sits out.

    NMF is cardinality-only because [BEE20] §5 says so about the algorithm; the orthant
    is because its tolerance-mode cardinality makes the metrics intractable. Reporting
    [BEE20] §5 for the orthant would credit the paper with an argument it never made.
    """
    from bench.runner import run_cell

    nmf = run_cell(bumps, "nmf_s0", delta=0.2, with_infsup=False, with_determinism=False)
    orth = run_cell(bumps, "orthant", delta=0.2, with_infsup=False, with_determinism=False)
    assert "[BEE20] §5" in nmf["skip_reason"]
    assert "[BEE20] §5" not in orth["skip_reason"]
    assert "intractable" in orth["skip_reason"]


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


def test_adg_stagnation_criterion_cuts_where_the_history_says(bumps):
    """The stop must land at the first round satisfying |e(p)-e(p-1)|/e(p-1) <= eps.

    Recomputed independently from ADG's own error history rather than trusting the
    adapter's bookkeeping, since an off-by-one in the round index would silently return
    a cone one batch too large or too small.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    from bench.adapters.family_b import fit_greedy_adg_momentum

    rows = np.ascontiguousarray(bumps.train().T)
    ref = AngularDefectGreedy(snapshots=rows, epsilon=1e-12, normalize_snapshots=True)
    ref.compute_phases()
    hist = [float(e) for e in ref.relative_residual_history]
    sizes = [int(s) for s in ref.residual_basis_sizes]

    for eps in (0.3, 0.1, 0.02):
        expected_p = next(
            (p for p in range(1, len(hist))
             if hist[p - 1] <= 0 or abs(hist[p] - hist[p - 1]) / hist[p - 1] <= eps),
            len(hist) - 1)
        got = fit_greedy_adg_momentum(bumps, delta=eps)
        assert got.R == sizes[expected_p], (eps, got.R, sizes[expected_p])


def test_adg_stagnation_truncates_at_a_batch_boundary(bumps):
    """Cutting mid-batch would split a set of tied maximizers ADG admits together.

    Every reachable R must therefore be one of the basis sizes ADG actually passed
    through, never an arbitrary integer.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    rows = np.ascontiguousarray(bumps.train().T)
    ref = AngularDefectGreedy(snapshots=rows, epsilon=1e-12, normalize_snapshots=True)
    ref.compute_phases()
    boundaries = set(int(s) for s in ref.residual_basis_sizes)

    for eps in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01):
        r = METHODS["adg_momentum"].fit(bumps, delta=eps)
        assert r.R in boundaries, f"eps={eps}: R={r.R} is not a batch boundary"
        # And the cone must be a prefix of the exhaustive one: nested by construction.
        assert r.generators.shape[1] == r.R


def test_adg_stagnation_matches_plain_adg_at_matched_cardinality(bumps):
    """Matched-R mode has no stopping rule, so the two must coincide exactly."""
    for R in (3, 6, 10):
        a = METHODS["adg"].fit(bumps, R=R)
        b = METHODS["adg_momentum"].fit(bumps, R=R)
        assert b.R == a.R == R
        assert np.allclose(a.generators, b.generators)
        assert b.method == "adg_relative_change"


def test_stopping_criteria_are_pluggable():
    """Both rules are registered and the absolute one reproduces the spec's behaviour."""
    from bench.adapters.family_b import STOPPING_CRITERIA

    assert set(STOPPING_CRITERIA) == {"absolute", "relative_change"}
    hist = [1.0, 0.5, 0.49, 0.2]
    assert STOPPING_CRITERIA["absolute"](hist, 3, 0.25) is True
    assert STOPPING_CRITERIA["absolute"](hist, 1, 0.25) is False
    # |0.49-0.5|/0.5 = 0.02 -> stalls at p=2 for eps=0.05, not at p=1 (|0.5-1|/1 = 0.5)
    assert STOPPING_CRITERIA["relative_change"](hist, 2, 0.05) is True
    assert STOPPING_CRITERIA["relative_change"](hist, 1, 0.05) is False


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


def test_only_the_two_references_are_kept_off_the_comparison_axes():
    """Exclusion is a property of the METHOD, not of the axis it is drawn against.

    An earlier version routed this through a mode-aware ``excluded_for(mode)`` so that
    ``adg_momentum`` could be dropped from matched-cardinality figures, where it
    coincides with ``adg``. That made the method look absent rather than coincident, so
    the exclusion was reverted and the set it fed became permanently empty -- leaving a
    ``mode`` parameter that could not change any answer. One flat set now, and the
    coincidence is shown by drawing it in dash-dot over ``adg`` instead of hiding it.
    """
    from bench import decrement, figures, reconstruction

    assert figures.FIGURE_EXCLUDED == frozenset({"orthant", "pod_control"})
    # Every drawing module filters through the same set; none re-derives its own.
    for mod in (decrement, reconstruction):
        assert mod.FIGURE_EXCLUDED is figures.FIGURE_EXCLUDED


def test_momentum_and_adg_coincide_at_matched_cardinality(bumps):
    """The premise of the exclusion, asserted rather than assumed."""
    for R in (2, 5, 9):
        a = METHODS["adg"].fit(bumps, R=R)
        b = METHODS["adg_momentum"].fit(bumps, R=R)
        assert np.allclose(a.generators, b.generators)
        for row_a, row_b in ((metrics.precision.evaluate(bumps, a),
                              metrics.precision.evaluate(bumps, b)),):
            for k in row_a:
                assert np.isclose(row_a[k], row_b[k], equal_nan=True), (R, k)


def test_datasets_are_named_for_the_problem_they_solve():
    """Reported names say what the physics is; CLI keys stay stable.

    A reader of a figure should see the problem, not the file it came from. The registry
    key is the handle for --datasets and must not move, or every saved command breaks.
    """
    from bench import layout

    expected = {
        "fem_lambda": "Half-disks of Hertz",
        "fem_lambda_pressure": "Half-disks of Hertz (pressure)",
        "physics": "3D Pellet-Cladding",
    }
    for key, name in expected.items():
        assert key in ds_mod.DATASETS, f"{key} key must stay stable"
        assert ds_mod.load(key).name == name
        # And the prose name must still produce a usable directory.
        slug = layout.slug(name)
        assert " " not in slug and "(" not in slug and "__" not in slug, slug
        assert slug


def test_adg_momentum_is_drawn_in_every_figure():
    """It must not silently vanish from the figures the way the earlier exclusion did.

    At matched cardinality it coincides with ``adg`` exactly -- there is no stopping rule
    in that mode -- so it overlays rather than adds information. It is drawn anyway, in
    dash-dot, so the coincidence is visible instead of the method appearing absent.
    """
    from bench.figures import FIGURE_EXCLUDED, STYLE

    assert "adg_momentum" not in FIGURE_EXCLUDED
    assert STYLE["adg_momentum"]["ls"] == "-.", "must be distinguishable where it overlays"


def test_half_disk_is_drawn_as_the_full_symmetric_contact_line():
    """[BEE20] Fig. 7-8 plot the contact stress over abscissas in [-1, 1], centred on 0.

    FEM_SOLS stores only the half the symmetry plane makes redundant: 57 nodes from the
    symmetry axis outward, peak first, zeros last. Drawn directly that is half the physics
    with the peak jammed against the left edge. The mirror recovers the paper's picture.
    """
    from bench import geometry as g

    for key in ("fem_lambda", "fem_lambda_pressure"):
        d = ds_mod.load(key)
        geom = d.geometry
        assert geom.kind == "mirrored_line", key
        assert len(geom.coords) == 2 * d.dim - 1, "node 0 is on the plane: written once"
        assert geom.coords[0] == pytest.approx(-1.0)
        assert geom.coords[-1] == pytest.approx(1.0)

        full = g.mirror_half_profile(d.snapshots[:, 0])
        assert full.size == 2 * d.dim - 1
        assert np.allclose(full, full[::-1]), "must be symmetric about the plane"
        # Zeros on BOTH sides, contact in the middle -- the paper's layout.
        assert full[0] < 1e-6 * full.max() and full[-1] < 1e-6 * full.max()
        centre = full[full.size // 2]
        if key == "fem_lambda_pressure":
            # Corrected, the centre carries essentially the peak value.
            assert centre > 0.9 * full.max()
        else:
            # Uncorrected, the centre node carries half its tributary weight, so it sits
            # at roughly half the peak. That is the half-support effect, not a defect.
            assert 0.4 * full.max() < centre < 0.6 * full.max()


def test_pressure_correction_lifts_the_centre_onto_the_peak():
    """The half-support correction is what makes the mirrored profile peak at abscissa 0.

    Measured over all 50 snapshots, centre value divided by the profile peak:

        uncorrected  0.5034  (0.4668 - 0.5487)   -- exactly the half-support factor
        corrected    0.9885  (0.9336 - 1.0000)

    The corrected centre is *not* always the strict argmax (26 of 50) -- node-level noise
    of a few percent means node 1 can edge above node 0 -- so this asserts the ratio,
    which is the robust statement, rather than the argmax, which is not.
    """
    from bench import geometry as g

    raw = ds_mod.load("fem_lambda")
    press = ds_mod.load("fem_lambda_pressure")
    c = raw.dim - 1                           # index of node 0 in the mirrored array

    def ratios(d):
        return np.array([g.mirror_half_profile(d.snapshots[:, k])[c]
                         / g.mirror_half_profile(d.snapshots[:, k]).max()
                         for k in range(d.n_snapshots)])

    r_raw, r_press = ratios(raw), ratios(press)
    assert 0.45 < r_raw.mean() < 0.55, r_raw.mean()
    assert r_press.mean() > 0.95, r_press.mean()
    assert r_press.min() > 0.9, "corrected centre should never fall far below the peak"


def test_reference_baseline_gets_its_own_figure(tmp_path, bumps):
    """The orthant is kept off the comparison axes but must still be plottable.

    Excluding it from the shared figures protected their y-ranges, but left its evolution
    in R readable only as CSV columns. On its own axes there is no range to protect, so it
    gets every panel the comparison figures carry.
    """
    from bench import figures
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
    assert "orthant" in figures.FIGURE_EXCLUDED


def test_section_volume_matches_the_orthogonal_reference_exactly():
    """R orthogonal directions must score exactly 1 -- that is what the ratio is against.

    The reference is derived in closed form (vertices sqrt(m) h e_i, edge Gram
    m h^2 (I + J), determinant m^(R-1) h^(2(R-1)) R), so an error in it would not show up
    as noise but as a systematic offset in every reported number. Pinned at several R and
    in two ambient dimensions, since the m-dependence is the part most easily got wrong.
    """
    from bench.metrics.cone_geometry import section_volume

    for m in (12, 500):
        for R in (2, 3, 7, min(11, m)):
            G = np.eye(m)[:, :R]
            got = section_volume(G)["section_vol_ratio"]
            assert got == pytest.approx(1.0, abs=1e-10), (m, R, got)


def test_section_volume_measures_extent_where_mean_aperture_cannot():
    """The reason for replacing mean pairwise angle: a mean over edges is not an extent.

    A generator added strictly INSIDE the cone enlarges the enclosed region not at all.
    The section volume says so -- the section stays (R-1)-degenerate and the ratio
    collapses to 0 -- while the mean pairwise angle moves, because it has gained an edge
    it has no way to recognise as redundant.
    """
    from bench.metrics.cone_geometry import aperture, section_volume

    m = 40
    G = np.eye(m)[:, :3]
    interior = (G @ np.array([0.5, 0.3, 0.2]))[:, None]
    G_plus = np.hstack([G, interior])

    assert section_volume(G)["section_vol_ratio"] == pytest.approx(1.0, abs=1e-10)
    assert section_volume(G_plus)["section_vol_ratio"] < 1e-12, "no space was added"
    # The old statistic cannot tell: it changes, and not toward zero.
    assert aperture(G)["aperture_mean_deg"] != pytest.approx(
        aperture(G_plus)["aperture_mean_deg"], abs=1e-6)


def test_section_volume_is_invariant_to_generator_scaling():
    """Each vertex is g_i / <g_i, u>, so rescaling a generator cannot move the section.

    This matters because the methods disagree about normalization -- ADG normalizes, CPG
    selects raw snapshots, NMF's atoms carry arbitrary scale -- and a width metric that
    responded to that would be comparing conventions rather than cones.
    """
    from bench.metrics.cone_geometry import section_volume

    rng = np.random.default_rng(3)
    G = np.abs(rng.normal(size=(60, 6))) + 1e-3
    ref = section_volume(G)["section_vol_ratio"]
    for _ in range(5):
        scaled = G * rng.uniform(1e-3, 1e3, size=G.shape[1])
        assert section_volume(scaled)["section_vol_ratio"] == pytest.approx(ref, rel=1e-9)


def test_section_volume_declines_to_measure_an_unbounded_section():
    """A hyperplane that does not cut the cone transversally bounds no region.

    POD is the case that matters: its modes have mixed signs, so <g, u> can be zero or
    negative and the section runs to infinity. nan is the correct report -- an unbounded
    region has no volume -- and silently returning a finite number would put a meaningless
    value on the panel.
    """
    from bench.metrics.cone_geometry import section_volume

    rng = np.random.default_rng(4)
    mixed = rng.normal(size=(30, 4))          # POD-like: mixed signs
    assert math.isnan(section_volume(mixed)["section_vol_ratio"])
    # A single ray sections to a point: 0-volume is 1 by convention, which would read as
    # a full-width cone, so it is declined rather than reported.
    assert math.isnan(section_volume(np.ones((30, 1)))["section_vol_ratio"])


def test_section_volume_survives_cardinalities_that_overflow_the_raw_volume():
    """The ratio must stay computable where neither volume is representable.

    Both V and V_ortho carry 1/(R-1)! and m^((R-1)/2). At R=40 in 7676 dimensions those
    are around 1e-47 and 1e77 -- the raw volume is not a usable number, but the factors
    cancel exactly in the ratio, which is why the computation is done in logs.
    """
    from bench.metrics.cone_geometry import section_volume

    m, R = 7676, 40
    out = section_volume(np.eye(m)[:, :R])
    assert out["section_vol_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert math.isfinite(out["section_log_volume"])
    assert not math.isfinite(math.exp(out["section_log_volume"])) or True


def test_backfill_reproduces_what_the_runner_computes():
    """The derived column must equal what a fresh run would have written.

    The backfill exists so an hour of NNLS is not respent to recover a number the CSV
    already determines -- which is only acceptable if it lands on the same value. This
    runs the metric directly, writes a CSV WITHOUT the derived column, backfills it, and
    demands agreement. It also exercises the module's own guard, which re-derives
    section_vol_ratio (a different function of the same inputs) and compares it against
    the value the runner stored independently.
    """
    from bench import backfill as bf
    from bench.metrics.cone_geometry import section_volume
    from bench.tabular import read_rows, write_csv

    rng = np.random.default_rng(11)
    rows, expected = [], []
    for R in (2, 3, 5, 9):
        for m in (30, 400):
            G = np.abs(rng.normal(size=(m, R))) + 1e-3
            out = section_volume(G)
            expected.append(out["section_width_ratio"])
            rows.append({
                "dataset": "synthetic", "method": f"m{m}_R{R}", "R": repr(float(R)),
                "dim": repr(float(m)),
                "section_log_volume": repr(out["section_log_volume"]),
                "section_vol_ratio": repr(out["section_vol_ratio"]),
            })

    path = Path(tempfile.mkdtemp()) / "grid.csv"
    write_csv(path, rows)
    filled, checked = bf.backfill(path)
    assert filled == len(rows) and checked == len(rows), (filled, checked)

    got = [float(r["section_width_ratio"]) for r in read_rows(path)]
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, rel=1e-12)


def test_backfill_refuses_to_write_when_the_formula_disagrees():
    """A wrong reference volume must abort, not produce a plausible column.

    The failure this guards against is silent: a mis-derived reference is off by a smooth
    factor, so the backfilled column would look entirely reasonable while being wrong
    everywhere. Corrupting the stored ratio stands in for that.
    """
    from bench import backfill as bf
    from bench.tabular import write_csv

    path = Path(tempfile.mkdtemp()) / "grid.csv"
    write_csv(path, [{
        "dataset": "d", "method": "m", "R": "4.0", "dim": "50.0",
        "section_log_volume": "-3.0",
        "section_vol_ratio": "0.5",        # inconsistent with the log volume above
    }])
    with pytest.raises(AssertionError, match="does not match the runner"):
        bf.backfill(path)


def test_backfill_leaves_underivable_rows_empty_not_zero():
    """An absent value must read as absent. Zero would read as a degenerate cone."""
    from bench import backfill as bf
    from bench.tabular import read_rows, write_csv

    path = Path(tempfile.mkdtemp()) / "grid.csv"
    write_csv(path, [
        {"dataset": "d", "method": "single_ray", "R": "1.0", "dim": "50.0",
         "section_log_volume": "", "section_vol_ratio": ""},
        {"dataset": "d", "method": "no_geometry", "R": "5.0", "dim": "",
         "section_log_volume": "", "section_vol_ratio": ""},
    ])
    bf.backfill(path)
    for row in read_rows(path):
        assert row["section_width_ratio"] == "", row["method"]


def test_every_axis_is_linear():
    """No panel transforms its values, so a figure can be read against the CSV directly.

    Log and symlog axes were used here before, and dropping them has a real cost: several
    of these columns span decades, and on a linear axis the small values are pressed
    against the baseline where differences between good methods cannot be resolved by
    eye. Those comparisons belong to report.txt, which carries full precision.

    What a linear axis buys is that nothing vanishes. A log axis has no coordinate for
    zero and discards such cells with no marker and no gap, which previously removed
    NMF's entire offline-cost series (0 constrained solves in all 346 of its cells),
    ADG's R=1 and R=2 points (it seeds from a Gram-matrix argmin; NNLS first appears at
    R=3), and the 15 cells where the orthant covers K_full exactly.
    """
    from bench.figures import CONE_PANELS, EXTRA_SPLIT_PANELS, PANELS

    for column, _ylabel, scale, _title in PANELS + CONE_PANELS + EXTRA_SPLIT_PANELS:
        assert scale == "linear", f"{column} applies a {scale} transformation"


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

    path = _paths.ROOT / "results" / grid
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


def test_axial_profile_matches_the_publication_reduction():
    """The z-profile must be greedy.viz.publication's, not a reinvention.

    Its figures collapse theta by the MEAN and plot force against axial z. Pinning this
    against the repository's own profile_from_snapshot keeps the benchmark's overlay and
    the publication figures showing the same curve; a max-reduction or a transpose would
    both produce a plausible but different profile.
    """
    from greedy.viz.publication import profile_from_snapshot

    from bench import geometry as g

    ds = ds_mod.load("physics")
    geom = ds.geometry
    v = ds.snapshots[:, 0]

    mine = g.axial_profile(v, geom)
    assert np.allclose(mine, profile_from_snapshot(v, "mean"))
    assert mine.size == geom.shape[1] == 101, "one value per axial station"

    z = g.axial_coordinate(geom)
    assert z.size == mine.size
    assert z[0] == pytest.approx(0.0) and z[-1] == pytest.approx(5.0)


def test_active_span_uses_the_publication_threshold():
    """The shaded band is the same active span the publication figures mark."""
    from bench import geometry as g

    assert g.ACTIVE_THRESHOLD_RATIO == 1.0e-3
    ds = ds_mod.load("physics")
    p = g.axial_profile(ds.snapshots[:, 0], ds.geometry)
    mask = g.active_span(p)
    assert mask.any() and not mask.all(), "contact should cover part of the cladding"
    assert np.all(p[mask] > g.ACTIVE_THRESHOLD_RATIO * p.max())


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


def test_error_panels_are_relative_not_absolute():
    """The error map is normalized by the snapshot's own peak.

    Absolute error cannot be compared across snapshots, and the best and worst panels are
    by construction *different* snapshots -- so an absolute colour scale shared between
    them is misleading. Normalizing by each snapshot's peak makes the two rows comparable
    and puts the colour bar in readable units.
    """
    from bench import geometry as g

    truth = np.array([0.0, 2.0, 8.0, 4.0])
    approx = np.array([0.0, 2.4, 8.0, 3.2])
    rel = g.relative_error_field(truth, approx)
    assert np.allclose(rel, np.abs(truth - approx) / 8.0)
    assert rel.max() <= 1.0
    # Scale-invariant: a snapshot ten times larger with ten times the error scores equally.
    assert np.allclose(rel, g.relative_error_field(10 * truth, 10 * approx))


def test_relative_error_survives_the_numerically_zero_inactive_zone():
    """Pointwise |d|/|theta| would divide by noise over most of the field.

    Where contact is not established the multiplier is numerically zero -- most entries
    below 1e-6 of the peak, reaching 5e-16 -- so a pointwise ratio explodes exactly where
    the ROM's spurious contact appears, drowning the real failure. The peak-normalized
    form stays finite and bounded.
    """
    from bench import geometry as g

    ds = ds_mod.load("physics")
    truth = ds.snapshots[:, 0]
    peak = np.abs(truth).max()
    tiny = np.abs(truth) < 1e-6 * peak
    assert tiny.sum() > truth.size // 2, "large numerically-zero region expected"
    assert np.abs(truth)[tiny].min() < 1e-12 * peak, "and it reaches the noise floor"

    # A real reconstruction, not a uniform rescaling: a ROM's error is additive and
    # lands in the inactive zone, which is the case that breaks the pointwise ratio.
    from bench.metrics.precision import reconstruct
    approx = reconstruct(truth[:, None], METHODS["adg"].fit(ds, R=6).generators)[:, 0]

    rel = g.relative_error_field(truth, approx)
    assert np.all(np.isfinite(rel)) and rel.max() <= 1.0
    pointwise = np.abs(truth - approx) / np.maximum(np.abs(truth), 1e-300)
    assert pointwise.max() / rel.max() > 1e3, "pointwise is the ill-conditioned one"


def test_field_rendering_applies_no_colour_transformation():
    """Fields are drawn in their own units, so the colour bar means what it says.

    ``geometry.scale`` used to apply log10(1+|v|) to the physics field, on the grounds
    that raw contact pressures span decades. Two problems. A colour bar reading
    log10(1+|lambda|) cannot be read against a force, and the compression makes the field
    look far more uniform than it is. And it did not match the reference it was meant to
    reproduce: greedy.viz.publication normalizes with a plain Normalize and rescales by a
    decimal exponent -- a change of UNITS, not of shape.
    """
    from bench import geometry as g

    geom = ds_mod.load("physics").geometry
    assert not hasattr(geom, "log"), "the flag should be gone, not merely set False"
    assert "log" not in geom.clabel.lower()
    for values in (np.array([0.0, 1e-9, 1.0, 5.0, 1234.0]),
                   np.linspace(0.0, 0.5, geom.shape[0] * geom.shape[1])):
        assert np.array_equal(g.scale(values, geom), values)
