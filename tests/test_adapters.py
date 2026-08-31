"""The adapter layer: two implementation families, made comparable without editing either.

This is where the harness earns its keep and where a bug would be silent. The duplicate
CPG/mCPG transcriptions are deliberately retained (see the top-level README), so the
adapters absorb two convention differences instead of removing them:

* **Orientation.** ``rb_vi_common`` takes snapshots as columns, ``greedy.core`` as rows.
  A transpose bug would not raise -- both accept a 2-D array of either shape -- it would
  quietly benchmark ``dim`` snapshots of length ``n``. So orientation is pinned from both
  directions before anything else.
* **Tolerance.** ``greedy.core``'s and ``[NDEE22]``'s are relative; ``[BEE20]``'s Eq. (58)
  is absolute. The conversion is asserted rather than assumed.

Also here: each algorithm's own spec invariants (ADG's tied batches, its widest-angle
initialization, the stopping criteria), the baselines that bound the comparison from
either side, and the cross-family agreement metric that is the reason the duplicates
stay registered at all.

These do **not** re-test the algorithms themselves -- each source repository has its own
suite, and ``repos/rb_vi_shared/tests/test_equivalence.py`` covers the shared library.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import datasets as ds_mod
from bench import metrics
from bench.adapters import CROSS_FAMILY_PAIRS, METHODS
from bench.instrument import count_solver_calls, summarize
from _fixtures import make_bumps


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

    base = make_bumps(dim=24, n=6, seed=3).snapshots          # (dim, n)
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
    nearly every coordinate (R = 5001 at delta = 0.5 on physics, dim 7676 -- measured
    before physics carried a split, on all 94 columns), and precision alone would then be
    one NNLS solve per snapshot against a 7676 x 5001 matrix.
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


def test_registry_is_self_consistent():
    """Registry keys must match the labels the adapters actually return."""
    for key, method in METHODS.items():
        assert method.key == key
        assert method.family in ("rb_vi_common", "greedy.core", "baseline")
    for a, b in CROSS_FAMILY_PAIRS:
        assert a in METHODS and b in METHODS
        assert METHODS[a].family != METHODS[b].family, (
            f"{a}/{b} are not a cross-family pair")


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


@pytest.mark.parametrize("key", ["toy_bee20", "obstacle_ndee22", "fem_lambda",
                                 "gaussian_synth", "hertz_2d"])
def test_refitting_per_R_equals_one_greedy_run_truncated(key):
    """The cardinality sweep must simulate ONE greedy observed at each iteration.

    The runner fits afresh for every R, which is only legitimate if a greedy stopped at R
    produces exactly the first R generators it would have produced running further. For a
    deterministic greedy that is true -- each iteration appends to the cone it already has
    and never revisits -- so the sweep really is one run observed step by step, and the
    curves are the algorithm's own trajectory rather than a sequence of unrelated bases.

    Nothing checked it, and it is the assumption the whole matched-cardinality comparison
    rests on. If an implementation ever reorders its selections when given a larger budget
    -- a lookahead, a restart, a global re-selection -- every cardinality figure silently
    becomes a plot of unrelated runs, and this is what would catch it.

    NMF is excluded because it genuinely does not have the property: it is refitted from
    scratch at each R and its atoms are optimized rather than accumulated, so its bases at
    successive R are unrelated. That is the drawback [BEE20] §5 raises against it, not a
    defect here, and the extent and decrement figures both show its consequences.
    """
    ds = ds_mod.load(key)
    R_max = min(12, ds.train().shape[1])
    for method in ("cpg_bee20", "mcpg_ndee22", "adg", "adg_momentum", "orthant"):
        full = METHODS[method].fit(ds, R=R_max).generators
        # ADG's trajectory starts at 2: its initialization emits the largest-mutual-angle
        # PAIR, so there is no R=1 state. See test_adg_has_no_R1_state.
        start = 2 if method.startswith("adg") else 1
        for k in range(start, full.shape[1] + 1):
            part = METHODS[method].fit(ds, R=k).generators
            assert part.shape[1] == k, (method, k, part.shape)
            assert np.allclose(part, full[:, :k], atol=1e-10), (
                f"{key}/{method}: fitting to R={k} does not reproduce the first {k} "
                f"generators of the run to R={R_max}; the cardinality sweep is not one "
                "greedy trajectory"
            )


def test_adg_has_no_R1_state_and_refuses_to_invent_one():
    """ADG's first iteration emits TWO generators, so R=1 is off its trajectory.

    AngularDefectGreedy initializes from the pair of snapshots at the largest mutual
    angle, so its cardinalities run 2, 3, 4, ... The upstream fixed-component helper
    answers R=1 anyway, through a separate `components == 1` branch returning the
    largest-norm snapshot -- a different selection rule. On gaussian_synth and
    membrane_2d that snapshot is not in the initialization pair, so the R=1 point was not
    the first step of the curve drawn beside it; on the other datasets it coincides only
    because the max-norm snapshot happens to fall in the pair, which is why the
    discrepancy survived unnoticed.

    The adapter refuses R=1 rather than reporting a point from a different rule.
    """
    ds = ds_mod.load("gaussian_synth")
    for method in ("adg", "adg_momentum", "adg_raw"):
        with pytest.raises(ValueError, match="not a state on its trajectory"):
            METHODS[method].fit(ds, R=1)

    # The upstream branch that produced it still exists and still disagrees -- this is
    # what the adapter is shielding the benchmark from, not a hypothetical.
    from greedy.pipelines.component_sweep import fit_angle_fixed_components

    rows = np.ascontiguousarray(ds.train().T)
    one, idx_one, _ = fit_angle_fixed_components(rows, 1, zero_tol=1e-12)
    _two, idx_two, _ = fit_angle_fixed_components(rows, 2, zero_tol=1e-12)
    assert idx_one[0] not in idx_two, "the two rules agree here; pick another dataset"
