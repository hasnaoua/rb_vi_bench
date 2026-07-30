"""Cross-implementation agreement: do independent transcriptions build the same cone?

This is the metric that justifies the merge. Five CPG-family implementations coexist
across the four repositories, deliberately un-merged, because collapsing them "would
mean silently attributing one paper's convention to the other". The cost of that
decision is that nothing guarantees they agree; the benefit is that agreement becomes
*measurable*. This module does the measuring.

It generalizes ``repos/rb_vi_shared/tests/test_equivalence.py`` -- which checks the two
family-A CPG transcriptions against each other on synthetic bumps -- along two axes it
does not cover: **real datasets** (FEM multipliers, Hertz pressures, 2-D contact) and
**the second implementation family** (``greedy.core``, written independently against the
same papers).

Four criteria, in increasing strictness, following that test:

1. **Same cardinality** ``R`` at matched tolerance.
2. **Same selection order** -- the sequence of chosen parameters. Both papers leave
   ``argmax`` tie-breaking [UNSPECIFIED] (item 2 in [BEE20], item 4 in [NDEE22]), so a
   divergence here is legitimate where snapshots tie, and is reported rather than
   failed.
3. **Same cone as a set** -- projecting arbitrary probes onto both cones gives the same
   answer. This is the criterion that actually matters: it is invariant to generator
   scaling *and* to column ordering, and two cones that pass it are interchangeable in
   every downstream use.
4. **Generators equal up to column normalization** -- the exact relationship the
   conventions predict. [BEE20] Alg. 2 line 6 appends the raw snapshot; [NDEE22]
   Rmk 4.3 appends ``theta_q/||theta_q||``. So a raw/normalized pair should match after
   normalizing, and a raw/raw pair should match outright.

Disagreement is a **finding, not an error**. mCPG's line 9 is solved by SLSQP in
family A and by ``solve_cone_shift_projection`` in family B -- both [UNSPECIFIED]
choices -- so its two implementations may legitimately produce different cones. What
would be a genuine bug is criterion 3 failing between two *CPG* implementations at a
matched tolerance, since CPG has no free numerical choice beyond tie-breaking.
"""

from __future__ import annotations

import numpy as np

from .. import _paths  # noqa: F401
from ..types import BasisResult

from rb_vi_common.cone_projection import project_onto_cone


def _normalized(G: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(G, axis=0)
    return G / np.where(norms > 0, norms, 1.0)


# Two agreement thresholds, because two different questions are being asked.
#
# EXACT is for implementations with no free numerical choice: CPG selects snapshots and
# appends them, so two correct CPGs must agree to floating-point associativity. Anything
# above this between two CPGs is a bug.
#
# SOLVER is for implementations that call different constrained optimizers on the same
# subproblem. mCPG's line 9 is [UNSPECIFIED] in [NDEE22] -- family A solves it with
# SLSQP, family B with ``solve_cone_shift_projection`` -- so their cones can only be
# expected to agree to the accuracy of those solvers. Calling a 1e-7 gap "divergent"
# would report a solver tolerance as a disagreement between papers.
EXACT_ATOL = 1e-10
SOLVER_ATOL = 1e-5

# Relative gap in achieved projection error below which two geometrically different
# cones count as equally good. 1% is well above the observed mCPG spread (the two
# implementations match to 3-5 significant figures) and well below any difference that
# would matter when choosing between them.
ACCURACY_RTOL = 1e-2

# Errors below this (already normalized by the snapshot scale) count as exact, and the
# relative gap between two of them is not computed.
#
# Set against what the benchmark actually asks for, not against machine epsilon. The
# tightest tolerance in the grid is 1e-2, so a cone reproducing its snapshots to 1e-6 is
# four orders better than anything requested -- the residual there is NNLS's own
# accuracy, not a property of the cone. Without this floor, two implementations that
# both reproduce the training set exactly (2.5e-7 against 6.8e-9 on membrane_2d, both
# far past exact, but a factor of 37 apart) report a 97% relative gap, and a genuine
# agreement is labelled a divergence.
ACCURACY_FLOOR = 1e-6


def cones_agree_as_sets(Ga: np.ndarray, Gb: np.ndarray, *,
                        n_probes: int = 12, seed: int = 0,
                        atol: float = SOLVER_ATOL) -> dict[str, float]:
    """Project random non-negative probes onto both cones and compare.

    Probes are drawn non-negative and on the scale of the generators: a cone in the
    non-negative orthant is only distinguishable from another by how it treats vectors
    that point into that orthant, and a probe far outside the snapshots' scale projects
    to a boundary ray for both cones regardless of their difference.
    """
    if Ga.shape[1] == 0 or Gb.shape[1] == 0 or Ga.shape[0] != Gb.shape[0]:
        return {"set_max_diff": float("nan"), "set_agree": float("nan")}
    rng = np.random.default_rng(seed)
    scale = max(float(np.abs(Ga).max()), 1e-300)
    worst = 0.0
    for _ in range(n_probes):
        y = rng.random(Ga.shape[0]) * scale
        pa, _ = project_onto_cone(y, Ga, mass=None)
        pb, _ = project_onto_cone(y, Gb, mass=None)
        worst = max(worst, float(np.abs(pa - pb).max() / scale))
    return {"set_max_diff": worst, "set_agree": float(worst <= atol)}


def accuracy_gap(dataset, a: BasisResult, b: BasisResult) -> dict[str, float]:
    """Do two cones that differ *geometrically* also differ in what they achieve?

    This is the criterion that decides whether a set-level divergence matters. Two cones
    can occupy visibly different positions -- ``set_max_diff`` of 0.2 -- and still
    represent the snapshot manifold equally well, in which case the algorithm simply has
    a non-unique output and neither implementation is wrong.

    That is exactly the mCPG case: its generators are *residuals built on earlier
    generators*, so a small difference in the [UNSPECIFIED] line-9 solve at step ``r``
    propagates into every later generator and compounds with ``R``. CPG has no such
    accumulation -- each generator is an independently selected snapshot -- which is why
    its two implementations stay bit-identical at any ``R``.
    """
    from .precision import projection_errors

    out: dict[str, float] = {}
    if dataset is None:
        return out
    scale = dataset.scale
    for label, cols in (("train", dataset.train()), ("test", dataset.test())):
        if cols is None:
            continue
        ea = float(projection_errors(cols, a.generators).max() / scale)
        eb = float(projection_errors(cols, b.generators).max() / scale)
        out[f"{label}_err_a"] = ea
        out[f"{label}_err_b"] = eb
        # Both exact => no gap. Comparing two zeros relatively is meaningless.
        out[f"{label}_err_gap"] = (
            0.0 if max(ea, eb) <= ACCURACY_FLOOR
            else abs(ea - eb) / max(ea, eb)
        )
    gaps = [v for k, v in out.items() if k.endswith("_err_gap")]
    if gaps:
        out["max_err_gap"] = max(gaps)
    return out


def compare(a: BasisResult, b: BasisResult, *, seed: int = 0, dataset=None) -> dict[str, float]:
    """All four criteria for one pair of fitted cones, plus the accuracy check.

    ``dataset`` is optional only so the geometric criteria can be used standalone; the
    runner always passes it, because without it a set-level divergence cannot be told
    apart from a genuine loss of accuracy.
    """
    row: dict[str, float] = {
        "R_a": float(a.R),
        "R_b": float(b.R),
        "same_R": float(a.R == b.R),
    }

    # Criterion 2 -- compared over the common prefix, so a cardinality difference does
    # not also register as a selection disagreement (they are different failures).
    sa, sb = a.selected_indices, b.selected_indices
    if sa and sb:
        k = min(len(sa), len(sb))
        row["same_order"] = float(sa == sb)
        row["prefix_agree"] = float(sa[:k] == sb[:k])
        row["first_divergence"] = float(
            next((i for i in range(k) if sa[i] != sb[i]), k)
        )
        row["selected_set_agree"] = float(set(sa) == set(sb))
    else:
        # NMF and POD optimize atoms rather than selecting snapshots, so there is no
        # order to compare -- absent, not failed.
        row["same_order"] = float("nan")
        row["prefix_agree"] = float("nan")
        row["first_divergence"] = float("nan")
        row["selected_set_agree"] = float("nan")

    row.update(cones_agree_as_sets(a.generators, b.generators, seed=seed))

    # Criterion 4 -- only meaningful at equal cardinality and equal ordering.
    if a.R == b.R and a.generators.shape == b.generators.shape:
        scale = max(float(np.abs(a.generators).max()), 1e-300)
        row["raw_max_diff"] = float(np.abs(a.generators - b.generators).max() / scale)
        na, nb = _normalized(a.generators), _normalized(b.generators)
        row["normalized_max_diff"] = float(np.abs(na - nb).max())
    else:
        row["raw_max_diff"] = float("nan")
        row["normalized_max_diff"] = float("nan")

    row.update(accuracy_gap(dataset, a, b))
    return row


def verdict(row: dict[str, float], pair: tuple[str, str]) -> str:
    """A one-word reading of a comparison row, for the report.

    The bands separate the three things that can be true of two cones, so that a
    solver tolerance is never reported as a disagreement between transcriptions:

    * ``equivalent`` -- same cone to floating-point associativity, same selection order.
      What two CPG implementations must produce.
    * ``equivalent-within-solver-tol`` -- same cone to the accuracy of the differing
      [UNSPECIFIED] optimizers behind mCPG line 9. Interchangeable downstream.
    * ``same-cone-different-order`` -- the tie-breaking case both papers leave open
      (item 2 in [BEE20], item 4 in [NDEE22]).
    * ``different-cone-same-accuracy`` -- the cones differ geometrically but represent
      the manifold equally well. The algorithm's output is non-unique; neither
      implementation is wrong. This is the observed mCPG case, and calling it
      "divergent" would wrongly imply one of the two is defective.
    * ``divergent`` / ``divergent-cardinality`` -- the cones differ *and* one is
      measurably worse. Between two CPGs, either is a bug.
    """
    diff = row.get("set_max_diff", float("nan"))
    if not np.isfinite(diff):
        return "not-comparable"
    if diff > SOLVER_ATOL:
        # A geometric difference is only a real disagreement if it costs accuracy.
        gap = row.get("max_err_gap", float("nan"))
        if np.isfinite(gap) and gap <= ACCURACY_RTOL:
            return "different-cone-same-accuracy"
        return "divergent-cardinality" if row.get("same_R", 0.0) < 1.0 else "divergent"
    if row.get("same_order", 0.0) < 1.0 and np.isfinite(row.get("same_order", float("nan"))):
        return "same-cone-different-order"
    return "equivalent" if diff <= EXACT_ATOL else "equivalent-within-solver-tol"
