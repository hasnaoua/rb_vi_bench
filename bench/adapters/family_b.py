"""Adapters for the ``greedy.core`` family -- the class-based implementations.

This family is an independent implementation of CPG and mCPG plus **Angular Defect
Greedy (ADG)**, which has no counterpart in either paper. That independence is the
benchmark's most useful property: where a method exists in both families,
disagreement is a finding rather than a rounding difference, and
``metrics.agreement`` is what checks it.

Two conventions differ from family A and are normalized here (see ``bench.types``):

* **Orientation.** ``greedy.core`` takes snapshots as rows ``(n, dim)`` and returns
  ``basis_matrix`` as rows ``(R, dim)``. Both are transposed at the boundary.
* **Tolerance.** ``epsilon`` is relative, applied as ``epsilon * max_q ||theta_q||`` --
  the same normalizer as [NDEE22] Eq. (13), so it maps to the canonical ``delta``
  directly, with no rescaling.

For matched-cardinality runs this module reuses ``greedy.pipelines.component_sweep``'s
fixed-component fitters rather than reimplementing incremental growth: they are the
same functions ``greedy.pipelines.physics_reduction`` already relies on, so a
matched-R benchmark number is produced by tested code. They return
``(basis_rows, selected_indices, angles)``.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _paths  # noqa: F401  -- sys.path side effect
from ..instrument import count_solver_calls, summarize
from ..types import BasisResult

from greedy.core.angle_defect_greedy import AngularDefectGreedy
from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.pipelines.component_sweep import (
    fit_angle_fixed_components,
    fit_cpg_fixed_components,
    fit_mcpg_fixed_components,
)

# ``ConeGreedy`` defaults differ per subclass (CPG 0.0, mCPG 1e-12, ADG 1e-14); the
# adapters keep each class's own default rather than imposing one, since the value
# interacts with each algorithm's zero tests.
_ZERO_TOL = {"cpg": 0.0, "mcpg": 1e-12, "adg": 1e-14}


def _fit_to_tolerance(cls, dataset, delta, **kwargs):
    """Run a ``ConeGreedy`` subclass to its stopping tolerance."""
    rows = np.ascontiguousarray(dataset.train().T)   # (dim, n) -> (n, dim)
    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        model = cls(snapshots=rows, epsilon=float(delta), **kwargs)
        model.compute_phases()
    seconds = time.perf_counter() - t0
    basis = model.basis_matrix
    if basis is None or basis.size == 0:
        generators = np.empty((dataset.dim, 0))
    else:
        generators = np.ascontiguousarray(np.asarray(basis, float).T)   # -> (dim, R)
    return model, generators, seconds, summarize(counts)


def _fit_to_cardinality(fitter, dataset, R, **kwargs):
    """Run one of the fixed-component fitters to exactly ``R`` generators."""
    rows = np.ascontiguousarray(dataset.train().T)
    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        basis_rows, selected, _angles = fitter(rows, int(R), **kwargs)
    seconds = time.perf_counter() - t0
    basis_rows = np.asarray(basis_rows, float)
    if basis_rows.size == 0:
        generators = np.empty((dataset.dim, 0))
    else:
        generators = np.ascontiguousarray(basis_rows.T)
    return generators, [int(i) for i in selected], seconds, summarize(counts)


def fit_greedy_cpg(dataset, *, delta=None, R=None) -> BasisResult:
    """``greedy.core.CPG`` -- generators are the selected snapshots, un-normalized.

    Same convention as ``family_a.fit_bee20_cpg`` on generator scaling, but a
    *relative* tolerance like [NDEE22]'s. Those two should therefore select the same
    snapshots in the same order at matched tolerance.
    """
    if R is not None:
        generators, selected, seconds, counts = _fit_to_cardinality(
            fit_cpg_fixed_components, dataset, R, zero_tol=_ZERO_TOL["cpg"]
        )
        errors: list[float] = []
    else:
        model, generators, seconds, counts = _fit_to_tolerance(CPG, dataset, delta)
        selected = [int(i) for i in model.selected_indices]
        errors = [float(e) for e in model.relative_residual_history]
    return BasisResult(
        method="cpg_greedy",
        family="greedy.core",
        paper_tag="[BEE20]",
        generators=generators,
        R=int(generators.shape[1]),
        selected_indices=selected,
        errors=errors,
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=False,
        notes="independent CPG implementation; raw generators, relative tolerance",
    )


def fit_greedy_mcpg(dataset, *, delta=None, R=None) -> BasisResult:
    """``greedy.core.mCPG`` -- independent implementation of [NDEE22] Algorithm 2.

    Line 9's constrained shift is solved by ``reduction_common.solve_cone_shift_projection``
    here, against family A's SLSQP. Both are [UNSPECIFIED] choices, so a difference in
    the resulting cone is legitimate and is reported rather than reconciled.
    """
    if R is not None:
        generators, selected, seconds, counts = _fit_to_cardinality(
            fit_mcpg_fixed_components, dataset, R, zero_tol=_ZERO_TOL["mcpg"]
        )
        errors: list[float] = []
    else:
        model, generators, seconds, counts = _fit_to_tolerance(
            mCPG, dataset, delta, zero_tol=_ZERO_TOL["mcpg"]
        )
        selected = [int(i) for i in model.selected_indices]
        errors = [float(e) for e in model.relative_residual_history]
    return BasisResult(
        method="mcpg_greedy",
        family="greedy.core",
        paper_tag="[NDEE22]",
        generators=generators,
        R=int(generators.shape[1]),
        selected_indices=selected,
        errors=errors,
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=True,
        notes="independent mCPG implementation; line 9 via solve_cone_shift_projection",
    )


def fit_greedy_adg(dataset, *, delta=None, R=None, normalize_snapshots=True) -> BasisResult:
    """Batch Normalized Angular-Defect Greedy -- in neither paper; ``greedy_algos``' own.

    Selects on the *angle* between a candidate and its cone projection rather than on
    residual magnitude, and admits every snapshot attaining ``theta_max`` in one batch.
    It has no family-A counterpart, so it is excluded from cross-implementation
    agreement and appears only in the precision, stability and performance grids.

    **The algorithm is defined on the normalized snapshot set** ``S_norm = {x/||x||}``,
    which is why ``normalize_snapshots`` defaults to True here. Two things depend on it:

    * The stopping scale becomes 1, so the tolerance is exactly the spec's
      ``epsilon in (0,1)`` -- a genuine per-snapshot relative check, rather than one
      absolute threshold ``epsilon * max_q ||x_q||`` that a single large-magnitude
      snapshot would set for everything else.
    * More fundamentally, the selection step is only well posed on ``S_norm``. The
      spec takes ``J_p`` to be the argmax over *all* unselected snapshots, while the
      implementation searches the *unresolved* ones; those agree because on unit
      snapshots ``e_K(x) = sin(theta_K(x))``, so projection error and angular defect
      induce the same ordering. Un-normalized, that equivalence fails and the two
      argmaxes can differ.

    Running it un-normalized (``adg_raw``) is therefore a deliberately non-standard
    variant, kept only to show what the shared-threshold stopping rule costs.
    """
    label = "adg" if normalize_snapshots else "adg_raw"
    if R is not None:
        generators, selected, seconds, counts = _fit_to_cardinality(
            fit_angle_fixed_components,
            dataset,
            R,
            zero_tol=_ZERO_TOL["adg"],
            normalize_snapshots=normalize_snapshots,
        )
        errors: list[float] = []
    else:
        model, generators, seconds, counts = _fit_to_tolerance(
            AngularDefectGreedy,
            dataset,
            delta,
            zero_tol=_ZERO_TOL["adg"],
            normalize_snapshots=normalize_snapshots,
        )
        selected = [int(i) for i in model.selected_indices]
        errors = [float(e) for e in model.relative_residual_history]
    return BasisResult(
        method=label,
        family="greedy.core",
        paper_tag="",
        generators=generators,
        R=int(generators.shape[1]),
        selected_indices=selected,
        errors=errors,
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=False,
        notes=f"angle-based selection; normalize_snapshots={normalize_snapshots}",
    )


def fit_greedy_adg_raw(dataset, *, delta=None, R=None) -> BasisResult:
    """ADG on un-normalized snapshots -- non-standard; see ``fit_greedy_adg``."""
    return fit_greedy_adg(dataset, delta=delta, R=R, normalize_snapshots=False)


# ---------------------------------------------------------------------------
# ADG with a pluggable stopping criterion
# ---------------------------------------------------------------------------

def _stop_absolute(errors: list[float], p: int, eps: float) -> bool:
    """The spec's own rule: ``r_p <= epsilon``. Stop once the error is small enough."""
    return errors[p] <= eps


def _stop_relative_change(errors: list[float], p: int, eps: float) -> bool:
    """Stagnation: ``|e(p) - e(p-1)| / e(p-1) <= epsilon``.

    Stop when an enrichment round stops *buying* anything, rather than when the error
    reaches a target. These answer different questions and neither implies the other:

    * The absolute rule needs a meaningful error target. On a problem whose achievable
      accuracy is unknown -- or bounded below by the training set, as every dataset here
      is past some R -- there is no principled epsilon to ask for, and too tight a value
      simply runs to ``R = n_train``.
    * The relative rule needs no target. It stops where the convergence curve flattens,
      which is the cardinality a user would pick by eye off the decrement plot. It is
      scale-free, so the same epsilon transfers across datasets whose errors differ by
      decades.

    The risk is the mirror image: a greedy can stall for one round and then improve
    sharply -- exactly what ADG's batch admission produces when a tied batch is split
    across rounds -- so this rule can stop early on a temporary plateau where the absolute
    rule would not. That is a real tradeoff, not a defect, and is why both are offered
    rather than one replacing the other.
    """
    prev = errors[p - 1]
    if prev <= 0:
        return True          # nothing left to improve on
    return abs(errors[p] - prev) / prev <= eps


#: Stopping rules selectable by ``fit_greedy_adg_with_criterion``. Each takes the error
#: history, the current round, and the tolerance, and returns whether to stop at ``p``.
#: Round 0 is never offered to a rule that needs a predecessor.
STOPPING_CRITERIA = {
    "absolute": _stop_absolute,
    "relative_change": _stop_relative_change,
}

#: Tolerance used to drive ADG to exhaustion before a criterion is applied.
#: Below any achievable residual, so the stopping rule -- not this -- decides where to cut.
_EXHAUSTIVE_EPS = 1e-12


def fit_greedy_adg_with_criterion(dataset, *, delta=None, R=None,
                                  criterion="relative_change") -> BasisResult:
    """ADG stopped by a chosen rule rather than only by ``r_p <= epsilon``.

    ADG is run once to exhaustion and the resulting error history is cut at the first
    round satisfying the criterion. That is exact rather than an approximation: the cone
    is nested and grows by whole batches, and ``residual_basis_sizes[p]`` records the
    basis size at each round, so truncating there yields precisely the cone ADG would have
    held had it stopped at ``p``. Cutting at an arbitrary R would *not* be safe -- it could
    land mid-batch, splitting a set of tied maximizers the algorithm admits together.

    ``R`` overrides the criterion entirely: matched-cardinality mode has no stopping rule,
    so this falls through to plain ADG and the two coincide by construction.
    """
    if R is not None:
        result = fit_greedy_adg(dataset, R=R)
        result.method = f"adg_{criterion}"
        return result

    rule = STOPPING_CRITERIA[criterion]
    model, generators, seconds, counts = _fit_to_tolerance(
        AngularDefectGreedy, dataset, _EXHAUSTIVE_EPS,
        zero_tol=_ZERO_TOL["adg"], normalize_snapshots=True)

    errors = [float(e) for e in model.relative_residual_history]
    sizes = [int(s) for s in model.residual_basis_sizes]
    eps = float(delta)

    stop_at = len(errors) - 1
    start = 0 if criterion == "absolute" else 1
    for p in range(start, len(errors)):
        if rule(errors, p, eps):
            stop_at = p
            break

    keep = sizes[stop_at] if sizes else generators.shape[1]
    keep = max(1, min(keep, generators.shape[1]))
    return BasisResult(
        method=f"adg_{criterion}",
        family="greedy.core",
        paper_tag="",
        generators=generators[:, :keep],
        R=keep,
        selected_indices=[int(i) for i in model.selected_indices[:keep]],
        errors=errors[: stop_at + 1],
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=False,
        notes=(f"ADG stopped by '{criterion}' at round {stop_at} of {len(errors) - 1}; "
               f"R={keep}"),
    )


def fit_greedy_adg_relchange(dataset, *, delta=None, R=None) -> BasisResult:
    return fit_greedy_adg_with_criterion(dataset, delta=delta, R=R,
                                         criterion="relative_change")
