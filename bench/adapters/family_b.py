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


def fit_greedy_adg(dataset, *, delta=None, R=None, normalize_snapshots=False) -> BasisResult:
    """Angular Defect Greedy -- in neither paper; ``greedy_algos``' own method.

    Selects on the *angle* between a candidate and its cone projection rather than on
    residual magnitude. It has no family-A counterpart, so it is excluded from
    cross-implementation agreement and appears only in the precision, stability and
    performance grids.

    ``normalize_snapshots`` changes what the stopping tolerance means: with it on, the
    scale is measured on unit-norm snapshots, turning the criterion into a genuine
    per-snapshot relative check instead of one absolute threshold shared across all
    snapshots. It is off by default here so ADG's tolerance is comparable to
    CPG's/mCPG's; the normalized variant is registered as a separate method, because
    comparing the two at a nominally equal ``delta`` would otherwise be misleading.
    """
    label = "adg_norm" if normalize_snapshots else "adg"
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


def fit_greedy_adg_normalized(dataset, *, delta=None, R=None) -> BasisResult:
    return fit_greedy_adg(dataset, delta=delta, R=R, normalize_snapshots=True)
