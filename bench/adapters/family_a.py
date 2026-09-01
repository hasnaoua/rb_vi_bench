"""Adapters for the ``rb_vi_common`` family -- the paper-faithful transcriptions.

Three CPG-family functions live in the shared library, and they are *deliberately*
not merged (see the top-level README and both REPRODUCTION_NOTES):

* ``cone_projected_greedy`` -- [BEE20] Algorithm 2 as [BEE20] states it: raw
  (un-normalized) generators, **absolute** tolerance Eq. (58), explicit
  ``||.||_Lambda`` Gram matrix.
* ``cpg`` -- the same algorithm as [NDEE22] Remark 4.3 describes it: generators
  normalized, **relative** tolerance Eq. (13).
* ``mcpg`` -- [NDEE22] Algorithm 2, the modified variant whose generators are
  cone-constrained residuals rather than selected snapshots.

Benchmarking all three is the point, not an oversight: the first two should agree up
to generator scaling, and ``metrics.agreement`` measures whether they do instead of
assuming it.

Snapshots are already in this family's native orientation (columns), so no transpose
happens here -- only the tolerance conversion described in ``bench.types``.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _paths  # noqa: F401  -- sys.path side effect
from ..instrument import count_solver_calls, summarize
from ..types import BasisResult, Dataset

from rb_vi_common.cone_greedy import cone_projected_greedy, cpg, mcpg

# ``cone_projected_greedy`` requires eps_du > 0 ([BEE20] Algorithm 2), and the
# relative variants are meaningless at exactly 0. For matched-cardinality runs the
# stopping rule must not fire at all -- ``max_R`` is what stops the loop -- so the
# tolerance is set below any achievable residual rather than to zero.
_EXHAUSTIVE_TOL = 1e-14


def _require_delta(delta: float | None) -> float:
    """``delta`` is mandatory whenever ``R`` is not given -- say so once.

    Every adapter takes exactly one of the two knobs, and the runner always supplies
    one. Stated here rather than at six call sites, and as a named error rather than
    the ``TypeError`` that ``float(None)`` would raise three frames deeper.
    """
    if delta is None:
        raise ValueError("pass delta= when R= is not given")
    return float(delta)


def _run(fn, snapshots, tol, max_R, mass=None):
    kwargs = {"max_R": max_R} if max_R is not None else {}
    if mass is not None:
        kwargs["mass"] = mass
    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        result = fn(snapshots, tol, **kwargs)
    return result, time.perf_counter() - t0, summarize(counts)


def fit_bee20_cpg(dataset: Dataset, *, delta: float | None = None,
        R: int | None = None) -> BasisResult:
    """[BEE20] Algorithm 2, absolute tolerance Eq. (58).

    The benchmark's canonical knob is a *relative* tolerance, so ``delta`` is
    converted to [BEE20]'s absolute ``eps_du`` by multiplying through by the
    snapshot scale ``max_q ||theta_q||``. [BEE20]'s REPRODUCTION_NOTES flags exactly
    this: its tolerance "is **absolute**, so it must be read against the snapshot
    scale".

    ``mass`` is passed through when the dataset supplies one; with ``mass=None`` the
    ``||.||_Lambda`` of [BEE20] §5 reduces to the Euclidean norm, which is
    [UNSPECIFIED] item 1 in its notes.
    """
    train = dataset.train()
    if R is not None:
        tol = _EXHAUSTIVE_TOL
    else:
        tol = _require_delta(delta) * dataset.scale
    res, seconds, counts = _run(cone_projected_greedy, train, tol, R, mass=dataset.mass)
    # Residuals are absolute here; report them on the same relative footing as the
    # other methods so the error histories are comparable across families.
    errors = [float(r) / dataset.scale for r in res.residuals]
    return BasisResult(
        method="cpg_bee20",
        family="rb_vi_common",
        paper_tag="[BEE20]",
        generators=np.asarray(res.generators, float),
        R=int(res.R),
        selected_indices=[int(i) for i in res.selected_indices],
        errors=errors,
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=False,
        notes="raw generators; absolute tolerance Eq. (58) converted from relative delta",
    )


def fit_ndee22_cpg(dataset: Dataset, *, delta: float | None = None,
        R: int | None = None) -> BasisResult:
    """CPG as [NDEE22] Remark 4.3 describes it: normalized generators, Eq. (13)."""
    tol = _EXHAUSTIVE_TOL if R is not None else _require_delta(delta)
    res, seconds, counts = _run(cpg, dataset.train(), tol, R)
    return BasisResult(
        method="cpg_ndee22",
        family="rb_vi_common",
        paper_tag="[NDEE22]",
        generators=np.asarray(res.generators, float),
        R=int(res.R),
        selected_indices=[int(i) for i in res.order],
        errors=[float(e) for e in res.errors],
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=True,
        notes="Remark 4.3 baseline; normalized generators, relative tolerance Eq. (13)",
    )


def fit_ndee22_mcpg(dataset: Dataset, *, delta: float | None = None,
        R: int | None = None) -> BasisResult:
    """[NDEE22] Algorithm 2 -- mCPG.

    Line 9 solves ``Upsilon_r in K_{r-1} cap (theta_q - W^+)`` with SLSQP, which is
    [UNSPECIFIED] item 2 in [NDEE22]'s notes; it is the reason this method's
    ``minimize`` count is non-zero where CPG's is zero.
    """
    tol = _EXHAUSTIVE_TOL if R is not None else _require_delta(delta)
    res, seconds, counts = _run(mcpg, dataset.train(), tol, R)
    return BasisResult(
        method="mcpg_ndee22",
        family="rb_vi_common",
        paper_tag="[NDEE22]",
        generators=np.asarray(res.generators, float),
        R=int(res.R),
        selected_indices=[int(i) for i in res.order],
        errors=[float(e) for e in res.errors],
        fit_seconds=seconds,
        solver_calls=counts,
        normalized_generators=True,
        notes="Algorithm 2; cone-constrained residual generators (line 9 via SLSQP)",
    )
