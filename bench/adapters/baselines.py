"""Two baselines, included for opposite reasons.

**NMF** ([BEE20] §6.4, Eq. 66-69) is the comparison method [BEE20] argues against. It
is a genuine competitor -- its generators are *optimized atoms* free to sit anywhere
in the non-negative orthant, where CPG's are *selected snapshots* -- and [BEE20]'s own
REPRODUCTION_NOTES records the honest result that NMF beats CPG on reconstruction
error at matched cardinality by 10-20% on its toy. Reproducing that comparison across
many more datasets is one of the things this benchmark is for.

NMF is **cardinality-only**: it cannot be run to a tolerance. That is not a limitation
of this adapter but the precise drawback [BEE20] §5 raises against it -- "the user does
not specify an error tolerance but only the cardinality" -- so the runner records the
tolerance mode as unsupported rather than faking an equivalent. It is also
**non-deterministic** in its initialization, which is why the seed is part of the
method label and why ``metrics.stability`` measures across-seed spread for it.

**POD** is a *negative control*, not a competitor. [BEE20] §5 is explicit that POD is
inapplicable to the dual: "the POD is not appropriate to build W_R^+", because its
modes have mixed signs and would destroy ``lambda >= 0``. It is benchmarked anyway
because it supplies two things nothing else does: the best achievable unconstrained
reconstruction error at a given cardinality -- the floor every cone method is measured
against -- and a direct measurement of the sign violation that motivates the entire
cone construction. Its non-negativity check is *expected to fail*, and a run where it
passes means the dataset is too degenerate to discriminate between methods.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _paths  # noqa: F401  -- sys.path side effect
from ..instrument import count_solver_calls, summarize
from ..types import BasisResult

from nmf_baseline import nmf
from rb_online import pod_basis


def fit_nmf(dataset, *, delta=None, R=None, seed=0) -> BasisResult:
    """NMF dual basis, [BEE20] §6.4 Eq. (66)-(69).

    Raises on tolerance mode by design -- see the module docstring.
    """
    if R is None:
        raise ValueError(
            "NMF is cardinality-only ([BEE20] §5: the user 'does not specify an error "
            "tolerance but only the cardinality'); run it in matched-R mode"
        )
    train = dataset.train()
    R_eff = min(int(R), train.shape[1])
    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        W = nmf(train, R_eff, seed=seed)
    seconds = time.perf_counter() - t0
    return BasisResult(
        method=f"nmf_s{seed}",
        family="baseline",
        paper_tag="[BEE20]",
        generators=np.asarray(W, float),
        R=int(np.asarray(W).shape[1]),
        selected_indices=[],           # atoms are optimized, not selected
        errors=[],
        fit_seconds=seconds,
        solver_calls=summarize(counts),
        normalized_generators=False,
        notes=f"multiplicative updates, 200 iters (UNSPECIFIED item 3), seed={seed}",
    )


def fit_nmf_seed1(dataset, *, delta=None, R=None) -> BasisResult:
    return fit_nmf(dataset, delta=delta, R=R, seed=1)


def fit_nmf_seed2(dataset, *, delta=None, R=None) -> BasisResult:
    return fit_nmf(dataset, delta=delta, R=R, seed=2)


def fit_orthant(dataset, *, delta=None, R=None) -> BasisResult:
    """``span_+`` of the R most active coordinate directions -- the naive valid basis.

    The counterpart to POD at the other extreme. POD is the *smart but inadmissible*
    reference: optimal in least squares, but its mixed-sign modes cannot build ``W_R^+``
    at all. This is the *naive but admissible* one: generators are standard basis vectors
    ``e_i``, so non-negativity is preserved trivially, and at ``R = dim`` it is the whole
    positive orthant ``W^+`` -- the largest cone any of these methods is allowed to build.

    It uses no information about the snapshot manifold beyond which coordinates carry
    multiplier mass, so a cone method that cannot beat it is not earning its offline cost.
    It is also the natural upper anchor for the excess axis in ``metrics.cone_geometry``:
    ``W^+`` contains ``K_full`` entirely, so it misses nothing and is maximally too large.

    Coordinates are ranked by peak activity ``max_q theta_{q,i}``, and the projection onto
    ``span_+{e_i : i in S}`` of a non-negative vector is exact on ``S`` and drops
    everything else -- so the residual is just the norm of the discarded coordinates,
    computable in closed form with no NNLS. That makes both modes cheap and exact.
    """
    train = dataset.train()
    dim = train.shape[0]
    activity = train.max(axis=1)
    order = np.argsort(activity)[::-1]

    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        # Residual of snapshot q after keeping the first k coordinates of `order` is the
        # norm of what is dropped -- a suffix sum of squares in that ordering.
        sq = train[order, :] ** 2
        tail = np.concatenate([np.cumsum(sq[::-1], axis=0)[::-1][1:],
                               np.zeros((1, train.shape[1]))])
        worst = np.sqrt(np.max(tail, axis=1))
        if R is not None:
            k = int(min(max(R, 0), dim))
        else:
            rel = worst / dataset.scale
            below = np.flatnonzero(rel <= float(delta))
            k = int(below[0] + 1) if below.size else dim
        keep = order[:k]
        G = np.zeros((dim, k))
        G[keep, np.arange(k)] = 1.0
    seconds = time.perf_counter() - t0

    return BasisResult(
        method="orthant",
        family="baseline",
        paper_tag="",
        generators=G,
        R=k,
        selected_indices=[int(i) for i in keep],   # coordinates, not snapshots
        errors=[],
        fit_seconds=seconds,
        solver_calls=summarize(counts),
        normalized_generators=True,
        notes=f"span_+ of {k} coordinate directions; = W^+ at R = dim ({dim})",
    )


def fit_pod(dataset, *, delta=None, R=None) -> BasisResult:
    """POD of the *dual* snapshots -- the negative control of [BEE20] §5.

    Cardinality-only for the same structural reason as NMF: the SVD gives a spectrum,
    not a cone, and choosing ``N`` from retained energy is [UNSPECIFIED] item 6 in
    [BEE20]'s notes. Kept in matched-R mode only, where it is directly interpretable
    as the unconstrained error floor.
    """
    if R is None:
        raise ValueError("POD is run in matched-R mode only (it has no cone tolerance)")
    train = dataset.train()
    R_eff = min(int(R), min(train.shape))
    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        U = pod_basis(train, R_eff)
    seconds = time.perf_counter() - t0
    return BasisResult(
        method="pod_control",
        family="baseline",
        paper_tag="[BEE20]",
        generators=np.asarray(U, float),
        R=int(np.asarray(U).shape[1]),
        selected_indices=[],
        errors=[],
        fit_seconds=seconds,
        solver_calls=summarize(counts),
        normalized_generators=True,
        notes="NEGATIVE CONTROL: mixed-sign modes, cannot preserve lambda >= 0",
    )
