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
    """Canonical directions selected along **mCPG's own iteration** -- the widest cone.

    The counterpart to POD at the other extreme. POD is the *smart but inadmissible*
    reference: optimal in least squares, but its mixed-sign modes cannot build ``W_R^+``
    at all. This is the *admissible but maximally wide* one -- generators are standard
    basis vectors ``e_i``, so non-negativity holds trivially, every pair is exactly 90
    degrees apart (the largest aperture any cone in ``W^+`` can have), and at ``R = dim``
    it is the whole positive orthant.

    **How the directions are chosen matters, and an earlier version got it wrong.**
    Ranking coordinates by global peak activity ``max_q theta_{q,i}`` builds a cone that
    has nothing to do with what the greedy methods are doing, so differences between it
    and mCPG confounded two things at once: coordinate-vs-snapshot generators, and two
    unrelated selection rules.

    Instead this **tracks mCPG's iteration**. mCPG is run to the same cardinality, and at
    each of its steps -- where it would append its cone-constrained residual ``nu_r`` --
    this appends the dominant *canonical* direction of the snapshot mCPG just selected,
    skipping coordinates already taken. Selection order is therefore mCPG's, and the only
    difference is the generator: a canonical axis instead of mCPG's vector. That isolates
    the question the baseline exists to answer -- what does replacing a greedy generator
    with the widest possible admissible direction cost or buy?

    Projection onto ``span_+{e_i : i in S}`` of a non-negative vector keeps ``S`` exactly
    and drops the rest, so the residual is the norm of the discarded coordinates -- closed
    form, no NNLS.

    **Its reported ``calls_total`` is the mCPG ordering run, not the cone assembly.**
    Assembling the cone itself is free; every solver call counted here comes from the mCPG
    pass used to decide *which* axes, and it therefore matches mCPG's count almost exactly
    (159 NNLS + 5 minimize against mCPG's 159 + 5 on toy_bee20 at R=6). So this baseline is
    cheap in the sense that matters -- it needs nothing but the coordinate directions --
    but the offline-cost panel does not show that, and reading its curve as "what a naive
    baseline costs" is wrong. Tracking mCPG is what buys comparability; the price is that
    the cost column measures the tracking, not the baseline.
    """
    from rb_vi_common.cone_greedy import mcpg

    train = dataset.train()
    dim, n = train.shape

    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        cap = dim if R is None else min(int(R), dim)
        # mCPG's selection order. A tiny tolerance with max_R lets the cardinality be set
        # by the cap rather than by a stopping rule, so the two stay step-for-step aligned.
        order = list(mcpg(train, 1e-14, max_R=min(cap, n)).order)

        chosen: list[int] = []
        seen: set[int] = set()
        for q in order:
            if len(chosen) >= cap:
                break
            # Dominant not-yet-taken coordinate of the snapshot mCPG selected at this step.
            ranked = np.argsort(train[:, q])[::-1]
            for i in ranked:
                i = int(i)
                if i not in seen and train[i, q] > 0:
                    chosen.append(i)
                    seen.add(i)
                    break

        # mCPG can stop before `cap` (it caps at the number of snapshots). Top up with the
        # globally most active remaining coordinates so the cone still reaches R, and so
        # the R = dim limit really is the whole orthant.
        if len(chosen) < cap:
            for i in np.argsort(train.max(axis=1))[::-1]:
                if len(chosen) >= cap:
                    break
                i = int(i)
                if i not in seen:
                    chosen.append(i)
                    seen.add(i)

        if delta is not None and R is None:
            # Trim to the shortest prefix meeting the tolerance, using the closed form.
            keep_mask = np.zeros(dim, bool)
            k = len(chosen)
            for j, i in enumerate(chosen, start=1):
                keep_mask[i] = True
                resid = np.sqrt(np.sum(train[~keep_mask, :] ** 2, axis=0)).max()
                if resid / dataset.scale <= float(delta):
                    k = j
                    break
            chosen = chosen[:k]

        G = np.zeros((dim, len(chosen)))
        G[chosen, np.arange(len(chosen))] = 1.0
    seconds = time.perf_counter() - t0

    return BasisResult(
        method="orthant",
        family="baseline",
        paper_tag="",
        generators=G,
        R=len(chosen),
        selected_indices=[int(i) for i in chosen],   # coordinates, not snapshots
        errors=[],
        fit_seconds=seconds,
        solver_calls=summarize(counts),
        normalized_generators=True,
        notes=(f"canonical directions along mCPG's iteration; {len(chosen)} of {dim} "
               f"axes, all mutually orthogonal (90 deg aperture)"),
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
