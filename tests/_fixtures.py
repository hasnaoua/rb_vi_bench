"""The one synthetic dataset most tests run against.

Non-negative bumps whose support moves with the parameter -- structurally what a contact
multiplier looks like, at a size where a full method sweep costs milliseconds. Almost
every test uses it, so the suite stays fast enough to run on every edit.

It lives in its own module rather than in ``conftest.py`` because two things need it in
two different ways: ``conftest`` wraps it as the session ``bumps`` fixture, and a handful
of tests need to build a *variant* (a different dimension, a different seed) which a
fixture cannot express.
"""

from __future__ import annotations

import numpy as np

from bench.types import Dataset


def make_bumps(dim: int = 30, n: int = 18, seed: int = 0) -> Dataset:
    """A small, cheap, structurally realistic non-negative dataset."""
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


def basis_of(dataset, generators):
    """A ``BasisResult`` wrapping explicit generators, for metric-level tests.

    The metrics take a fitted result, but several of them have to be pinned against a
    cone whose generators are *chosen*, not fitted -- the full training cone, a strict
    sub-cone, a deliberately scaled one. This wraps bare columns in the shape the
    metrics expect without going through an adapter.
    """
    from bench.types import BasisResult

    return BasisResult(
        method="explicit", family="test", paper_tag="", generators=np.asarray(generators),
        R=int(np.asarray(generators).shape[1]), selected_indices=[], errors=[],
        fit_seconds=0.0, solver_calls={}, normalized_generators=False, notes="",
    )


__all__ = ["basis_of", "make_bumps"]


def make_solvable_obstacle(dim: int = 24, n: int = 12, seed: int = 0):
    """A tiny 1-D obstacle problem complete enough to SOLVE, not just to score a cone.

    ``metrics.online`` needs what a snapshot matrix cannot supply: the operator, the load
    and the gap. The two shipped datasets that carried them were removed from the
    registry, so the metric's tests build their own rather than lapse -- the capability is
    generic (any ``Dataset`` given ``A``, ``B_of_mu``, ``rhs_of_mu``, ``gap_of_mu`` and
    primal snapshots revives it), and tying its coverage to one dataset is what let it
    disappear in the first place.

    A 1-D Laplacian under a parameter-dependent downward load against a flat obstacle,
    solved exactly per parameter by the same dual QP the reduced model approximates, so
    the primal and dual snapshots are genuine solutions rather than fabricated fields.
    ``B = I``: the constraint is ``u <= g`` directly, which also makes it the case
    ``rb_online.solve_reduced`` can be checked against.
    """
    import numpy as np
    from scipy.optimize import minimize as _minimize

    from bench.types import Dataset

    h = 1.0 / (dim + 1)
    A = (np.diag(2.0 * np.ones(dim)) - np.diag(np.ones(dim - 1), 1)
         - np.diag(np.ones(dim - 1), -1)) / h**2
    # The load pushes the membrane UP into the obstacle. With ``u <= g`` a downward load
    # never reaches an upper bound, and the only nodes that ever bind are the two at the
    # boundary -- every snapshot then has the same two-node support and the dual set
    # carries no structure for a cone algorithm to compress. Pushed the other way the
    # contact set is an interval about the centre that widens with the load, which is
    # what makes the multipliers a parametric family: 6 active dofs at the lightest load
    # and 16 at the heaviest, over 24.
    gap = np.full(dim, 0.05)
    rng = np.random.default_rng(seed)
    params = np.sort(rng.uniform(0.6, 2.4, size=n))

    A_inv = np.linalg.inv(A)
    U, L = [], []
    for mu in params:
        f = float(mu) * np.ones(dim)
        c = A_inv @ f - gap
        res = _minimize(lambda a: 0.5 * a @ A_inv @ a - a @ c, np.zeros(dim),
                        jac=lambda a: A_inv @ a - c, method="L-BFGS-B",
                        bounds=[(0.0, None)] * dim,
                        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
        lam = np.maximum(res.x, 0.0)
        U.append(A_inv @ (f - lam))
        L.append(lam)
    S_du = np.column_stack(L)
    S_pri = np.column_stack(U)

    idx = np.arange(n)
    test_idx = idx[::3]
    return Dataset(
        name="solvable_obstacle",
        snapshots=S_du,
        description="synthetic 1-D obstacle carrying its own operator, load and gap",
        params=params[:, None],
        train_idx=np.setdiff1d(idx, test_idx),
        test_idx=test_idx,
        primal_snapshots=S_pri,
        A=A,
        B_of_mu=lambda i, d=dim: np.eye(d),
        rhs_of_mu=lambda i, p=params, d=dim: float(p[i]) * np.ones(d),
        gap_of_mu=lambda i, g=gap: g,
    )
