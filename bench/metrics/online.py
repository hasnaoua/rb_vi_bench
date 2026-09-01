"""Does the reduced model actually solve the problem?

Every other metric family here scores a cone against snapshots: how well ``K_R`` projects
the multipliers, how much of ``K_full`` it captures, how well conditioned its Gram is.
None of them solves anything. But a dual cone is not built to reproduce multipliers -- it
is built so that the *reduced saddle-point problem* can be solved cheaply and still return
the right displacement. That is the quantity [BEE20] and [NDEE22] report, and until now
the benchmark had no way to express it.

The gap this closes is not cosmetic. Projection error and solved error are different
functionals of the same cone, and they can disagree: the reduced solve chooses its
multiplier by minimizing an energy over ``W_R^+``, not by projecting the true one onto it,
so a cone can project every training multiplier well and still steer the solve badly, or
project poorly in directions the solve never visits. On ``toy_bee20`` the two components
already separate by a factor of four -- primal 5.3e-2 against dual 2.1e-1 for the same
cone -- because the primal is far less sensitive to the multiplier than the multiplier is
to itself.

**What is measured.** For each held-out parameter: assemble the reduced system from the
primal POD basis ``V_N`` and the method's dual cone ``W_R^+``, solve it, and compare both
components against the high-fidelity snapshots. Reported as relative errors, mean and max
over the test set.

**The primal basis is a confound and is handled explicitly.** The solved error depends on
``N`` as well as ``R``, so a cone could be blamed for error that is really the primal
space's. ``N`` is therefore made deliberately generous (``PRIMAL_MAX`` modes, or every
training snapshot if there are fewer) rather than trimmed to a retained-energy target.
An energy criterion is the wrong instrument here: at 99.99% it selected **two** modes on
``obstacle_ndee22``, a primal space so coarse that it, not the cone, set the error -- the
opposite of what the metric is for.

Alongside each method the **full-cone reference** is reported: the same solve driven by
``span_+{all training multipliers}``, the widest cone the training data can produce.
It is *not* a lower bound, and must not be read as one. The reduced solve minimizes an
energy over ``W_R^+`` -- it does not project the true multiplier onto it -- so enlarging
the cone can overshoot, and a smaller cone is free to land closer. This is common, not a
curiosity: sweeping R=2..12 over both supported datasets, 25 of 220 method/component
comparisons come in below the full-cone reference, NMF on ``toy_bee20`` at R=9 by a
factor of 2.4 (1.35e-2 against 3.29e-2). Read it as "what using everything achieves",
not as "the best achievable".

**Only some datasets can support this**, and that is a property of the source rather than
a limitation here -- see ``Dataset.supports_online``. Solving needs the operator, the
load and the obstacle, and most sources in the merge ship a snapshot matrix and nothing
else. Those report nothing at all rather than a number resting on an invented
right-hand side.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .. import _paths  # noqa: F401
from ..types import BasisResult, Dataset

from rb_online import pod_basis

#: Size of the primal POD space ``V_N``, capped so the reduced solve cannot approach the
#: cost of the high-fidelity one it replaces. Generous on purpose: the metric exists to
#: measure the DUAL cone, so the primal space must not be what limits the answer.
PRIMAL_MAX = 30


def primal_basis(dataset: Dataset) -> np.ndarray:
    """POD basis of the TRAINING primal snapshots, sized generously.

    Training only. Sizing or building ``V_N`` from the test snapshots would leak the
    answer into the space the answer is computed in.

    Not sized by retained energy. That criterion tracks how compressible the primal
    snapshots are, which is unrelated to whether the primal space is fine enough to stop
    limiting the solve: on ``obstacle_ndee22`` a 99.99% target chose two modes, and the
    resulting error was the primal space's rather than the cone's.
    """
    U = np.asarray(dataset.primal_snapshots, float)
    train = dataset.train_idx if dataset.train_idx is not None else np.arange(U.shape[1])
    Utr = U[:, np.asarray(train, int)]
    return pod_basis(Utr, max(1, min(PRIMAL_MAX, Utr.shape[1])))


def solve_reduced_general(A, f, gap, V, Xi, B):
    """Reduced saddle-point solve, [BEE20] Eq. (53), with a general constraint operator.

    ``rb_online.solve_reduced`` is the reference implementation and is used unchanged
    wherever it applies. It assembles ``B_hat = Xi.T @ V``, i.e. it takes the constraint
    operator to be the identity -- true for ``toy_bee20``, whose constraint is ``u <= g``
    directly. It is not merely inexact for a dataset with a genuine ``B(mu)``: for
    ``obstacle_ndee22`` the multiplier lives on 40 collocation points while the
    displacement lives on ~200 nodes, so ``Xi.T @ V`` does not even have compatible
    shapes and the reference cannot be called at all.

    The only change here is ``B_hat = Xi.T @ B @ V`` ([BEE20] Eq. 44b without ``K = I``).
    Everything else -- the dual form, the non-negativity bound that keeps the reduced
    multiplier inside ``W_R^+``, the recovery of ``u`` -- follows the reference line for
    line, and ``test_general_solve_matches_the_reference_when_B_is_identity`` pins the two
    together on the dataset where both can run.
    """
    V = np.asarray(V, float)
    Xi = np.asarray(Xi, float)
    A_hat = V.T @ A @ V                         # Eq. (44a)
    f_hat = V.T @ np.asarray(f, float)          # Eq. (45a)
    B_hat = Xi.T @ np.asarray(B, float) @ V     # Eq. (44b), general K
    g_hat = Xi.T @ np.asarray(gap, float)       # Eq. (45b)

    A_hat_inv = np.linalg.inv(A_hat)
    Q = B_hat @ A_hat_inv @ B_hat.T
    c = B_hat @ A_hat_inv @ f_hat - g_hat

    R = Xi.shape[1]
    res = minimize(
        lambda a: 0.5 * a @ Q @ a - a @ c, np.zeros(R),
        jac=lambda a: Q @ a - c, method="L-BFGS-B",
        bounds=[(0.0, None)] * R,               # span_+ , [BEE20] §4.1
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12},
    )
    alpha = np.maximum(res.x, 0.0)
    return V @ (A_hat_inv @ (f_hat - B_hat.T @ alpha)), Xi @ alpha


def _rel(approx: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(approx - truth) / denom)


def solved_errors(dataset: Dataset, cone: np.ndarray, V: np.ndarray) -> dict[str, float]:
    """Relative primal and dual errors of the reduced solve, over the test parameters."""
    idx = dataset.test_idx if dataset.test_idx is not None else np.arange(dataset.n_snapshots)
    idx = np.asarray(idx, int)
    if idx.size == 0 or np.asarray(cone).shape[1] == 0:
        return {}
    # ``supports_online`` already implies all five are present, but a type checker
    # cannot narrow an Optional through a property, and calling four of them below
    # would be a TypeError if a caller ever skipped that guard. Bind them once.
    A, B_of_mu = dataset.A, dataset.B_of_mu
    rhs_of_mu, gap_of_mu = dataset.rhs_of_mu, dataset.gap_of_mu
    if (A is None or B_of_mu is None or rhs_of_mu is None
            or gap_of_mu is None or dataset.primal_snapshots is None):
        return {}
    U = np.asarray(dataset.primal_snapshots, float)
    L = np.asarray(dataset.snapshots, float)
    primal, dual = [], []
    for q in idx:
        q = int(q)
        u_hat, lam_hat = solve_reduced_general(
            A, rhs_of_mu(q), gap_of_mu(q),
            V, cone, B_of_mu(q),
        )
        primal.append(_rel(u_hat, U[:, q]))
        dual.append(_rel(lam_hat, L[:, q]))
    primal = np.asarray(primal, float)
    dual = np.asarray(dual, float)
    return {
        "online_primal_mean_rel": float(np.nanmean(primal)),
        "online_primal_max_rel": float(np.nanmax(primal)),
        "online_dual_mean_rel": float(np.nanmean(dual)),
        "online_dual_max_rel": float(np.nanmax(dual)),
    }


def evaluate(dataset: Dataset, result: BasisResult) -> dict[str, float]:
    """Solved error for one method's cone, with the full-cone floor beside it."""
    if not dataset.supports_online:
        return {}
    G = np.asarray(result.generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {}
    V = primal_basis(dataset)
    row = solved_errors(dataset, G, V)
    if not row:
        return {}
    row["online_N_primal"] = float(V.shape[1])
    # Reference, not a bound: the same solve driven by span_+{every training multiplier}.
    # It says what using the whole training cone achieves, which is the scale a method's
    # error should be read against. It is NOT a floor -- the reduced solve minimizes over
    # the cone rather than projecting onto it, so a smaller cone can land closer.
    ref = solved_errors(dataset, dataset.train(), V)
    if ref:
        row["online_primal_fullcone"] = ref["online_primal_mean_rel"]
        row["online_dual_fullcone"] = ref["online_dual_mean_rel"]
    return row
