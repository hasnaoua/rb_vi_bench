"""NMF dual-basis construction -- the baseline CPG is compared against.

Paper: Benaceur, Ern, Ehrlacher, hal-02081485v2, Section 6.4 (Appendix).

Section 5 attributes this approach to [2] (Balajewicz, Amsallem, Farhat, 2016)
and gives two reasons to prefer CPG over it:
  * "the resulting dual RB cone can be less accurate than the primal RB space";
  * "the user does not specify an error tolerance but only the cardinality of
    the family of vectors generating the dual RB cone."

The second point is the structural one and is visible in this module's
signature: ``R`` is an input here, whereas ``cone_projected_greedy`` takes a
tolerance and *discovers* R. Section 6.4 closes on the same note: "in contrast
to the POD, the integer R is a required input for the NMF."
"""

from __future__ import annotations

import numpy as np


def nmf(T, R, n_iter=200, seed=0, eps=1e-10):
    """W = NMF(T, R), the procedure of Eq. (66).

    Section 6.4 poses the optimization problem, Eq. (68):

        (W, H) = argmin_{W~ >= 0, H~ >= 0} ||T - W~ H~||^2

    with ||.|| the Frobenius norm of Eq. (67), and solves it with the
    multiplicative update rules of Eq. (69), for which [17] proves that the
    Frobenius norm decreases:

        H_ij <- H_ij (W^T T)_ij / (W^T W H)_ij
        W_ij <- W_ij (T H^T)_ij / (W H H^T)_ij

    Parameters
    ----------
    T : (R_dim, P) array
        Non-negative snapshot matrix; Section 6.4 requires all entries >= 0.
    R : int
        Requested number of generators. Required input -- see module docstring.
    n_iter : int
        [UNSPECIFIED] Section 6.4 gives the update rules but no stopping rule
        or iteration count.
        Using: 200 fixed iterations.
        Alternatives: stop on relative decrease of Eq. (67) below a tolerance;
        stop on small relative change in W. Because Eq. (68) is "not convex in
        both variables together" and "only the recovery of local minima is
        considered", the iterate reached depends on this choice.
    seed : int
        [UNSPECIFIED] Section 6.4 does not state the initialization of W and H.
        Using: uniform random on [0,1) at this seed.
        Alternatives: NNDSVD (the usual deterministic choice), random columns
        drawn from T. Section 6.4 notes the factorization is not unique -- "any
        positive matrix D satisfies T = W D D^-1 H" -- so different seeds give
        genuinely different W, not merely permuted ones.
    eps : float
        [UNSPECIFIED] Eq. (69) divides by (W^T W H)_ij and (W H H^T)_ij, which
        can underflow to zero; the paper does not mention a guard.
        Using: additive eps in the denominators.
        Alternatives: clip factors away from 0 each sweep.

    Returns
    -------
    W : (R_dim, R) array
        R non-negative generating vectors (w_1, ..., w_R).

    Notes
    -----
    The paper writes T in R_+^{R x P} and W in R_+^{R x R}, overloading R as
    both the multiplier dimension and the basis cardinality. Here the first is
    ``R_dim`` and the second ``R``; the returned shape is (R_dim, R).
    """
    T = np.asarray(T, dtype=float)
    if T.min() < 0:
        raise ValueError("Section 6.4 requires all entries of T to be non-negative")
    if not 1 <= R <= min(T.shape):
        raise ValueError(f"R must be in [1, {min(T.shape)}], got {R}")

    rng = np.random.default_rng(seed)
    R_dim, P = T.shape
    W = rng.random((R_dim, R))
    H = rng.random((R, P))

    for _ in range(n_iter):
        # Eq. (69), first rule. Order matters: H is updated with the current W,
        # then W with the just-updated H, as written.
        H *= (W.T @ T) / (W.T @ W @ H + eps)
        # Eq. (69), second rule.
        W *= (T @ H.T) / (W @ H @ H.T + eps)

    return W


def frobenius_error(T, W, H):
    """Eq. (67): ||A - B||^2 := sum_ij (A_ij - B_ij)^2, reported as the norm."""
    return float(np.linalg.norm(np.asarray(T) - np.asarray(W) @ np.asarray(H), "fro"))
