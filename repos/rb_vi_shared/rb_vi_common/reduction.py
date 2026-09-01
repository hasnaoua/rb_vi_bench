"""Reduced spaces, supremizers, and inf-sup constants.

Origin: [NDEE22] §2-§3 in full. See ``rb_vi_common/__init__.py`` for the tag key.

CITATION CONVENTION FOR THIS MODULE
-----------------------------------
Every section, equation and lemma number in this module refers to **[NDEE22]**
(Niakh, Drouet, Ehrlacher, Ern, ESAIM: M2AN 2022, doi 10.1051/m2an/2022077).
Nothing here comes from [BEE20]. Algorithm references are tagged explicitly
anyway, because both papers have an "Algorithm 1" and an "Algorithm 2" and they
denote different algorithms.

WHITENED COORDINATES
--------------------
Every quantity in the paper is measured in the V- and W-inner products, not the
Euclidean one. Rather than carry Gram matrices through every formula, we change
coordinates once. With G_V = L^T L (Cholesky) and G_W = M^T M,

    v_tilde := L v,     eta_tilde := M eta

so ||v||_V = ||v_tilde||_2 and ||eta||_W = ||eta_tilde||_2, and

    b(mu; v, eta) = eta^T B(mu) v = eta_tilde^T  [ M^-T B(mu) L^-1 ]  v_tilde.

We call B_hat(mu) := M^-T B(mu) L^-1 the whitened constraint matrix. In these
coordinates orthogonal projections are ordinary ones and every norm is
Euclidean, so the formulas below read exactly as in the paper.

A useful consequence: the supremizer operator (15), T(mu) = J . B_HF(mu) with J
the Riesz isomorphism, becomes simply

    T_tilde(mu) eta = B_hat(mu)^T eta_tilde

since J = G_V^-1 and L G_V^-1 B^T = L (L^T L)^-1 B^T = L^-T B^T = (B L^-1)^T.
The Riesz map disappears into the change of coordinates -- it is not dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


class Whitener:
    """Change of coordinates making the V and W inner products Euclidean."""

    def __init__(self, G_V, G_W=None):
        self.L = cholesky(np.asarray(G_V, dtype=float), lower=False)
        self.G_W = G_W
        self.M = None if G_W is None else cholesky(np.asarray(G_W, float), lower=False)

    def v_to_tilde(self, v):
        return self.L @ v

    def v_from_tilde(self, vt):
        return solve_triangular(self.L, vt, lower=False)

    def w_to_tilde(self, e):
        return e if self.M is None else self.M @ e

    def B_hat(self, B):
        """M^-T B L^-1, the whitened constraint matrix."""
        # B L^-1: solve X L = B, i.e. L^T X^T = B^T.
        # scipy documents trans as 0/1/2 or "N"/"T"/"C"; only the int form survives
        # pyright's inference of the unstubbed source, which reads the default 0.
        X = solve_triangular(self.L, np.asarray(B, float).T, lower=False,
                             trans="T").T  # type: ignore[arg-type]
        if self.M is None:
            return X
        return solve_triangular(self.M, X, lower=False, trans="T")  # type: ignore[arg-type]


def pod(snapshots_tilde, delta):
    """POD with the relative tolerance of Eq. (11).

    Eq. (11):
        e_POD(N) = [ sum_p ||(I - Pi_{V_N}) u(mu_p)||^2 ]^(1/2)
                   / [ sum_p ||u(mu_p)||^2 ]^(1/2)          <=  delta_POD

    In whitened coordinates the numerator is the discarded singular-value
    energy, so N is the smallest integer with
    sqrt(sum_{k>N} s_k^2 / sum_k s_k^2) <= delta.

    Returns an orthonormal basis (columns) of V_N, Eq. (10).
    """
    U, s, _ = np.linalg.svd(np.asarray(snapshots_tilde, float), full_matrices=False)
    total = np.sum(s**2)
    if total == 0:
        return U[:, :1]
    tail = np.concatenate([np.cumsum(s[::-1] ** 2)[::-1][1:], [0.0]])
    rel = np.sqrt(tail / total)
    N = int(np.searchsorted(-rel, -delta) + 1)   # first N with rel[N-1] <= delta
    N = max(1, min(N, len(s)))
    return U[:, :N]


def supremizer_space(B_hat_mu, cone_generators_tilde):
    """S_R(mu) of Eq. (17): Span{ T(mu) chi_r }.

    In whitened coordinates T(mu) chi_r = B_hat(mu)^T chi_r_tilde.

    Returns an orthonormal basis of S_R(mu). Rank deficiency is real and
    expected -- distinct cone generators can produce parallel supremizers -- so
    we drop directions below a relative tolerance rather than assume full rank.
    """
    S = np.asarray(B_hat_mu, float).T @ np.asarray(cone_generators_tilde, float)
    U, s, _ = np.linalg.svd(S, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        return np.zeros((S.shape[0], 0))
    keep = s > max(S.shape) * np.finfo(float).eps * s[0]
    return U[:, keep]


def sigma_S(Z_mu, Q):
    """sigma_S(mu) of Eq. (21).

    Eq. (21):  sigma_S(mu) = || (I - Pi_{V_N + S}) |_{S_R(mu)} ||_{L(V)}

    With Z_mu an orthonormal basis of S_R(mu) and Q an orthonormal basis of
    V_N + S, a vector v in S_R(mu) is Z_mu c with ||v|| = ||c||, so the operator
    norm is the largest singular value of (I - Q Q^T) Z_mu.

    Returns (sigma, c_max) where c_max is the maximizing coefficient vector --
    line 5 of [NDEE22] Algorithm 1 (PGA) needs the argmax, not just the value.
    """
    if Z_mu.shape[1] == 0:
        return 0.0, None
    Rz = Z_mu - Q @ (Q.T @ Z_mu) if Q.shape[1] else Z_mu
    U, s, Vt = np.linalg.svd(Rz, full_matrices=False)
    return float(s[0]), Vt[0]


def orth_union(*blocks):
    """Orthonormal basis of the sum of several subspaces, e.g. V_N + S."""
    cols = [b for b in blocks if b is not None and b.shape[1] > 0]
    if not cols:
        raise ValueError("orth_union needs at least one non-empty block")
    A = np.hstack(cols)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > max(A.shape) * np.finfo(float).eps * s[0]
    return U[:, keep]


def inf_sup(B_hat_mu, Q_primal, cone_tilde, n_starts=12, n_iter=800, seed=0):
    """Inf-sup constant over a primal SUBSPACE and a dual CONE.

    Computes, for a primal space with orthonormal basis Q and a dual cone
    W_R^+ = Span^+{chi_r}:

        beta = inf_{eta in W_R^+}  sup_{v in Q}  b(mu; v, eta) / (||v|| ||eta||)

    This is Eq. (14) for the decorrelated pair, Eq. (18) for the online-enriched
    pair, and Eq. (30) for the offline-enriched pair -- only the primal space Q
    changes between them.

    The inner sup over a subspace is exact: for fixed eta it equals
    ||Q^T B_hat^T eta_tilde||. Writing eta_tilde = X alpha with alpha >= 0
    (X the whitened generators), the problem becomes

        beta = min_{alpha >= 0, alpha != 0}  ||C alpha|| / ||X alpha||,
        C := Q^T B_hat^T X.

    [UNSPECIFIED] The paper reports beta throughout §5 but does not state how it
    is computed; it says only that the algorithms use cvxopt (§5). The outer
    minimization is over a CONE, not a subspace, so it is not an eigenvalue
    problem and has no closed form.
    Using: multi-start projected gradient on the generalized Rayleigh quotient,
    with the unconstrained generalized eigenvector as one of the starts.
    Alternatives: an SOCP/QP sequence (the natural cvxopt route); enumerating
    the faces of the cone (exact but exponential in R).
    Because this is a non-convex minimization, the value returned is an UPPER
    bound on beta. It is therefore reliable for showing beta is SMALL (the
    paper's instability claim) and only suggestive when showing beta is large.
    """
    X = np.asarray(cone_tilde, float)
    if X.shape[1] == 0:
        return 0.0
    C = Q_primal.T @ np.asarray(B_hat_mu, float).T @ X
    A1 = C.T @ C
    A2 = X.T @ X

    # --- Step 1: exact test for beta = 0 ---------------------------------
    # The quotient is scale-invariant, so we may normalize to the simplex.
    # On the simplex ||X alpha|| is bounded away from 0 for a pointed cone with
    # non-zero generators, hence
    #
    #     beta = 0   <=>   min_{alpha in simplex} ||C alpha|| = 0.
    #
    # That minimization is a CONVEX QP (A1 is PSD, constraints linear), so it is
    # solved reliably -- unlike the quotient itself. This matters: the paper's
    # central negative claim is that beta^dec vanishes, and whenever
    # dim(V_N) < R the matrix C has a non-trivial null space that generically
    # meets the non-negative orthant. A local method on the quotient will miss
    # that minimum and report a comfortably positive beta.
    n_a = X.shape[1]
    qp = minimize(
        lambda a: a @ A1 @ a, np.full(n_a, 1.0 / n_a),
        jac=lambda a: 2.0 * A1 @ a, method="SLSQP",
        bounds=[(0.0, None)] * n_a,
        constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1.0,
                      "jac": lambda a: np.ones(n_a)}],
        options={"maxiter": 500, "ftol": 1e-16},
    )
    a_qp = np.maximum(qp.x, 0.0)
    if a_qp.sum() > 0:
        a_qp = a_qp / a_qp.sum()
        num = np.linalg.norm(C @ a_qp)
        den = np.linalg.norm(X @ a_qp)
        scale = np.linalg.norm(C, 2)
        if den > 0 and num <= 1e-8 * max(scale, 1.0):
            return 0.0

    def rayleigh(a):
        d = a @ A2 @ a
        return np.inf if d <= 0 else (a @ A1 @ a) / d

    # --- Step 2: refine the value ----------------------------------------
    rng = np.random.default_rng(seed)
    starts = [np.ones(X.shape[1])]
    if a_qp.sum() > 0:
        starts.append(a_qp)      # the QP optimum is a strong starting point
    try:
        # Unconstrained generalized eigenvector, clipped into the orthant: if it
        # happens to be non-negative it is the exact minimizer.
        w, V = np.linalg.eigh(np.linalg.solve(A2, A1))
        starts.append(np.abs(V[:, 0]))
    except np.linalg.LinAlgError:
        pass
    starts += [rng.random(X.shape[1]) for _ in range(n_starts)]

    best = np.inf
    for a0 in starts:
        a = np.maximum(a0, 0.0)
        if np.all(a == 0):
            continue
        a = a / np.linalg.norm(a)
        step = 1.0
        cur = rayleigh(a)
        for _ in range(n_iter):
            # d/da of the quotient, up to a positive factor
            d = a @ A2 @ a
            grad = 2.0 * (A1 @ a - cur * (A2 @ a)) / d
            trial = np.maximum(a - step * grad, 0.0)
            nrm = np.linalg.norm(trial)
            if nrm == 0:
                step *= 0.5
                continue
            trial /= nrm
            val = rayleigh(trial)
            if val < cur:
                a, cur, step = trial, val, step * 1.2
            else:
                step *= 0.5
                if step < 1e-14:
                    break
        best = min(best, cur)

    return float(np.sqrt(max(best, 0.0)))


def boundedness_c_S(B_hat_mu, Q_primal, cone_tilde):
    """c_S(mu) of Eq. (22).

    Eq. (22):  c_S(mu) := sup_{eta in W_R^+} sup_{v in V_N + S_R(mu) + S}
                          b(mu; v, eta) / (||v||_V ||eta||_W)   <=  C_0

    We return the supremum over the SPAN of the cone rather than the cone
    itself. That is an upper bound on c_S, and since c_S enters Prop. 3.1 only
    through the criterion sigma_S(mu) < beta^on/c_S (Eq. 23) and the bound
    Eq. (24), overestimating it makes both CONSERVATIVE -- the verified
    conclusion stays valid. Over a span the supremum is exactly a largest
    singular value, so no non-convex optimization is involved.
    """
    X = np.asarray(cone_tilde, float)
    if X.shape[1] == 0:
        return 0.0
    C = Q_primal.T @ np.asarray(B_hat_mu, float).T @ X
    # max over the span of ||C a|| / ||X a||: largest singular value of C W^-1
    # where W is a whitening of X^T X.
    G = X.T @ X
    w, U = np.linalg.eigh(G)
    keep = w > max(G.shape) * np.finfo(float).eps * w.max()
    Winv = U[:, keep] / np.sqrt(w[keep])
    return float(np.linalg.norm(C @ Winv, 2))


def inf_sup_hf(B_hat_mu, n_starts=12, seed=0):
    """beta_HF(mu) of Eq. (3): the primal space is all of V, the dual cone all of W^+.

    With the full primal space the inner sup is ||B_hat^T eta||, so
    beta_HF = min_{eta >= 0, ||eta||=1} ||B_hat^T eta||.
    """
    Bh = np.asarray(B_hat_mu, float)
    n_w = Bh.shape[0]
    return inf_sup(Bh, np.eye(Bh.shape[1]), np.eye(n_w),
                   n_starts=n_starts, seed=seed)
