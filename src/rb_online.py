"""[BEE20] Algorithm 1 -- the online stage, plus the [NDEE22] inf-sup fix.

Tag key ([BEE20] = Benaceur/Ern/Ehrlacher; [NDEE22] = Niakh/Drouet/Ehrlacher/Ern,
the sequel): see ``rb_vi_common/__init__.py``. Tags are mandatory in this file
because it now mixes both papers, and their equation numbers collide.
[BEE20] numbers are hal-02081485**v2** numbering.

SCOPE WARNING
-------------
[BEE20] Algorithm 1 has four lines:

  1: Assemble f_hat(mu) and A_hat(mu) using [BEE20] (49)
  2: Compute kappa_hat(mu, v_hat) and gamma_hat(mu, v_hat) using [BEE20] (52)
  3: Compute D^kappa(mu) using kappa_hat(mu, v_hat) and [BEE20] (54)
  4: Solve the reduced saddle-point problem [BEE20] (53) to obtain u_hat, lambda_hat

Lines 2 and 3 are the EIM treatment of the NONLINEAR constraint ([BEE20] §4.3)
and are NOT implemented here, because ``toy_problem.py`` has a linear,
parameter-independent constraint for which the EIM separation is the identity.
What this module implements is line 1 (trivially, the operator being
mu-independent in the toy) and line 4 -- the reduced saddle-point solve, which
is where the CPG cone actually gets used.

Implementing lines 2-3 requires the EIM greedy of [BEE20] §5, Task (T1), which
[BEE20] explicitly declines to detail ("Since Task (T1) can be considered to be
standard, we only discuss Task (T2)"). Doing it faithfully means going to
[BEE20]'s reference [3] (Barrault et al.), not to this paper. See
REPRODUCTION_NOTES.md.

THE INF-SUP GAP, AND WHY [NDEE22] IS IMPORTED HERE
--------------------------------------------------
[BEE20] Algorithm 1 line 4 solves the reduced saddle-point problem over the pair
(V_N, W_R^+) built independently -- POD for the primal, CPG for the dual. It
contains **no inf-sup stabilization**, and [BEE20] does not supply one.

[NDEE22] is the paper that closes that gap, and its §2.3 states the problem
directly: nothing guarantees that the "decorrelated" pair (V_N, W_R^+) is inf-sup
stable, and its Eq. (14) inf-sup constant beta^dec can vanish. The fix is to
enrich the primal space with SUPREMIZERS ([NDEE22] Eq. 15-17),

    V^on_{N,R}(mu) := V_N + S_R(mu),    S_R(mu) := Span{ T(mu) chi_r },

for which [NDEE22] Eq. (18) gives beta^on >= beta_HF > 0. When b(mu; . , .) is
parameter-dependent, so is S_R(mu), and the enrichment "has to be constructed in
the online phase, which is computationally inefficient" -- which is what
[NDEE22] Algorithm 1 (PGA) removes, by building one parameter-INDEPENDENT
subspace S_R^red offline ([NDEE22] Eq. 27-29).

``solve_reduced`` therefore takes an optional ``enrichment``. Omitting it
reproduces [BEE20] Algorithm 1 line 4 bit for bit, which is the default so that
the [BEE20] transcription stays intact and testable. Passing the output of
``pga_enrichment`` applies the [NDEE22] fix.

HONEST CAVEAT FOR *THIS* TOY -- read before quoting a PGA number from here
--------------------------------------------------------------------------
``toy_problem.py`` has B = I: the constraint is u <= gap, so b(v, eta) = eta^T v
with NO mu-dependence. Consequently S_R(mu) = S_R for every mu, and PGA's
headline benefit -- removing an *online* construction cost -- **is vacuous on
this problem**. What PGA still does here is genuine but smaller: it compresses
S_R to a subspace of lower dimension at a prescribed tolerance ([NDEE22]
Eq. 27), so the enriched primal space is cheaper than the full supremizer
enrichment.

Exercising the parameter-dependent case needs a mu-dependent B. The sibling
repository's ``hf_model.ObstacleHF`` has one; see also ``hertz_infsup_probe.py``
here for whether this repository's Hertz contact model could host it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

import _shared_path  # noqa: F401  -- puts rb_vi_common on sys.path
from rb_vi_common import S_R_full, Whitener, inf_sup, orth_union, pga


def solve_reduced(A, f, gap, primal_basis, dual_cone, enrichment=None):
    """Reduced counterpart of [BEE20] Eq. (53), in the linear-constraint case.

    [BEE20] §4.1 decomposes the reduced solutions as

        u_hat(mu) = sum_n u_hat_n(mu) theta_n      (coefficients free in sign)
        lambda_hat(mu) = sum_r lambda_hat_r(mu) xi_r   (coefficients >= 0)

    The sign restriction on the dual coefficients is the whole point of
    [BEE20] §5: it is what keeps lambda_hat inside the cone W_R^+, and it is
    imposed below as a bound in the reduced QP.

    Parameters
    ----------
    A, f, gap : the HF operator, load and obstacle.
    primal_basis : (N_dim, N) array
        Basis (theta_n) of V_hat_N, from POD ([BEE20] §5).
    dual_cone : (N_dim, R) array
        Generators (xi_r) of W_hat_R^+, from ``cone_projected_greedy``.
    enrichment : (N_dim, n_s) array, optional
        Supremizer directions spanning S, in PHYSICAL coordinates, from
        ``pga_enrichment``. The primal space becomes V_N + S, which is
        [NDEE22] Eq. (29) when S = S_R^red and [NDEE22] Eq. (17) when
        S = S_R(mu).

        **Not from [BEE20].** ``None`` (the default) reproduces [BEE20]
        Algorithm 1 line 4 exactly, with no stabilization -- see the module
        docstring.

    Returns
    -------
    u_hat, lam_hat : reconstructed HF-dimension primal and dual solutions.
    """
    Theta = np.asarray(primal_basis, dtype=float)   # (N_dim, N)
    Xi = np.asarray(dual_cone, dtype=float)         # (N_dim, R)

    if enrichment is not None and np.size(enrichment):
        # [NDEE22] Eq. (17)/(29): V_N + S. Re-orthonormalizing the union keeps
        # A_hat well conditioned; the SPACE is what the theory constrains, not
        # the basis, so any basis of the same span gives the same u_hat.
        Theta = orth_union(Theta, np.asarray(enrichment, dtype=float))

    # [BEE20] Algorithm 1, line 1: assemble the reduced operator and load.
    # [BEE20] Eq. (49) writes these as affine combinations of offline-computed
    # terms; with a mu-independent A the sum has a single term and reduces to a
    # Galerkin projection.
    A_hat = Theta.T @ A @ Theta          # [BEE20] Eq. (44a)
    f_hat = Theta.T @ f                  # [BEE20] Eq. (45a)
    B_hat = Xi.T @ Theta                 # [BEE20] Eq. (44b) with K = I
    g_hat = Xi.T @ gap                   # [BEE20] Eq. (45b)

    # [BEE20] Algorithm 1, line 4: solve the reduced saddle-point problem (53).
    # Dual form, as in the HF solve, so the non-negativity of the reduced
    # multiplier coefficients is exact rather than approximate.
    A_hat_inv = np.linalg.inv(A_hat)
    Q = B_hat @ A_hat_inv @ B_hat.T
    c = B_hat @ A_hat_inv @ f_hat - g_hat

    def obj(a):
        return 0.5 * a @ Q @ a - a @ c

    def grad(a):
        return Q @ a - c

    R = Xi.shape[1]
    res = minimize(
        obj, np.zeros(R), jac=grad, method="L-BFGS-B",
        # The bound a >= 0 is the discrete form of "span_+" from [BEE20] §4.1.
        bounds=[(0.0, None)] * R,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12},
    )
    alpha = np.maximum(res.x, 0.0)

    u_coeff = A_hat_inv @ (f_hat - B_hat.T @ alpha)
    return Theta @ u_coeff, Xi @ alpha


def pod_basis(snapshots, n_modes):
    """Primal RB space V_hat_N by POD ([BEE20] §5).

    [BEE20] §5: "it is natural to compress these computations by means of a
    Proper Orthogonal Decomposition (POD) [15, 16] to define the primal RB
    subspace V_N." POD is appropriate for the primal precisely because the
    primal coefficients carry no sign constraint.

    [UNSPECIFIED] [BEE20] does not state whether snapshots are mean-centred
    before the SVD, nor whether N is fixed or chosen from an energy criterion.
    Using: no centring, N given explicitly.
    Alternatives: subtract the snapshot mean; choose N by retained energy.
    """
    U, _, _ = np.linalg.svd(np.asarray(snapshots, dtype=float), full_matrices=False)
    return U[:, :n_modes]


# ---------------------------------------------------------------------------
# [NDEE22] stabilization, applied to this repository's [BEE20] reduced solve.
# Everything below is [NDEE22]; nothing below is in [BEE20].
# ---------------------------------------------------------------------------

def pga_enrichment(A, dual_cone, B=None, delta_pga=0.5, primal_basis=None):
    """Build the supremizer enrichment S, offline, via [NDEE22] Algorithm 1.

    Returns S in PHYSICAL coordinates, ready to pass to ``solve_reduced``.

    The pipeline, all [NDEE22]:
      1. Whiten. [NDEE22] measures everything in the V- and W-inner products.
         We equip V with the ENERGY inner product G_V = A -- the same choice the
         sibling repository makes (``Whitener(hf.K, None)``) -- and W with l^2.
         [UNSPECIFIED] Neither paper states the inner product for this toy;
         the toy is not in either paper.
         Using: G_V = A (energy), G_W = I.
         Alternatives: G_V = I; a mass matrix for W. The energy norm is the one
         for which [NDEE22] Eq. (3) beta_HF is the natural constant.
      2. Supremizers. [NDEE22] Eq. (15)-(17): S_R(mu) = Span{T(mu) chi_r}, and
         in whitened coordinates T(mu) eta = B_hat(mu)^T eta_tilde.
      3. Compress. [NDEE22] Algorithm 1 (PGA) reduces S_R (Eq. 20) to S_R^red
         (Eq. 28) subject to sup_mu sigma_{S_R^red}(mu) <= delta_pga (Eq. 27).
      4. Un-whiten, so ``solve_reduced`` can hstack it with the POD basis.

    Parameters
    ----------
    A : (n, n) array
        HF operator; also the V inner product (see step 1).
    dual_cone : (n_w, R) array
        Cone generators in physical coordinates, from CPG or mCPG.
    B : (n_w, n) array, optional
        Constraint matrix. ``None`` means B = I, which is ``toy_problem``'s
        case -- and which makes B parameter-INDEPENDENT, so PGA's online-cost
        benefit is vacuous here. See the module docstring.
    delta_pga : float
        The tolerance delta_PGA of [NDEE22] Eq. (27).
        [UNSPECIFIED] [NDEE22] §5.1 calibrates its own value to its 2-D
        discretization, so it does not transfer.
        Using: 0.5, swept by the caller.
    primal_basis : (n, N) array, optional
        V_N. PGA measures what V_N already captures ([NDEE22] Eq. 21), so
        passing it yields a smaller S. ``None`` compresses S_R on its own.

    Returns
    -------
    S : (n, n_s) array
        Enrichment directions in physical coordinates.
    info : dict
        ``dim_S_R`` (Eq. 20), ``dim_S_red`` (Eq. 28), ``n_iter``, ``sigmas``.
    """
    A = np.asarray(A, dtype=float)
    X = np.asarray(dual_cone, dtype=float)
    n = A.shape[0]

    wh = Whitener(A, None)                       # G_V = A, G_W = I
    B_mat = np.eye(X.shape[0], n) if B is None else np.asarray(B, dtype=float)
    B_hat = wh.B_hat(B_mat)

    # A single parameter: B does not depend on mu in this toy, so D_train
    # collapses to one entry and [NDEE22] Algorithm 1 lines 2/9 have nothing to
    # choose between. That is a property of the toy, not of PGA.
    B_hats = [B_hat]

    V_N = (np.linalg.qr(wh.L @ np.asarray(primal_basis, float))[0]
           if primal_basis is not None else np.zeros((n, 0)))
    if V_N.shape[1] == 0:
        # [NDEE22] Eq. (21) measures sigma_S(mu) against V_N + S, so PGA needs a
        # V_N to project against. With none supplied there is nothing to
        # compress relative to, and the honest answer is S_R itself (Eq. 20),
        # which trivially satisfies Eq. (27) since sigma_{S_R}(mu) = 0.
        S_tilde = S_R_full(B_hats, X)
        info = {"dim_S_R": S_tilde.shape[1], "dim_S_red": S_tilde.shape[1],
                "n_iter": 0, "sigmas": []}
        return _unwhiten(wh, S_tilde), info

    out = pga(B_hats, V_N, X, delta=delta_pga)
    S_full = S_R_full(B_hats, X)
    info = {"dim_S_R": S_full.shape[1], "dim_S_red": out.S.shape[1],
            "n_iter": out.n_iter, "sigmas": out.sigmas}
    return _unwhiten(wh, out.S), info


def online_enrichment(A, dual_cone, B=None):
    """The full supremizer space S_R(mu) of [NDEE22] Eq. (17), un-compressed.

    This is the enrichment [NDEE22] §2.3 describes before PGA: exact, giving
    beta^on >= beta_HF by [NDEE22] Eq. (18), but of dimension up to R and (for a
    genuinely mu-dependent B) rebuilt per parameter online. Provided as the
    reference against which PGA's compression is measured.
    """
    A = np.asarray(A, dtype=float)
    X = np.asarray(dual_cone, dtype=float)
    wh = Whitener(A, None)
    B_mat = np.eye(X.shape[0], A.shape[0]) if B is None else np.asarray(B, float)
    return _unwhiten(wh, S_R_full([wh.B_hat(B_mat)], X))


def _unwhiten(wh, S_tilde):
    """Map a whitened basis back to physical coordinates: v = L^-1 v_tilde."""
    if S_tilde.shape[1] == 0:
        return S_tilde
    return solve_triangular(wh.L, S_tilde, lower=False)


def infsup_report(A, primal_basis, dual_cone, B=None, enrichment=None):
    """Inf-sup constants for the pair used by ``solve_reduced``.

    Returns (beta_dec, beta_enr):
      beta_dec : [NDEE22] Eq. (14), the DECORRELATED pair (V_N, W_R^+) -- i.e.
                 exactly what [BEE20] Algorithm 1 line 4 uses when
                 ``enrichment`` is None. [NDEE22] §5.1 is the claim that this can
                 be zero.
      beta_enr : [NDEE22] Eq. (30), the pair enriched with ``enrichment``.

    Both are UPPER bounds -- ``rb_vi_common.reduction.inf_sup`` minimizes a
    quotient over a cone, which is non-convex; see its docstring. The separate
    beta = 0 test inside it is exact (a convex QP), so a reported 0 is
    trustworthy while a reported large value is only suggestive.
    """
    A = np.asarray(A, dtype=float)
    X = np.asarray(dual_cone, dtype=float)
    wh = Whitener(A, None)
    B_mat = np.eye(X.shape[0], A.shape[0]) if B is None else np.asarray(B, float)
    B_hat = wh.B_hat(B_mat)

    V_N = np.linalg.qr(wh.L @ np.asarray(primal_basis, float))[0]
    beta_dec = inf_sup(B_hat, V_N, X)                       # [NDEE22] Eq. (14)
    if enrichment is None or not np.size(enrichment):
        return beta_dec, None
    V_off = orth_union(V_N, wh.L @ np.asarray(enrichment, float))  # Eq. (29)
    return beta_dec, inf_sup(B_hat, V_off, X)               # [NDEE22] Eq. (30)


# Kept for callers that want the Cholesky factor directly.
def energy_cholesky(A):
    """Upper Cholesky factor L of the energy inner product, A = L^T L."""
    return cholesky(np.asarray(A, dtype=float), lower=False)
