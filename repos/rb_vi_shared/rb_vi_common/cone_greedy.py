"""Cone-building greedy algorithms: CPG (two transcriptions) and mCPG.

See ``rb_vi_common/__init__.py`` for the [BEE20] / [NDEE22] tag key and for the
v2-numbering caveat that applies to every [BEE20] number below.

WHAT IS IN HERE
---------------
  ``cone_projected_greedy``  [BEE20] Algorithm 2 -- CPG, as transcribed from the
                             hal-02081485v2 preprint.
  ``cpg``                    [NDEE22] Remark 4.3 -- the same algorithm, as
                             transcribed from [NDEE22]'s description of its own
                             reference [9] (which IS [BEE20]).
  ``mcpg``                   [NDEE22] Algorithm 2 -- mCPG. **The primary
                             algorithm**: [NDEE22] §4 supersedes plain CPG with
                             it, and Remark 4.3 states the difference exactly.
  ``e_orth``                 [NDEE22] Eq. (41) -- the near-orthogonality
                             diagnostic used to compare the two.

WHY TWO CPGs, AND NOT ONE
-------------------------
``cone_projected_greedy`` and ``cpg`` are the SAME algorithm read out of two
different papers, and they are kept as two functions on purpose. They differ in
three surface conventions, each of which is faithful to its own source:

  (a) Generators. [BEE20] line 6 sets K_n^+ := span_+{lambda(mu_1),...,
      lambda(mu_n)} -- the RAW snapshots. [NDEE22] Remark 4.3 says CPG "simply
      sets nu_r = theta_{q_r} / ||theta_{q_r}||_W" -- the NORMALIZED snapshots.
      These generate the IDENTICAL CONE: positively rescaling a generator does
      not change span_+. Only the returned matrix differs, by a column scaling.

  (b) Stopping tolerance. [BEE20] Eq. (58) is ABSOLUTE, in the ||.||_Lambda norm:
          max_mu ||lambda(mu) - Pi(lambda(mu))||_Lambda <= eps_du.
      [NDEE22] Eq. (13) is RELATIVE, normalized by the largest snapshot norm:
          max_p ||(I - Pi) lambda(mu_p)|| / max_p ||lambda(mu_p)|| <= delta.
      [BEE20] §5 explicitly notes the alternative: "One can also consider a
      relative error criterion instead of an absolute one by dividing the
      left-hand side of (58) by ||lambda(mu)||_Lambda." So the two are the two
      options [BEE20] itself names, with [NDEE22] taking the second.

  (c) Candidate set for the argmax. [BEE20] line 5 maximizes over all of P_tr;
      [NDEE22] line 12 maximizes over q not in I_r. These agree, because a
      selected snapshot lies in the cone and so has projection error 0.

  (d) Weighted norm. [BEE20] carries a ||.||_Lambda Gram matrix (``mass``);
      [NDEE22] whitens coordinates first, so its norm is Euclidean.

Collapsing them into one function would mean picking one paper's convention and
silently attributing it to the other. ``tests/test_equivalence.py`` checks claim
(a) numerically -- that the two produce the same cone and the same selection
order -- which is the useful thing to know, and is a claim rather than an
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .cone_projection import norm_Lambda, project_onto_cone


# ---------------------------------------------------------------------------
# Result containers. Two, not one: the two papers report different quantities
# and the field names are part of each transcription's readability.
# ---------------------------------------------------------------------------

@dataclass
class CPGResult:
    """Output of [BEE20] Algorithm 2."""

    generators: np.ndarray          # (R_dim, R) -- the cone W_R^+, Output line
    selected_indices: list          # indices into the training set, mu_1..mu_R
    residuals: list = field(default_factory=list)  # r_n per iteration
    R: int = 0                      # [BEE20] Algorithm 2, line 10: R := n-1


@dataclass
class ConeResult:
    """Output of [NDEE22] Algorithm 2 (and of the Remark 4.3 baseline)."""

    generators: np.ndarray   # (dim, R) whitened generators nu_r (or chi_r)
    order: list              # indices q_r of the selected snapshots
    errors: list             # e_r per iteration
    R: int


def _proj(y, G):
    """Pi_{K}(y) in Euclidean geometry -- the [NDEE22] whitened-coordinate case."""
    return project_onto_cone(y, G, mass=None)[0]


# ---------------------------------------------------------------------------
# [BEE20] Algorithm 2 -- CPG
# ---------------------------------------------------------------------------

def cone_projected_greedy(snapshots, eps_du, mass=None, max_R=None):
    """[BEE20] Algorithm 2 -- Cone-Projected Greedy, transcribed line by line.

    This is [BEE20]'s core contribution: a weak-greedy algorithm that builds a
    hierarchical dual reduced-basis cone from Lagrange-multiplier snapshots while
    preserving their non-negativity.

    Why not POD, and why not NMF -- both stated in [BEE20] §5:
      * POD: "Bearing in mind that the dual RB cone W_R^+ is meant to represent
        the set of Lagrange multipliers, its spanning vectors should all have
        non-negative components. Consequently, the POD is not appropriate to
        build W_R^+."  POD modes have mixed signs.
      * NMF (the [BEE20]-reference-[2] approach): "the resulting dual RB cone can
        be less accurate than the primal RB space. Moreover, the user does not
        specify an error tolerance but only the cardinality." CPG is
        tolerance-driven and hierarchical.

    Its stopping criterion is [BEE20] Eq. (58):

        max_{mu in P_tr} || lambda(mu) - Pi_{K_{n-1}^+}(lambda(mu)) ||_Lambda  <=  eps_du

    Parameters
    ----------
    snapshots : (R_dim, n_train) array
        S_du := {lambda(mu)}_{mu in P_tr}, one snapshot per column
        ([BEE20] Algorithm 2, line 1). These are HF Lagrange multipliers and must
        be non-negative -- they are the multipliers of an inequality constraint.
    eps_du : float
        The tolerance eps_du of [BEE20] Eq. (58). Algorithm 2 requires eps_du > 0.
    mass : (R_dim, R_dim) array, optional
        Gram matrix of ||.||_Lambda. See ``cone_projection.norm_Lambda``.
    max_R : int, optional
        Not in [BEE20]. A safety cap so the loop cannot run to n_train when
        eps_du is set below the achievable accuracy. Defaults to n_train, at
        which point the cone contains every snapshot and the residual is 0.

    Returns
    -------
    CPGResult
    """
    S = np.asarray(snapshots, dtype=float)
    if S.ndim != 2:
        raise ValueError("snapshots must be 2-D, shape (R_dim, n_train)")
    if eps_du <= 0:
        raise ValueError("[BEE20] Algorithm 2 requires eps_du > 0")
    # The multipliers of an inequality constraint are non-negative by
    # construction ([BEE20] §2: lambda(mu) in W_R^+). A negative entry means the
    # HF solve is wrong, and CPG would silently build a cone that cannot
    # represent the snapshots.
    if S.min() < 0:
        raise ValueError(
            f"snapshots contain negative entries (min={S.min():.3e}); "
            "Lagrange multipliers of the contact constraint must be >= 0"
        )

    n_train = S.shape[1]
    cap = n_train if max_R is None else min(max_R, n_train)

    selected: list[int] = []
    residuals: list[float] = []

    # [BEE20] Algorithm 2, line 2: K_0^+ := {0}
    # Represented by an empty generator matrix; project_onto_cone maps it to 0.
    generators = np.zeros((S.shape[0], 0))

    # [BEE20] Algorithm 2, line 3: n := 1 and r_1 := 2 * eps_du
    # r_1 is seeded above the tolerance purely so the while test at line 4
    # cannot short-circuit before the first iteration.
    r_n = 2.0 * eps_du

    # [BEE20] Algorithm 2, line 4: while (r_n > eps_du) do
    while r_n > eps_du and len(selected) < cap:
        # [BEE20] Algorithm 2, line 5:
        #   mu_n in argmax_{mu in P_tr} ||lambda(mu) - Pi_{K_{n-1}^+}(lambda(mu))||_Lambda
        # At n = 1 the cone is {0} and this reduces to [BEE20] Eq. (56),
        #   mu_1 in argmax ||lambda(mu)||_Lambda.
        errors = _projection_errors(S, generators, mass)

        # Already-selected snapshots lie in the cone, so their error is 0 up to
        # round-off. Masking them makes "argmax" well defined -- [BEE20]'s
        # "in argmax" notation admits ties without prescribing a tie-break.
        # [UNSPECIFIED] [BEE20] does not say how to break ties in line 5.
        # Using: lowest training index, via argmax's first-maximum convention.
        # Alternatives: random choice; largest ||lambda(mu)||.
        errors[selected] = -np.inf
        mu_n = int(np.argmax(errors))
        selected.append(mu_n)

        # [BEE20] Algorithm 2, line 6:
        #   K_n^+ := span_+{lambda(mu_1), ..., lambda(mu_n)}
        # Raw snapshots, NOT normalized -- see the module docstring, point (a).
        generators = S[:, selected]

        # [BEE20] Algorithm 2, line 7: n := n+1
        # [BEE20] Algorithm 2, line 8:
        #   r_n := max_{mu} ||lambda(mu) - Pi_{K_{n-1}^+}(lambda(mu))||_Lambda
        # After the increment at line 7, the "K_{n-1}" of line 8 is the cone
        # just built at line 6 -- so this measures the error of the CURRENT
        # cone, and is what the line-4 test consumes on the next pass.
        r_n = float(np.max(_projection_errors(S, generators, mass)))
        residuals.append(r_n)

    # [BEE20] Algorithm 2, line 10: R := n-1  (generators actually kept)
    # Output: W_R^+ := K_R^+
    return CPGResult(
        generators=generators,
        selected_indices=selected,
        residuals=residuals,
        R=generators.shape[1],
    )


def _projection_errors(S, generators, mass):
    """||lambda(mu) - Pi_{K^+}(lambda(mu))||_Lambda for every column of S.

    This is the expensive part of [BEE20] Algorithm 2: one cone projection --
    i.e. one NNLS solve -- per training parameter per greedy iteration, so the
    offline cost is O(R * n_train) projections.
    """
    out = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        lam = S[:, j]
        proj, _ = project_onto_cone(lam, generators, mass=mass)
        out[j] = norm_Lambda(lam - proj, mass=mass)
    return out


# ---------------------------------------------------------------------------
# [NDEE22] Remark 4.3 baseline -- CPG in [NDEE22]'s own conventions
# ---------------------------------------------------------------------------

def cpg(snapshots, delta, max_R=None):
    """Plain CPG as [NDEE22] describes its reference [9]; the Remark 4.3 baseline.

    [NDEE22] Remark 4.3 states the difference from mCPG exactly:

        "The main difference between the CPG and mCPG algorithms is that at each
         iteration r >= 1, the CPG algorithm does not execute line 9 and simply
         sets nu_r = theta_{q_r} / ||theta_{q_r}||_W. Instead, the mCPG algorithm
         computes nu_r as a member of W^+ (in fact of
         Span^+({theta_{q_n}}_{n in {1:r}}))."

    Greedily selects the snapshot worst represented by the current cone and
    appends it, NORMALIZED, as a new generator. Stopping criterion
    [NDEE22] Eq. (13):

        e_CPG(R) = max_p ||(I - Pi_{W_R^+}) lambda(mu_p)|| / max_p ||lambda(mu_p)||  <=  delta

    Equivalent to ``cone_projected_greedy`` up to generator scaling and the
    absolute-vs-relative tolerance -- see the module docstring.
    """
    T = np.asarray(snapshots, float)
    Q = T.shape[1]
    cap = Q if max_R is None else min(max_R, Q)
    norms = np.linalg.norm(T, axis=0)
    denom = norms.max()

    q1 = int(np.argmax(norms))
    chosen, gens, errs = [], [], []
    K = np.zeros((T.shape[0], 0))

    q = q1
    while len(chosen) < cap:
        chosen.append(q)
        # CPG: the generator IS the normalized snapshot ([NDEE22] Remark 4.3).
        nu = T[:, q] / np.linalg.norm(T[:, q])
        gens.append(nu)
        K = np.column_stack(gens)

        rest = [j for j in range(Q) if j not in chosen]
        if not rest:
            errs.append(0.0)
            break
        res = [np.linalg.norm(T[:, j] - _proj(T[:, j], K)) / denom for j in rest]
        k = int(np.argmax(res))
        e_r = res[k]
        errs.append(e_r)
        if e_r <= delta:
            break
        q = rest[k]

    return ConeResult(np.column_stack(gens), chosen, errs, len(gens))


# ---------------------------------------------------------------------------
# [NDEE22] Algorithm 2 -- mCPG. The primary algorithm.
# ---------------------------------------------------------------------------

def _closest_in_cone_below(theta, K):
    """Line 9 of [NDEE22] Algorithm 2.

    Solves    Upsilon_r  in  argmin_{Upsilon in K_{r-1} INTERSECT (theta_{q_r} - W^+)}
                              || theta_{q_r} - Upsilon ||_W

    Two constraints, and both matter:
      * Upsilon in K_{r-1}: Upsilon = K c with c >= 0;
      * Upsilon in theta - W^+: theta - Upsilon must lie in the positive cone,
        i.e. (with W^+ the non-negative orthant) K c <= theta componentwise.

    The second is what keeps nu_r = (theta - Upsilon)/||theta - Upsilon|| inside
    W^+. It is the reason [NDEE22] §4 can widen the aperture without "a departure
    from the positive cone", which is what an orthogonalization would cause.

    [UNSPECIFIED] [NDEE22] does not name a solver for this QP; its §5 says only
    that the algorithms use cvxopt.
    Using: SLSQP on the equivalent least-squares problem.
    Alternatives: any QP solver (cvxopt's is the authors' route).
    """
    if K.shape[1] == 0:
        return np.zeros_like(theta)

    def obj(c):
        r = theta - K @ c
        return r @ r

    def jac(c):
        return -2.0 * K.T @ (theta - K @ c)

    res = minimize(
        obj, np.zeros(K.shape[1]), jac=jac, method="SLSQP",
        bounds=[(0.0, None)] * K.shape[1],
        # theta - K c >= 0  (componentwise), i.e. K c in theta - W^+
        constraints=[{"type": "ineq",
                      "fun": lambda c: theta - K @ c,
                      "jac": lambda c: -K}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    c = np.maximum(res.x, 0.0)
    ups = K @ c
    # Enforce the membership exactly against solver round-off: Upsilon must not
    # exceed theta anywhere, or nu_r would leave W^+.
    return np.minimum(ups, theta)


def mcpg(snapshots, delta, max_R=None):
    """[NDEE22] Algorithm 2 -- mCPG, transcribed line by line.

    Require: {theta_q}_{q in {1:Q}}, Hilbert space W, tolerance delta > 0
    Ensure:  {nu_r}_{r in {1:R}} subset W^+, R <= Q

    CPG's generators are selected snapshots; mCPG's are snapshot RESIDUALS after
    removing what the existing cone already explains -- while staying inside W^+.
    [NDEE22] §4 motivates this: one wants "a basis such that the corresponding
    Gram matrix is as well-conditioned as possible", and ordinary Gram-Schmidt is
    unavailable because "this would lead to a departure from the positive cone
    W^+".
    """
    T = np.asarray(snapshots, float)
    Q = T.shape[1]
    cap = Q if max_R is None else min(max_R, Q)

    # 1: r := 0
    r = 0
    # 2: I_0 := empty
    I: list[int] = []
    # 3: K_0 := {0}
    K = np.zeros((T.shape[0], 0))
    gens: list[np.ndarray] = []
    # 4: e_0 := 1 + delta   (seeded above tolerance so the loop body runs once)
    e_r = 1.0 + delta
    errs: list[float] = []
    # 5: q_1 := argmax_q ||theta_q||_W
    norms = np.linalg.norm(T, axis=0)
    q = int(np.argmax(norms))
    denom = norms[q]        # ||theta_{q_1}||_W, the normalizer in lines 12-13

    # 6: while (e_r > delta) and (r < Q) do
    while e_r > delta and r < cap:
        # 7: r := r + 1
        r += 1
        # 8: I_r := I_{r-1} + {q_r}
        I.append(q)
        # 9: Upsilon_r := argmin over K_{r-1} INTERSECT (theta_{q_r} - W^+)
        ups = _closest_in_cone_below(T[:, q], K)
        # 10: nu_r := (theta_{q_r} - Upsilon_r) / ||theta_{q_r} - Upsilon_r||_W
        w = T[:, q] - ups
        nw = np.linalg.norm(w)
        # [NDEE22] Remark 4.2 (Proper termination): line 10 is only executed if
        # nu_r != 0. If Upsilon_r = theta_{q_r} the residual vanishes,
        # e_{r-1} = 0 < delta, and the loop would already have stopped -- so
        # reaching here with nw == 0 means the snapshot is already exactly in
        # the cone.
        if nw <= 1e-14 * max(denom, 1.0):
            r -= 1
            I.pop()
            break
        gens.append(w / nw)
        # 11: K_r := K_{r-1} + Span^+{nu_r}
        K = np.column_stack(gens)

        # 12: q_{r+1} := argmax over q not in I_r of ||(I - Pi_{K_r}) theta_q|| / ||theta_{q_1}||
        rest = [j for j in range(Q) if j not in I]
        if not rest:
            e_r = 0.0
            errs.append(e_r)
            break
        res = [np.linalg.norm(T[:, j] - _proj(T[:, j], K)) / denom for j in rest]
        k = int(np.argmax(res))
        q = rest[k]
        # 13: e_r := ||(I - Pi_{K_r}) theta_{q_{r+1}}|| / ||theta_{q_1}||
        e_r = res[k]
        errs.append(e_r)

    # 15: R := r      16: return {nu_n}
    return ConeResult(np.column_stack(gens) if gens else np.zeros((T.shape[0], 0)),
                      I, errs, len(gens))


def e_orth(generators):
    """[NDEE22] Eq. (41): e_orth(r) = ||(I - Pi_{K_r})(nu_{r+1})||_W  <=  1.

    [NDEE22] §5.1: "the larger e_orth(r) (i.e. close to 1), the closer nu_{r+1}
    to being orthogonal to the cone". The paper reports
    e_orth^CPG(r) <= e_orth^mCPG(r), i.e. mCPG produces generators that are more
    nearly orthogonal to what came before -- a wider cone, hence a
    better-conditioned Gram matrix.
    """
    out = []
    for r in range(1, generators.shape[1]):
        K = generators[:, :r]
        nu = generators[:, r]
        out.append(float(np.linalg.norm(nu - _proj(nu, K))))
    return np.array(out)
