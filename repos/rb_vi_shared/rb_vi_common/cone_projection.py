"""L2-orthogonal projection onto a finitely generated convex cone.

Origin: [BEE20] §5 (see ``rb_vi_common/__init__.py`` for the tag key and for the
v2-numbering caveat that applies to every [BEE20] number below).

This module implements the operator written  Pi_{K_n^+}  in [BEE20], used in
[BEE20] Eq. (57) and in line 5 of [BEE20] Algorithm 2. Everything else in the CPG
algorithm is cheap; this projection is the numerical core.

[NDEE22] uses the same operator (written Pi_{K_r}) in [NDEE22] Algorithm 2 lines
12-13 and in [NDEE22] Eq. (41). It works in whitened coordinates, where W is
Euclidean, so it calls this with ``mass=None`` -- see ``reduction.Whitener``. The
``mass`` argument exists for [BEE20], which carries the Gram matrix explicitly.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import nnls


def project_onto_cone(lam, generators, mass=None):
    """Project ``lam`` onto span_+{generators} in the ||.||_Lambda norm.

    [BEE20] §5, after Eq. (57), defines the cone

        K_{n-1}^+ = span_+ {lambda(mu_1), ..., lambda(mu_{n-1})}

    where span_+ restricts linear combinations to NON-NEGATIVE coefficients,
    and Pi_{K^+} as the "L2-orthogonal projector onto the convex cone K^+".

    Writing G for the matrix whose columns are the generators, that projector is
    the solution of

        min_{c >= 0}  || lam - G c ||_Lambda

    which is exactly a non-negative least squares problem. It is a projection
    onto a convex cone, not a subspace, so it is NOT linear in ``lam`` -- there
    is no projection matrix to precompute.

    Parameters
    ----------
    lam : (R,) array
        The Lagrange-multiplier snapshot to project.
    generators : (R, n) array
        Columns generate the cone. ``n == 0`` encodes K_0^+ = {0}
        ([BEE20] Algorithm 2, line 2; [NDEE22] Algorithm 2, line 3).
    mass : (R, R) array, optional
        Gram matrix of the ||.||_Lambda inner product. See ``norm_Lambda``.

    Returns
    -------
    proj : (R,) array
        The projection Pi_{K^+}(lam).
    coeffs : (n,) array
        The non-negative coefficients c. Returned because the greedy algorithms
        need only the residual, but callers inspecting sparsity find them useful.
    """
    lam = np.asarray(lam, dtype=float)
    G = np.asarray(generators, dtype=float)

    # [BEE20] Algorithm 2, line 2: K_0^+ := {0}. The projection onto the zero
    # cone is 0, which makes the first selection (line 5) reduce to [BEE20]
    # Eq. (56), mu_1 in argmax ||lambda(mu)||_Lambda. [BEE20] states (56)
    # separately; handling K_0 here makes the loop uniform and reproduces (56)
    # exactly. [NDEE22] Algorithm 2 line 3 sets K_0 := {0} identically.
    if G.size == 0 or G.shape[1] == 0:
        return np.zeros_like(lam), np.zeros(0)

    if mass is None:
        A, b = G, lam
    else:
        # ||x||_Lambda^2 = x^T M x. With the Cholesky factor M = U^T U,
        # ||x||_Lambda = ||U x||_2, so the weighted problem becomes an ordinary
        # NNLS on the transformed operator.
        U = cholesky(np.asarray(mass, dtype=float), lower=False)
        A, b = U @ G, U @ lam

    coeffs, _ = nnls(A, b)
    return G @ coeffs, coeffs


def norm_Lambda(x, mass=None):
    """The norm ||.||_Lambda used throughout [BEE20] §5.

    [BEE20] §5 (text following Eq. (56)): "The most natural choice for the norm
    on the Lagrange multipliers is ||.||_Lambda = ||.||_{L2(Gamma_c)}. However,
    at the discrete level, one can also consider the choice
    ||.||_Lambda = ||.||_{linf(Gamma_c,tr)}."

    We implement the L2 choice, which [BEE20] calls the natural one.

    [UNSPECIFIED] Which of the two norms produced the reported numerical results
    in [BEE20] §6 is not stated.
    Using: L2, i.e. the "most natural" choice named first in the text.
    Alternatives: linf over the discrete subset Gamma_c,tr. Note that switching
    to linf changes the projection in [BEE20] Eq. (57) from a non-negative least
    squares problem into a linear program, so ``project_onto_cone`` would need a
    different solver -- it is not a drop-in norm swap.

    ``mass=None`` uses the Euclidean norm, which is the L2(Gamma_c) norm only if
    the contact-boundary FE basis is orthonormal. Pass the boundary mass matrix
    for a genuine L2(Gamma_c) norm.

    [NDEE22] does not face this choice: it whitens coordinates once
    (``reduction.Whitener``) so that ||.||_W is Euclidean by construction, which
    is the ``mass=None`` branch here.
    """
    x = np.asarray(x, dtype=float)
    if mass is None:
        return float(np.linalg.norm(x))
    return float(np.sqrt(x @ np.asarray(mass, dtype=float) @ x))
