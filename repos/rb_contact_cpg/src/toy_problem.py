"""A parametrized obstacle problem, used to generate HF snapshots on CPU.

Paper: Benaceur, Ern, Ehrlacher, hal-02081485v2.

WHY THIS FILE EXISTS -- READ BEFORE INTERPRETING ANY RESULT
-----------------------------------------------------------
This is NOT from the paper. Section 6 runs on two-dimensional elastic
frictionless contact (Hertz half-disks, ring-on-block) discretized with
"a combination of Freefem++ [12] and Python", with continuous piecewise affine
elements on triangular meshes, a collocation or LAC constraint discretization,
and non-matching meshes. None of that is reproduced here.

What this file provides is the smallest problem with the right STRUCTURE to
exercise Algorithm 2: a parametrized minimization under an inequality
constraint whose Lagrange multipliers are non-negative and have a
parameter-dependent active set. That is what CPG compresses, and it is enough
to verify the algorithm end to end without a FEM stack.

Do not read accuracy numbers obtained here as reproductions of Section 6.

Deviations from the paper's setting, all deliberate:
  * 1-D obstacle problem, not 2-D linear elasticity (Section 3.1).
  * The constraint is LINEAR in u. The paper's is nonlinear, assumed
    quasi-linear (Section 2). With a linear constraint the Kacanov iteration of
    Eq. (12) converges in one step, so this file does not exercise it.
  * The geometry is parameter-independent, so the reference-domain mapping of
    Section 3.3 and the affine geometric mappings of Section 4.2 are absent.
  * No EIM: with an affine-in-mu operator, the separation of Section 4.3 is
    exact and trivial, so the EIM machinery is not exercised either.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def stiffness_1d(N):
    """1-D Dirichlet Laplacian, the stand-in for the elastic energy a(mu;.,.).

    Scaled by (N+1) so it approximates -d2/dx2 on a unit interval with N
    interior nodes.
    """
    h = 1.0 / (N + 1)
    main = 2.0 * np.ones(N)
    off = -1.0 * np.ones(N - 1)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / h**2


def obstacle_gap(N, mu):
    """Parametrized obstacle, playing the role of the initial gap g(mu).

    ``mu = (amplitude, centre)``: a smooth bump whose height and position vary
    with the parameter, so the contact (active) set moves as mu varies. That
    moving active set is precisely what makes the multiplier manifold hard to
    compress with a linear basis -- Section 5's argument for a cone.
    """
    amp, centre = mu
    x = np.linspace(0, 1, N + 2)[1:-1]
    return amp * (0.35 + 0.65 * np.exp(-40.0 * (x - centre) ** 2))


def solve_hf(A, f, gap):
    """Solve the HF saddle-point problem, the analogue of Eq. (9).

    Solves   min_u  1/2 u^T A u - f^T u   subject to   u <= gap,
    whose KKT system is the saddle-point problem of Eq. (9) with B = I:

        A u + lambda = f,    u <= gap,    lambda >= 0,    lambda^T (gap - u) = 0.

    Rather than iterate on the primal, we solve the dual, which is a
    bound-constrained QP in lambda alone:

        min_{lambda >= 0}  1/2 lambda^T Q lambda - lambda^T c
        with  Q = A^-1,  c = A^-1 f - gap

    and recover u = A^-1 (f - lambda). Solving in the dual guarantees
    lambda >= 0 exactly rather than up to a solver tolerance, which matters
    because ``cone_projected_greedy`` rejects negative snapshots.

    Returns
    -------
    u, lam : the primal and dual solutions of Eq. (9).
    """
    Ainv = np.linalg.inv(A)
    Q = Ainv
    c = Ainv @ f - gap

    def obj(lam):
        return 0.5 * lam @ Q @ lam - lam @ c

    def grad(lam):
        return Q @ lam - c

    n = len(f)
    res = minimize(
        obj, np.zeros(n), jac=grad, method="L-BFGS-B",
        bounds=[(0.0, None)] * n,
        options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12},
    )
    lam = np.maximum(res.x, 0.0)   # project away any round-off negativity
    u = Ainv @ (f - lam)
    return u, lam


def generate_snapshots(N=60, n_train=40, seed=0):
    """Build S_pri and S_du over a training set P_tr (Section 5, Task T2).

    Returns
    -------
    S_pri : (N, n_train) primal snapshots {u(mu)}
    S_du  : (N, n_train) dual snapshots  {lambda(mu)}, all entries >= 0
    params : (n_train, 2) the sampled mu values
    A, F : the operator and the per-parameter loads, for the reduced solve
    """
    rng = np.random.default_rng(seed)
    A = stiffness_1d(N)

    # P_tr: amplitude in [0.4, 1.2], centre in [0.25, 0.75].
    # [UNSPECIFIED-BY-ANALOGY] The paper samples its own parameter ranges
    # (Section 6.2); these are invented for this toy and correspond to nothing
    # in the paper.
    amps = rng.uniform(0.4, 1.2, n_train)
    centres = rng.uniform(0.25, 0.75, n_train)
    params = np.column_stack([amps, centres])

    f = np.full(N, 40.0)          # constant downward load
    S_pri, S_du, F = [], [], []
    for mu in params:
        gap = obstacle_gap(N, mu)
        u, lam = solve_hf(A, f, gap)
        S_pri.append(u)
        S_du.append(lam)
        F.append(f)

    return (np.array(S_pri).T, np.array(S_du).T, params, A, np.array(F).T)
