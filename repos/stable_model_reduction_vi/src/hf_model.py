"""High-fidelity model: 1-D obstacle problem with PARAMETER-DEPENDENT constraint.

Paper: Niakh, Drouet, Ehrlacher, Ern, "Stable model reduction for linear
variational inequalities with parameter-dependent constraints",
ESAIM: M2AN (2022), doi 10.1051/m2an/2022077.

This is a 1-D reduction of the membrane obstacle problem of §5.1. It is NOT the
paper's test case (which is 2-D, FreeFem++, 3594 primal / 467 dual dofs), but it
reproduces the one structural feature the whole paper turns on:

    the constraint operator b(mu; . , .) DEPENDS ON mu.

That is what makes the supremizer space S_R(mu) parameter-dependent (Eq. 17),
which is what forces the online enrichment of §2.3 to be redone for every new
parameter, which is the inefficiency §3 exists to remove. A test problem with a
parameter-independent constraint would make the paper's contribution invisible.

Setting, following §5.1
-----------------------
Domain Omega = (0,1), P1 finite elements, homogeneous Dirichlet conditions.
Obstacle active on a sub-interval omega(mu) = (c - r, c + r) with mu = (r, c),
mirroring §5.1 where omega(mu) is a disk of radius mu_1 centred at (mu_2, mu_3).

Following §5.1, the multiplier lives on a REFERENCE domain omega_hat, here
(-1, 1) with R_dim collocation points s_hat_i, mapped to physical positions

    s_i(mu) = c + r * s_hat_i.

The constraint is  u(s_i(mu)) >= psi_i, written in the paper's sign convention
b(mu; v, eta) <= g(mu; eta) as

    b(mu; v, eta) = sum_i eta_i * ( -v(s_i(mu)) ),      g(eta) = sum_i eta_i * ( -psi_i ).

§5.1 prescribes the obstacle elevation on the reference domain as
psi(mu)(x) = -1.25 * ((x-mu_2)^2 + (y-mu_3)^2) / mu_1^2. In 1-D at the mapped
point x = c + r*s_hat: (x-c)^2 / r^2 = s_hat^2, so

    psi_hat_i = -1.25 * s_hat_i^2

is PARAMETER-INDEPENDENT in reference coordinates, exactly as the paper's
psi_hat is. So B(mu) varies with mu while g does not. The paper notes the
coefficient -1.25 "is chosen so that the constraints at the boundary of
omega(mu) are inactive in order to avoid oscillations of the Lagrange
multiplier at the transition zone"; it is kept here for the same reason.

Deviations from the paper, all deliberate:
  * 1-D, not 2-D; P1 primal with collocation multipliers rather than the
    P1/P1 pair of §5.1.
  * a(mu; . , .) is taken PARAMETER-INDEPENDENT. In §5.1 it inherits mu through
    the geometric mapping h(mu) (Eq. 37). Making it mu-dependent would change
    none of the algorithms, since S_R(mu) depends on mu only through B(mu)
    (Eq. 15-17). Flagged again in REPRODUCTION_NOTES.md.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class ObstacleHF:
    """Assembles and solves the HF saddle-point problem (7)."""

    def __init__(self, n_elem=200, n_dual=40, load=-1.0, channel_gap=None):
        """
        channel_gap : float or None
            If given, adds an UPPER obstacle at psi_lo + channel_gap, so the
            membrane is confined to a channel:  psi_lo <= u <= psi_lo + gap.

            This is not in the paper, and it exists for a specific reason. With
            a single lower obstacle every constraint row is -phi(s_i) with the
            P1 basis phi >= 0, so every supremizer T(mu)chi_r lies in one
            halfspace and NO non-negative combination of them can cancel. By
            Gordan's theorem that makes beta^dec > 0 no matter how small N is,
            so the instability the paper's §3 repairs simply cannot arise --
            the dimension count N < R is necessary but not sufficient, because
            the inf in Eq. (14) is over a CONE, not a subspace.

            A channel gives constraint rows of both signs: the lower block
            pushes the membrane up, the upper block pushes it down, and a
            non-negative combination CAN cancel. That is what makes beta^dec
            able to vanish, and hence what makes PGA's stabilization testable.
            See REPRODUCTION_NOTES.md.
        """
        self.n_elem = n_elem
        self.h = 1.0 / n_elem
        # Interior nodes only (homogeneous Dirichlet, u = 0 on Gamma).
        self.nodes = np.linspace(0.0, 1.0, n_elem + 1)
        self.free = np.arange(1, n_elem)          # indices of interior nodes
        self.n_v = len(self.free)

        # a(u,v) = INT u' v'  ->  P1 stiffness. This is also the V inner
        # product: we equip V with the energy norm, so G_V = K.
        self.K = self._stiffness()
        # f(v) = INT l v  with l constant (§5.1 uses l_hat = -1).
        self.f = load * self.h * np.ones(self.n_v)

        # Reference obstacle domain omega_hat = (-1,1) with collocation points.
        # Endpoints excluded so the constraint is not evaluated exactly at the
        # boundary of omega(mu).
        self.s_hat = np.linspace(-1.0, 1.0, n_dual + 2)[1:-1]
        self.n_w = n_dual
        # psi_hat, parameter-independent -- see module docstring.
        #
        # RETUNED FOR 1-D. §5.1 uses psi_hat(s) = -1.25 s^2, whose peak is 0 --
        # exactly the Dirichlet level. In 1-D that makes the contact set a
        # single point (u - psi is then strictly convex), so the multiplier
        # carries no spatial structure and the cone algorithms have nothing to
        # compress. The paper states its own coefficient was picked for the same
        # kind of reason: it "is chosen so that the constraints at the boundary
        # of omega(mu) are inactive in order to avoid oscillations of the
        # Lagrange multiplier at the transition zone."
        #
        # Here the obstacle is offset below the Dirichlet level and flattened:
        #     psi_hat(s) = -(offset + curv * s^2)
        # with offset < |u_free|_max so the centre is in contact, and
        # offset + curv > |u_free|_max so the rim is not. The unconstrained
        # solution of -u'' = -1 with u(0)=u(1)=0 is u = (x^2-x)/2, min -0.125,
        # which is what these two numbers are set against.
        # Sizing: on the contact interval the membrane follows psi, outside it
        # is a parabola of curvature 1 rising to u(0)=u(1)=0. Matching value and
        # slope at the free boundary gives the contact half-width
        #     d = [1 - sqrt(1 - 4(0.125 - offset)/(k + 0.5))] / 2,   k = curv/r^2,
        # so curv must be well below r^2 (here r^2 in 0.01..0.048) for the
        # contact set to be an interval rather than a point. These values put
        # the contact patch at roughly 55-70% of the obstacle window across the
        # parameter range, and it moves and resizes with mu -- which is what
        # gives the dual snapshots something to compress.
        self.psi_offset, self.psi_curv = 0.060, 0.005
        self.psi_hat = -(self.psi_offset + self.psi_curv * self.s_hat**2)

        self.channel_gap = channel_gap
        if channel_gap is None:
            self.n_w = n_dual
            self.g = -self.psi_hat
        else:
            # Upper obstacle ("ceiling") on two FLANK bands, disjoint from the
            # obstacle window. It must not overlap the window: the floor forces
            # u >= -0.06 there while the ceiling forces u <= level < -0.06, which
            # would be infeasible. The flanks are where the lifted membrane rises
            # against the ceiling, so both blocks are active at once -- with
            # opposite signs, which is the whole point.
            n_c = n_dual // 2
            self.ceil_pts = np.concatenate([
                np.linspace(0.12, 0.30, n_c),
                np.linspace(0.70, 0.88, n_c),
            ])
            self.ceil_level = channel_gap      # a height, e.g. -0.11
            self.n_w = n_dual + 2 * n_c
            self.g = np.concatenate([
                -self.psi_hat,
                np.full(2 * n_c, self.ceil_level),
            ])

    def _stiffness(self):
        main = 2.0 * np.ones(self.n_v) / self.h
        off = -1.0 * np.ones(self.n_v - 1) / self.h
        return np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    def _p1_row(self, x):
        """Row vector of P1 basis functions evaluated at physical point x.

        Gives v(x) = row @ u for a coefficient vector u on the interior nodes.
        """
        row = np.zeros(self.n_v)
        if x <= 0.0 or x >= 1.0:
            return row                      # outside: v = 0 by the BC
        e = min(int(x / self.h), self.n_elem - 1)
        xi = (x - self.nodes[e]) / self.h   # local coordinate in [0,1]
        for node, val in ((e, 1.0 - xi), (e + 1, xi)):
            if 1 <= node <= self.n_elem - 1:
                row[node - 1] = val
        return row

    def B(self, mu):
        """Constraint matrix B(mu), with b(mu; v, eta) = eta^T B(mu) v.

        Row i evaluates -v at the mapped collocation point s_i(mu), so that
        b(mu; v, eta) <= g(eta) encodes v(s_i(mu)) >= psi_hat_i.

        THIS is the parameter-dependent object the paper is about.
        """
        r, c = mu
        pts = c + r * self.s_hat
        lower = -np.array([self._p1_row(x) for x in pts])
        if self.channel_gap is None:
            return lower
        # Upper block: +phi(x_j), encoding u(x_j) <= ceil_level. Opposite sign
        # to the lower block -- that is the point (see __init__). These points
        # are at fixed physical locations; the mu-dependence of B(mu) comes
        # entirely from the lower block, which is enough.
        upper = np.array([self._p1_row(x) for x in self.ceil_pts])
        return np.vstack([lower, upper])

    def solve(self, mu):
        """Solve the HF saddle-point problem (7). Returns (u, lam), lam >= 0.

        (7) is    a(u,v) + b(v,lambda) = f(v),    b(u,eta) <= g(eta),
        i.e. the KKT system of  min_{B u <= g} 1/2 u^T K u - f^T u.

        Solved in the dual, min_{lam >= 0} 1/2 lam^T (B K^-1 B^T) lam
        - lam^T (B K^-1 f - g), so that lambda >= 0 holds exactly rather than up
        to a solver tolerance -- the cone algorithms downstream require it.
        """
        Bm = self.B(mu)
        Kinv_f = np.linalg.solve(self.K, self.f)
        KiBt = np.linalg.solve(self.K, Bm.T)
        Q = Bm @ KiBt
        c = Bm @ Kinv_f - self.g
        # Q is positive semi-definite; a small shift keeps L-BFGS-B well posed
        # when the collocation rows are nearly dependent. It biases lambda by
        # O(reg) and is not from the paper.
        reg = 1e-12 * np.trace(Q) / max(len(c), 1)
        Qr = Q + reg * np.eye(len(c))

        res = minimize(
            lambda a: 0.5 * a @ Qr @ a - a @ c,
            np.zeros(len(c)),
            jac=lambda a: Qr @ a - c,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(c),
            options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-14},
        )
        lam = np.maximum(res.x, 0.0)
        u = np.linalg.solve(self.K, self.f - Bm.T @ lam)
        return u, lam

    def snapshots(self, params):
        """Compute {(u(mu_p), lambda(mu_p))} over a training set (§2.2)."""
        U, L = [], []
        for mu in params:
            u, lam = self.solve(mu)
            U.append(u)
            L.append(lam)
        return np.array(U).T, np.array(L).T


def training_set(n_r=5, n_c=5, r_range=(0.10, 0.22), c_range=(0.40, 0.60)):
    """Tensor-product training set, in the style of D_train in §5.1."""
    rs = np.linspace(r_range[0], r_range[1], n_r)
    cs = np.linspace(c_range[0], c_range[1], n_c)
    return np.array([[r, c] for r in rs for c in cs])


def validation_set(n=32, r_range=(0.10, 0.22), c_range=(0.40, 0.60), seed=1):
    """Random validation set, as D_valid in §5.1 ("chosen ... randomly")."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(r_range[0], r_range[1], n),
        rng.uniform(c_range[0], c_range[1], n),
    ])
