"""Generate a dataset of contact pressures (Lagrange multipliers).

Physics: frictionless normal contact between a rigid indenter and an elastic
half-plane (2-D plane strain / line contact). The contact pressure field is the
Lagrange multiplier of the non-interpenetration constraint -- the same object
Algorithm 2 of Benaceur/Ern/Ehrlacher compresses, and the same object §6.2 of
that paper computes for Hertz half-disks.

This is a boundary-integral (influence-coefficient) model, not the paper's FEM.
It is standard contact mechanics (Johnson, *Contact Mechanics*, Ch. 2), solved
here as a convex QP so the multipliers are non-negative by construction.

Surface normal displacement from a pressure distribution on a half-plane:

    u(x) = -(2 / (pi E*)) INT p(s) ln|x - s| ds + const

Discretized with piecewise-constant pressure on elements of half-width `a`,
the integral is analytic, giving the symmetric influence matrix C below.

Signorini conditions, with h the undeformed gap and delta the rigid approach:

    p >= 0,    h - delta + C p >= 0,    p . (h - delta + C p) = 0

which are the KKT conditions of

    min_{p >= 0}  1/2 p^T C p + p^T (h - delta)

a bound-constrained convex QP (C is SPD).

WHAT THIS IS NOT
----------------
Synthetic data from an idealized model. Not experimental measurements, not FEM
output, and not a reproduction of any figure in the paper. The half-plane model
assumes small strains, frictionless contact, and a half-space geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def influence_matrix(x, E_star):
    """Symmetric compliance matrix C for uniform-pressure elements.

    For an element of half-width `a` centred at x_j, the surface displacement at
    x_i depends only on d = x_i - x_j:

        INT ln|x-s| ds  =  (d+a) ln|d+a| - (d-a) ln|d-a| - 2a

    so  C_ij = -(2 / (pi E*)) [ (d+a) ln|d+a| - (d-a) ln|d-a| - 2a ].

    The 2-D half-plane displacement field is only defined up to an additive
    constant (the integral diverges logarithmically at infinity). That constant
    is absorbed into `delta`, so absolute approach values here are reference
    dependent -- the pressure DISTRIBUTION, which is what the dataset stores,
    is not affected.
    """
    a = 0.5 * (x[1] - x[0])
    d = x[:, None] - x[None, :]
    dp, dm = d + a, d - a
    # |d +/- a| vanishes only at element edges, never at a node pair, but guard
    # anyway so the log is finite.
    eps = 1e-300
    K = dp * np.log(np.abs(dp) + eps) - dm * np.log(np.abs(dm) + eps) - 2.0 * a
    return -(2.0 / (np.pi * E_star)) * K


def solve_contact(C, gap, delta):
    """Solve min_{p>=0} 1/2 p^T C p + p^T (gap - delta). Returns pressure p >= 0."""
    rhs = gap - delta

    def obj(p):
        return 0.5 * p @ C @ p + p @ rhs

    def jac(p):
        return C @ p + rhs

    res = minimize(
        obj, np.zeros(len(gap)), jac=jac, method="L-BFGS-B",
        bounds=[(0.0, None)] * len(gap),
        options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-14},
    )
    return np.maximum(res.x, 0.0)


def sample_gap(x, radius, offset, wav_amp, wav_len):
    """Undeformed gap profile h(x) >= 0.

    Parabolic indenter (the standard Hertz approximation to a cylinder of
    radius `radius`) plus optional sinusoidal waviness. Waviness is what turns
    a single contact patch into several disjoint ones, giving the dataset
    samples with genuinely varying support rather than one moving bump.
    """
    h = (x - offset) ** 2 / (2.0 * radius)
    if wav_amp > 0:
        h = h + wav_amp * (1.0 - np.cos(2.0 * np.pi * x / wav_len))
    return h - h.min()


def generate(n_samples=2000, N=128, L=1.0, E_star=1.0, seed=0, waviness=True):
    """Build the dataset.

    Returns
    -------
    P : (n_samples, N) float64, all entries >= 0
        Contact pressure fields -- the Lagrange multipliers.
    params : (n_samples, 5)
        Columns: radius, offset, wav_amp, wav_len, delta.
    x : (N,) node coordinates.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N)
    C = influence_matrix(x, E_star)

    rows, prm = [], []
    attempts = 0
    while len(rows) < n_samples:
        attempts += 1
        radius = rng.uniform(0.3, 3.0)
        offset = rng.uniform(-0.22, 0.22) * L
        if waviness and rng.random() < 0.6:
            wav_amp = rng.uniform(2e-4, 3e-3)
            wav_len = rng.uniform(0.08, 0.30) * L
        else:
            wav_amp, wav_len = 0.0, 1.0
        delta = rng.uniform(0.004, 0.05)

        gap = sample_gap(x, radius, offset, wav_amp, wav_len)
        p = solve_contact(C, gap, delta)

        # Reject degenerate samples: no contact at all, or contact running into
        # the domain edge (where the half-plane model's truncation dominates and
        # the pressure field is an artifact of the box, not the physics).
        act = p > 1e-10
        if act.sum() < 3:
            continue
        if act[0] or act[-1]:
            continue

        rows.append(p)
        prm.append([radius, offset, wav_amp, wav_len, delta])

    return np.array(rows), np.array(prm), x, attempts


def hertz_shape_error(x, p):
    """Relative deviation of a solved pressure from the semi-elliptical Hertz profile.

    Johnson, *Contact Mechanics* Ch. 4: for a parabolic indenter the contact
    patch is a strip |x - c| <= a carrying

        p(x) = p0 sqrt(1 - (x - c)^2 / a^2)

    We fit p0, a and c from the solved field itself and compare shapes, because
    the 2-D approach-to-load relation is reference dependent (see
    ``influence_matrix``) -- the shape is the falsifiable prediction, not the
    absolute magnitude.

    Returns ||p - p_hertz|| / ||p||, restricted to the contact patch.
    """
    act = np.flatnonzero(p > 1e-10)
    if act.size < 5:
        return np.nan
    lo, hi = x[act[0]], x[act[-1]]
    c, a = 0.5 * (lo + hi), 0.5 * (hi - lo)
    xa = x[act]
    ref = np.sqrt(np.clip(1.0 - ((xa - c) / a) ** 2, 0.0, None))
    # Least-squares amplitude, so only the shape is being compared.
    p0 = (ref @ p[act]) / (ref @ ref)
    return float(np.linalg.norm(p[act] - p0 * ref) / np.linalg.norm(p[act]))
