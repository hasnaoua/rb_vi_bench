"""How much of the *reachable* cone does a reduced cone capture?

Every error metric elsewhere scores a cone against the finite snapshot set. But the object
a dual basis is meant to represent is not that finite set -- it is the whole cone those
snapshots generate,

    K_full = span_+{theta_1, ..., theta_Q},

since any non-negative combination of admissible multipliers is itself an admissible
multiplier. A method can reproduce every training snapshot exactly and still capture a
thin sliver of ``K_full``; conversely a wider cone covers combinations never seen in
training. That difference is invisible to a snapshot-wise error, and it is what these
three statistics measure.

**1. Coverage** -- the direct answer to "how much of the full cone do we span". Sample
points inside ``K_full`` (random non-negative combinations of *all* training snapshots),
project each onto ``K_R``, and report the relative residual. Zero means ``K_R`` reproduces
everything ``K_full`` can express; large means the reduced cone is a narrow slice of it.
This generalizes the snapshot projection error from the finite set to its whole conic
hull, and it is the statistic that can distinguish two methods whose snapshot errors are
identical.

**2. Aperture** -- how *wide* the cone opens, as pairwise angles between normalized
generators. This is the quantity [NDEE22] §4 is really after when it asks for a
well-conditioned Gram matrix, and it is the cone-level analogue of ``e_orth``: ``e_orth``
measures each generator against the cone that preceded it, aperture measures all
generators against each other. Wider is better conditioned, but wider is *not*
automatically better coverage -- a cone can open wide in directions the snapshots never
occupy.

**3. Reach outside K_full** -- does ``K_R`` extend *beyond* the snapshot cone? For CPG and
ADG it cannot: their generators *are* snapshots, so ``K_R`` is a sub-cone of ``K_full`` by
construction and this is zero to round-off. **mCPG is different, and measurably so.**

Its generators are ``nu_r = (theta_q - Upsilon_r)/||.||`` with
``Upsilon_r in K_{r-1} cap (theta_q - W^+)``. The second constraint forces
``nu_r >= 0`` -- membership of the non-negative orthant ``W^+``, which is the property the
method actually needs and which line 9 exists to enforce. But a *difference* of two
elements of ``span_+{theta}`` need not lie in ``span_+{theta}``, so mCPG can and does
leave the snapshot cone while staying inside ``W^+``.

On ``toy_bee20`` at R=8, two of eight mCPG generators are outside ``span_+{theta}``, with
relative distances 0.15 and 0.82; on ``fem_lambda`` most of them are. Confirmed
independently of the NNLS residual by LP feasibility
(``exists c >= 0 : Theta c = nu_r``), which is infeasible for exactly those generators.
This means [NDEE22] Remark 4.3's parenthetical -- that ``nu_r`` lies "in fact of
``Span^+({theta_{q_n}})``" -- does **not** hold as stated; what holds is the ``W^+``
membership the same sentence claims first, and that is the part the construction rests on.

It is **data-dependent, not universal**: on smooth synthetic bumps every mCPG generator
stays inside the snapshot cone. So this is a property of the interaction between the
algorithm and the data, not a defect that shows up everywhere -- which is precisely why
it needs measuring per dataset rather than asserting once.

So the answer to "is mCPG's cone larger" is yes, and in a specific sense: it reaches
directions no non-negative combination of the training snapshots can reach. Whether that
is an advantage is what coverage measures -- reaching outside ``K_full`` is only useful if
the multipliers you meet at run time also lie outside it.

Read 1 and 2 together. Coverage alone rewards a cone for being large in any direction;
aperture alone rewards width without asking whether the width is useful.
"""

from __future__ import annotations

import numpy as np

from .. import _paths  # noqa: F401
from ..types import BasisResult, Dataset

from rb_vi_common.cone_projection import project_onto_cone


def _sample_cone(snapshots: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Draw points from ``span_+{snapshots}``, normalized to the unit sphere.

    Coefficients are Dirichlet(1) -- uniform on the simplex -- so the samples are spread
    over the conic hull rather than concentrated near a single generator. Sampling
    exponentials and normalizing would bias toward the centroid; sampling a fixed sparse
    support would bias toward the faces. Directions are what matter, since the projection
    residual is scale-invariant, so each sample is normalized.
    """
    rng = np.random.default_rng(seed)
    Q = snapshots.shape[1]
    coeffs = rng.dirichlet(np.ones(Q), size=n_samples).T          # (Q, n_samples)
    pts = snapshots @ coeffs
    norms = np.linalg.norm(pts, axis=0)
    keep = norms > 0
    return pts[:, keep] / norms[keep]


def coverage(dataset: Dataset, generators: np.ndarray, *,
             n_samples: int = 48, seed: int = 0) -> dict[str, float]:
    """Relative residual of random points of ``K_full`` projected onto ``K_R``."""
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {}
    pts = _sample_cone(dataset.train(), n_samples, seed)
    if pts.shape[1] == 0:
        return {}
    res = np.empty(pts.shape[1])
    for k in range(pts.shape[1]):
        proj, _ = project_onto_cone(pts[:, k], G, mass=None)
        res[k] = np.linalg.norm(pts[:, k] - proj)      # samples are unit norm
    return {
        "cover_mean_err": float(res.mean()),
        "cover_max_err": float(res.max()),
        # Fraction of the sampled cone represented to better than 10%: a coarse but
        # readable "how much of K_full is effectively inside K_R".
        "cover_frac_within_10pct": float(np.mean(res <= 0.10)),
    }


def excess(dataset: Dataset, generators: np.ndarray, *,
           n_samples: int = 48, seed: int = 1) -> dict[str, float]:
    """The other half of the comparison: how far ``K_R`` sticks out of ``K_full``.

    ``coverage`` samples ``K_full`` and projects onto ``K_R``, so it measures what the
    reduced cone **misses** -- how much too *small* it is. This samples ``K_R`` and
    projects onto ``K_full``, measuring what the reduced cone **adds** -- how much too
    *large* it is. The two are not redundant and neither implies the other: a cone can
    cover ``K_full`` perfectly while extending far beyond it, or sit strictly inside it
    while missing most of it.

    Together they bracket the cones from both sides, and
    ``max(cover_max_err, excess_max_err)`` is a Hausdorff-type distance between them on
    the unit sphere -- zero exactly when the two cones coincide.

    This is strictly stronger than the generator-level ``reach_outside``. That checks
    only the extreme rays; a cone whose generators all lie in ``K_full`` necessarily has
    zero excess, but the converse direction carries real information -- excess measures
    how much *volume* lies outside, not merely how many rays do.
    """
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {}
    pts = _sample_cone(G, n_samples, seed)
    if pts.shape[1] == 0:
        return {}
    full = dataset.train()
    res = np.empty(pts.shape[1])
    for k in range(pts.shape[1]):
        proj, _ = project_onto_cone(pts[:, k], full, mass=None)
        res[k] = np.linalg.norm(pts[:, k] - proj)      # samples are unit norm
    return {
        "excess_mean_err": float(res.mean()),
        "excess_max_err": float(res.max()),
        "excess_frac_outside_10pct": float(np.mean(res > 0.10)),
    }


def aperture(generators: np.ndarray) -> dict[str, float]:
    """Pairwise angles between normalized generators, in degrees."""
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] < 2:
        return {}
    norms = np.linalg.norm(G, axis=0)
    U = G / np.where(norms > 0, norms, 1.0)
    C = np.clip(U.T @ U, -1.0, 1.0)
    iu = np.triu_indices(C.shape[0], k=1)
    ang = np.degrees(np.arccos(C[iu]))
    return {
        "aperture_mean_deg": float(ang.mean()),
        "aperture_max_deg": float(ang.max()),
        "aperture_min_deg": float(ang.min()),
    }


#: A generator is counted as outside ``K_full`` past this relative NNLS residual. Well
#: above round-off (selected-snapshot generators land at ~1e-16) and well below the
#: distances actually observed for mCPG (0.15 and 0.82 on toy_bee20), so the count is not
#: sensitive to where in that gap the line is drawn.
OUTSIDE_TOL = 1e-8


def reach_outside(dataset: Dataset, generators: np.ndarray) -> dict[str, float]:
    """How far, and how often, generators sit outside ``span_+{training snapshots}``.

    Zero by construction for methods whose generators are selected snapshots. Non-zero
    for mCPG, which builds residual generators that stay in ``W^+`` without staying in the
    snapshot cone -- see the module docstring.

    ``min_entry`` is reported alongside because it is the property that actually has to
    hold: a cone method whose generators left ``W^+`` could not preserve ``lambda >= 0``,
    and that would be a bug rather than a measurement.
    """
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {}
    full = dataset.train()
    worst = 0.0
    n_out = 0
    for r in range(G.shape[1]):
        g = G[:, r]
        n = np.linalg.norm(g)
        if n <= 0:
            continue
        proj, _ = project_onto_cone(g, full, mass=None)
        rel = float(np.linalg.norm(g - proj) / n)
        worst = max(worst, rel)
        n_out += int(rel > OUTSIDE_TOL)
    return {
        "outside_K_full_max": worst,
        "outside_K_full_frac": float(n_out / G.shape[1]),
        "min_entry": float(G.min()),
    }


#: Above this many generators the sampled statistics are skipped.
#:
#: Every one of them costs an NNLS solve against a ``dim x R`` matrix, so the work grows
#: superlinearly in ``R`` while the answer stops being informative: a cone with thousands
#: of generators is essentially ``W^+`` already, so its coverage is ~0 and its excess is
#: ~maximal by construction, whatever the method. The case that forces this is the
#: orthant baseline in tolerance mode -- on ``physics`` (dim 7676) it needs R = 5001 at
#: delta = 0.5 and R = 7351 at delta = 0.01, which is itself the informative result and is
#: still reported. Only the O(R) sampled geometry is dropped, with a reason recorded.
MAX_R_FOR_SAMPLING = 512


def evaluate(dataset: Dataset, result: BasisResult, *,
             n_samples: int = 48, seed: int = 0) -> dict[str, float]:
    """Cone-geometry row for one (dataset, method) cell."""
    if result.R > MAX_R_FOR_SAMPLING:
        # Absent, not zero: a zero here would read as "covers K_full perfectly".
        return {"cone_geometry_skipped_R": float(result.R)}
    row: dict[str, float] = {}
    row.update(coverage(dataset, result.generators, n_samples=n_samples, seed=seed))
    row.update(excess(dataset, result.generators, n_samples=n_samples, seed=seed + 1))
    row.update(aperture(result.generators))
    row.update(reach_outside(dataset, result.generators))
    # Two-sided: zero exactly when K_R and K_full coincide. Reporting only one direction
    # lets a cone look perfect while being much too small, or much too large.
    if "cover_max_err" in row and "excess_max_err" in row:
        row["cone_hausdorff"] = max(row["cover_max_err"], row["excess_max_err"])
    return row
