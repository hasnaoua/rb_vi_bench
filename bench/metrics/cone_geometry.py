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

**2. Extent** -- how much space the cone actually encloses. Cut ``K_R`` by a hyperplane
shared by every method and measure the resulting section: a length in 2-D, an area in
3-D, an ``(R-1)``-volume in general, reported against what ``R`` mutually orthogonal
directions would give under the same cut. See ``section_volume``.

This replaces *mean pairwise angle* as the width statistic, because a mean over edges is
not an extent. It saturates once generators are mutually well separated, and a generator
added strictly *inside* the existing cone enlarges the enclosed region not at all while
still moving the mean -- so two cones with equal mean aperture can enclose very different
amounts of space. The section volume answers the question directly and degenerates to
exactly 0 when the generators become linearly dependent, which is the honest answer.

``aperture_mean_deg`` is still computed and still in the CSV, now labelled as what it
is: a **conditioning** diagnostic, the quantity [NDEE22] §4 is after when it asks for a
well-conditioned Gram matrix, and the cone-level analogue of ``e_orth``. Neither extent
nor aperture implies good coverage -- a cone can open wide, or enclose much space, in
directions the snapshots never occupy.

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
extent alone rewards enclosed space without asking whether that space is useful.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

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
    """Pairwise angles between normalized generators, in degrees.

    Retained as a *conditioning* diagnostic, which is what [NDEE22] §4 is after, and not
    as a measure of how much space the cone encloses -- see ``section_volume`` for that.
    A mean over edges is not an extent: it saturates once the generators are mutually
    well separated, and it barely moves when a generator is added *inside* the cone,
    which changes the enclosed region not at all but does change the conditioning.
    """
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


#: The hyperplane every cone is cut by, as the unit normal ``u`` with ``<x, u> = h``.
#:
#: ``u = 1/sqrt(m) * (1,...,1)``. The requirement is that the cut be *transversal* to
#: every cone being compared -- ``<g, u> > 0`` for every generator ``g`` -- or the
#: cross-section runs off to infinity and has no finite measure. Every generator here
#: lives in the non-negative orthant (snapshots are multipliers, the orthant baseline's
#: are canonical axes, NMF's atoms are non-negative), and the all-ones direction is
#: strictly positive on all of ``W^+ \ {0}``. It is also the one choice that treats every
#: coordinate alike, so no node is privileged by the measurement.
#:
#: Rejected alternatives: the mean snapshot direction is data-dependent, so the axis
#: would differ per dataset and the cut would tilt toward whatever the training set
#: happens to emphasize; the cone's own axis would differ per METHOD, which destroys the
#: comparison the metric exists to make. POD is the one basis this cannot measure -- its
#: modes have mixed signs, so ``<g, u>`` can vanish or go negative and the section is
#: unbounded. That returns nan rather than a number, which is correct: an unbounded
#: region has no volume, and POD is a negative control for exactly this reason.
SECTION_HEIGHT = 1.0


def section_volume(generators: np.ndarray) -> dict[str, float]:
    """Measure of the cone's cross-section at a common height -- how much space it encloses.

    Cut ``K_R = span_+{g_1..g_R}`` by the hyperplane ``<x, u> = h`` shared by every method
    (see ``SECTION_HEIGHT``). The section is the convex hull of ``p_i = h g_i / <g_i, u>``,
    a simplex whose ``(R-1)``-dimensional volume is the natural notion of *extent*: a disk
    area in the 3-D case, a segment length in 2-D, an ``(R-1)``-volume in general. Unlike a
    mean pairwise angle this responds to the whole configuration -- a generator added
    inside the existing cone contributes no volume, which is the correct answer.

    Each ``p_i`` is invariant to rescaling ``g_i``, so generator normalization cannot
    change the result.

    **Raw volumes at different R are not comparable, and are not reported as if they
    were.** An ``(R-1)``-volume and an ``R``-volume are quantities of different dimension;
    the sequence over R is not a curve of one quantity. Two things follow:

    * comparisons are *vertical* -- between methods at the same R, which is the
      per-iteration grouping this metric is built for and where the dimensions agree;
    * the reported number is a **ratio**, ``V / V_ortho``, against the section of ``R``
      mutually orthogonal directions cut by the same hyperplane. That reference is the
      widest configuration available inside ``W^+`` (every pair at 90 degrees, which is
      the orthant baseline), it has the same dimension as the cone being measured, and
      dividing cancels the units. The result is dimensionless and lands in [0, 1]:
      1 means "as wide as R directions can be", 0 means the generators are linearly
      dependent and the cone is degenerate.

    **Two forms of the ratio are reported, and the per-dimension one is what to plot.**
    ``section_vol_ratio`` is the volume ratio itself. It is a product of ``R-1`` width
    factors, each below 1, so it decays geometrically in R: across the benchmark it spans
    1e-139 to 1e-2, which no single axis can show and which mostly measures *R* rather
    than the method. ``section_width_ratio = (V / V_ortho)^(1/(R-1))`` is its geometric
    mean -- the typical width per generator direction, in the same units as a single
    ratio rather than a product of them. It stays in [0, 1] (orthogonal directions still
    give exactly 1), it does not drift with R merely because R grew, and in practice it
    lands between 0.001 and 0.3, which is readable. It is the ``R``-th root of a volume,
    the same normalization that turns a volume into a mean radius.

    Computed in logs throughout. Both ``V`` and ``V_ortho`` contain ``1/(R-1)!`` and a
    factor ``m^{(R-1)/2}``, which overflow and underflow long before R reaches 40 --
    and both cancel exactly in the ratio, so the ratio is well-conditioned even where
    neither volume is representable. Singular values are used rather than a determinant
    so a rank-deficient section gives 0 instead of a failure.
    """
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] < 2:
        # A single ray sections to a point. Its 0-volume is 1 by convention, which would
        # read as a full-width cone; nan says "not defined here", as gram_cond does.
        return {"section_vol_ratio": float("nan"), "section_log_volume": float("nan")}

    m, R = G.shape
    axis_inner = G.sum(axis=0) / np.sqrt(m)          # <g_i, u> for u = 1/sqrt(m) * ones
    if not np.all(axis_inner > 0):
        # Unbounded section: the hyperplane does not cut this cone transversally.
        return {"section_vol_ratio": float("nan"), "section_log_volume": float("nan")}

    P = SECTION_HEIGHT * G / axis_inner              # vertices of the section
    M = P[:, 1:] - P[:, :1]                          # edges from the first vertex
    sv = np.linalg.svd(M, compute_uv=False)

    # log V = -lgamma(R) + sum(log s_i); the reference differs only in its edge lengths.
    log_simplex = -float(gammaln(R))
    if np.any(sv <= 0):
        return {"section_vol_ratio": 0.0, "section_log_volume": float("-inf")}
    log_vol = log_simplex + float(np.sum(np.log(sv)))

    # R orthonormal axes under the same cut: vertices sqrt(m) h e_i, whose edge Gram is
    # m h^2 (I + J), with det = m^{R-1} h^{2(R-1)} R.
    log_vol_ortho = log_simplex + 0.5 * ((R - 1) * np.log(m) + np.log(R)) \
        + (R - 1) * np.log(SECTION_HEIGHT)
    log_ratio = log_vol - log_vol_ortho
    return {
        "section_vol_ratio": float(np.exp(log_ratio)),
        "section_width_ratio": float(np.exp(log_ratio / (R - 1))),
        "section_log_volume": float(log_vol),
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
    row.update(section_volume(result.generators))
    row.update(reach_outside(dataset, result.generators))
    # Two-sided: zero exactly when K_R and K_full coincide. Reporting only one direction
    # lets a cone look perfect while being much too small, or much too large.
    if "cover_max_err" in row and "excess_max_err" in row:
        row["cone_hausdorff"] = max(row["cover_max_err"], row["excess_max_err"])
    return row
