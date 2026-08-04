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
shared by every method and take the **mean width** of the resulting section, averaged over
directions of the *ambient* space, as a fraction of the same quantity for ``K_full``.
See ``section_extent``.

This replaces *mean pairwise angle* as the width statistic, because a mean over edges is
not an extent. It saturates once generators are mutually well separated, and a generator
added strictly *inside* the existing cone enlarges the enclosed region not at all while
still moving the mean -- so two cones with equal mean aperture can enclose very different
amounts of space.

Two properties are non-negotiable here and the metric is built to have them. The measure
is taken in the **dimension of the space**, not the number of generators: ``R`` rays in
``R^m`` are not a basis, and letting ``R`` set the dimension of the yardstick measures the
method's cardinality rather than its cone. And it is **monotone in R** -- ``K_R`` is a
sub-cone of ``K_{R+1}``, so the section is a subset of the next section and adding a
generator can only enlarge it. A statistic that falls there is reporting on its own
normalization. ``section_extent`` rises by construction, per Monte-Carlo realization and
not merely in expectation.

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

from functools import lru_cache

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


def section_extent(dataset: Dataset, generators: np.ndarray, *,
                   n_samples: int = 512, seed: int = 0) -> dict[str, float]:
    """How much space the cone encloses, measured in the ambient space and monotone in R.

    Cut ``K_R`` by the hyperplane ``<x, u> = h`` shared by every method (see
    ``SECTION_HEIGHT``). The section is the convex hull of ``p_i = h g_i / <g_i, u>``.
    What is reported is its **mean width**, against the mean width of the section of
    ``K_full`` cut by the same hyperplane.

    **Why not the volume of that hull.** An earlier version measured its
    ``(R-1)``-dimensional volume, which is wrong twice over. The generators of a cone are
    not a basis -- they are ``R`` rays in ``R^m``, usually with ``R << m`` and no claim to
    span anything -- so taking ``R-1`` as the dimension of the measurement builds the
    method's own cardinality into the yardstick. And because that dimension moved with
    ``R``, the reported number could *fall* when a generator was added. That is impossible
    for the object being measured: ``K_R`` is a sub-cone of ``K_{R+1}``, so the section is
    a subset of the next section and can only grow. A statistic that decreases there is
    measuring the yardstick, not the cone.

    Measuring in the ambient space instead runs into the reason the volume was tempting:
    the section of an ``R``-generator cone is at most ``(R-1)``-dimensional, so its
    ``(m-1)``-volume inside the hyperplane is *exactly* zero for every case here
    (``R <= 40``, ``m`` from 40 to 7676). Every ambient-dimensional volume, solid angle or
    covered-sphere-fraction vanishes identically for the same reason. The quantity has to
    be one that stays positive on a lower-dimensional body.

    **Mean width is that quantity.** For a convex body ``S``,

        w(S) = E_g [ max_i <p_i, g> - min_i <p_i, g> ],    g ~ N(0, I_m),

    the average extent of ``S`` over directions of the ambient space. It has exactly the
    properties the volume lacked:

    * **Ambient.** The dimension enters through ``g ~ N(0, I_m)`` -- the space's dimension,
      not the generator count. Nothing assumes the generators are independent, and adding
      a redundant one is handled correctly rather than making the object degenerate.
    * **Monotone, by construction.** More vertices means a maximum over a superset, so the
      integrand rises pointwise in ``g``. Adding a generator can never decrease it. The
      Gaussian directions are drawn from a fixed seed and cached, so the same ``g`` are
      used at every ``R`` and the monotonicity holds *per realization*, not merely in
      expectation -- Monte-Carlo noise cannot manufacture a decrease.

      The guarantee is about **nested** cones, and it is worth being precise about that,
      because one method here is not nested. Every greedy method appends to the cone it
      already has, so ``K_R`` really is a sub-cone of ``K_{R+1}`` and the measured extent
      rises: over the dense sweep this holds in 1485 of 1485 consecutive-R pairs for CPG,
      mCPG, ADG and the orthant, with no exceptions. NMF is refitted from scratch at each
      ``R`` and its atoms are optimized rather than accumulated, so its cones are unrelated
      across ``R`` and its extent falls in 82 of 294 pairs (worst drop 12%). That is not a
      failure of this metric but the drawback [BEE20] §5 raises against NMF, showing up
      geometrically.
    * **Positive on a thin cone**, where every ambient volume is zero.
    * **Same units at every R**, so the values are a single quantity across the sweep and
      the curve can be read horizontally as well as vertically.

    Reported as a fraction of the section of ``K_full = span_+{all training snapshots}``,
    which makes it dimensionless and gives it a meaningful ceiling: **1 means the cone is
    as wide, on average over directions, as everything the training snapshots generate.**

    Values **above 1 are possible and are a finding, not an error.** They mean the cone is
    wider than ``K_full`` in the mean -- which requires leaving it. mCPG does exactly that
    (see ``excess`` and ``reach_outside``), and so does the orthant baseline, whose
    canonical axes are not non-negative combinations of the snapshots.
    """
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {}
    m = G.shape[0]

    P = _section_vertices(G, m)
    if P is None:
        # The hyperplane does not cut this cone transversally -- POD's mixed-sign modes
        # are the case in practice. An unbounded section has no width to report.
        return {"section_extent": float("nan"), "section_width": float("nan")}
    F = _section_vertices(dataset.train(), m, drop_nontransversal=True)
    if F is None:
        return {}

    Z = _gaussian_directions(m, n_samples, seed)
    w = _mean_width(P, Z)
    w_full = _mean_width(F, Z)
    return {
        "section_extent": float(w / w_full) if w_full > 0 else float("nan"),
        "section_width": float(w),
    }


def _section_vertices(G: np.ndarray, m: int, *, drop_nontransversal: bool = False):
    """Vertices ``h g_i / <g_i, u>`` of the section, or ``None`` if it is unbounded.

    Scaling any ``g_i`` leaves its vertex unchanged, so the measure cannot respond to the
    methods disagreeing about normalization -- ADG normalizes, CPG selects raw snapshots,
    NMF's atoms carry arbitrary scale.
    """
    G = np.asarray(G, float)
    inner = G.sum(axis=0) / np.sqrt(m)            # <g_i, u> for u = ones / sqrt(m)
    if drop_nontransversal:
        keep = inner > 0
        if not keep.any():
            return None
        G, inner = G[:, keep], inner[keep]
    elif not np.all(inner > 0):
        return None
    return SECTION_HEIGHT * G / inner


@lru_cache(maxsize=8)
def _gaussian_directions(m: int, n_samples: int, seed: int) -> np.ndarray:
    """Fixed directions for the mean-width estimate, shared across every cell.

    Cached on purpose. Drawing fresh directions per cell would make the estimate
    non-monotone in R at the noise level, which is precisely the artefact this metric
    exists to avoid, and would redraw millions of normals per dataset for nothing.
    """
    return np.random.default_rng(seed).standard_normal((m, n_samples))


def _mean_width(P: np.ndarray, Z: np.ndarray) -> float:
    """``E_g[max_i <p_i, g> - min_i <p_i, g>]`` over the supplied directions."""
    proj = P.T @ Z                                 # (n_vertices, n_samples)
    return float((proj.max(axis=0) - proj.min(axis=0)).mean())


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
    row.update(section_extent(dataset, result.generators))
    row.update(reach_outside(dataset, result.generators))
    # Two-sided discrepancy: zero exactly when K_R and K_full coincide. Reporting only one
    # direction lets a cone look perfect while being much too small, or much too large.
    #
    # AVERAGED, not maximized, and that is the number to read. A max reports only whichever
    # direction happens to be larger, so it silently changes which quantity it is showing
    # from one method to the next: over the dense sweep it returns `cover` in 88% of CPG
    # and ADG cells and `excess` in 76-95% of mCPG, NMF and orthant cells. A panel that is
    # the missed-mass for some curves and the excess-mass for others cannot be compared
    # across methods, which is the only thing that panel is for. The max also discards the
    # smaller term entirely, so a cone that is both somewhat too small AND hugely too large
    # scores identically to one that is only hugely too large.
    #
    # The average keeps both faults in view and is still a metric on cones: it is
    # symmetric, it vanishes only when both directions do, and it inherits the triangle
    # inequality from the directed distances it averages.
    #
    # It is built from the MEAN residuals, so this panel is exactly the average of the two
    # panels beside it. The old max mixed statistics as well as directions -- it combined
    # the max-over-samples residuals while those panels plot the mean-over-samples ones,
    # so it could not be read against them at all.
    if "cover_mean_err" in row and "excess_mean_err" in row:
        row["cone_sym_err"] = 0.5 * (row["cover_mean_err"] + row["excess_mean_err"])
    # The textbook Hausdorff distance, kept in the CSV. sup-based and therefore the strict
    # "0 iff the cones coincide" statement, but for the reasons above it is not plotted.
    if "cover_max_err" in row and "excess_max_err" in row:
        row["cone_hausdorff"] = max(row["cover_max_err"], row["excess_max_err"])
    return row
