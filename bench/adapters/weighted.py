"""Norm-weighted residual greedy: one exponent interpolating CPG and ADG's selection.

Selection rule, at step ``r`` over the not-yet-selected training snapshots::

    q_r  in  argmax_q  || theta_q - Pi_{K_{r-1}}(theta_q) ||  /  || theta_q ||^gamma

with ``gamma in [0, 1]``. Generators are the selected snapshots, unchanged, so
``K_R subset K_full`` by construction and the cones are hierarchical -- the same two
properties CPG has, and for the same reason.

**The endpoints are exact, and were verified rather than assumed.** At matched ``R`` on
all three datasets:

* ``gamma = 0`` reproduces **CPG** ([BEE20] Algorithm 2) -- identical selections in
  identical order. The weight is 1, so the rule is CPG's absolute-residual argmax.
* ``gamma = 1`` reproduces **ADG's selection rule**, matching ``adg_k0``'s selected set.
  The weighted residual is then ``|| theta - Pi(theta) || / || theta ||``, which for a
  unit-normalized snapshot is ``sin`` of the angular defect ADG maximizes.

The second statement is about the *selection rule*, not about stock ``adg``: that also
seeds from the largest-mutual-angle pair and admits every candidate tied at
``theta_max`` as a batch. ``adg_k0`` is the variant with the pair seeding removed, which
is why it is the one this family lands on.

**The first step needs [BEE20] Eq. (56), and that is not a patch.** At ``K_0 = {0}`` the
projection is zero, so the residual *is* the snapshot and the score collapses to
``|| theta_q ||^(1 - gamma)``:

* for ``gamma < 1`` its argmax is ``argmax_q || theta_q ||``, which is exactly Eq. (56);
* at ``gamma = 1`` it is ``|| theta_q || / || theta_q || = 1`` for *every* candidate -- a
  mathematical tie across the whole training set.

Measured on 3D Pellet-Cladding, that tie resolves purely on floating-point round-off:
the 47 first-step scores take five distinct values spanning 0.9999999999999997 to
1.0000000000000002, and the argmax lands on snapshot 18 for no reason a reader could
defend. Taking Eq. (56) at the first step is therefore what the family already does at
every ``gamma < 1``, and the continuous limit from below at ``gamma = 1``; it makes the
family continuous in ``gamma`` rather than discontinuous at one endpoint.

**Cost is CPG's in order, and lower in constant factor -- which the offline-cost column
cannot express.** The rule needs one cone projection per unselected candidate per step,
so both are ``O(nR)``. This implementation issues exactly ``sum_{r=1}^{R-1} (n - r)``:
it skips candidates already selected (their residual is zero, so they cannot win the
argmax) and takes the first generator from Eq. (56) in closed form, with no projection.
The vendored CPG issues ``(2R - 1) * n``, which is 2.5x to 3.3x more on ``fem_lambda``.

That gap is **implementation, not rule**. ``adg_g0`` and ``cpg_bee20`` select identically
at every R, so a reader comparing their ``calls_total`` is comparing two transcriptions of
one algorithm, not two algorithms. Compare cost *within* the gamma family, where the
count is identical at every gamma by construction, and read gamma's real price in the
accuracy and conditioning columns instead.

**Tolerance mode uses the ABSOLUTE convention**, ``delta * max_q || theta_q ||``, the
same one CPG uses -- deliberately, so that ``delta`` means the same thing at every
``gamma`` and a tolerance sweep varies only the selection rule. The alternative, scaling
the stopping rule with ``gamma`` too, would move two things at once and make the
resulting ``R`` uninterpretable.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _paths  # noqa: F401  -- sys.path side effect
from ..instrument import count_solver_calls, summarize
from ..types import BasisResult, Dataset
from ._common import require_delta

from rb_vi_common.cone_projection import project_onto_cone

#: Below this a snapshot is treated as having no direction at all: it cannot be a
#: generator, and it must not be divided by. ``Dataset`` already drops numerically zero
#: columns at construction, so this is a guard rather than a filter.
ZERO_TOL = 1e-14


def _select_weighted(train: np.ndarray, gamma: float, *, max_R: int | None,
                     tol: float | None) -> tuple[list[int], list[float]]:
    """The greedy loop. Returns the selected indices and the residual history."""
    dim, n = train.shape
    norms = np.linalg.norm(train, axis=0)
    usable = norms > ZERO_TOL
    if not usable.any():
        return [], []

    # [BEE20] Eq. (56) -- see the module docstring for why this is the first step at
    # every gamma, not only where the weighted rule degenerates.
    first = int(np.argmax(np.where(usable, norms, -np.inf)))
    selected = [first]
    history: list[float] = []
    cap = n if max_R is None else min(int(max_R), n)

    while len(selected) < cap:
        G = train[:, selected]
        best, best_score, best_resid = None, -np.inf, 0.0
        worst_resid = 0.0
        for q in range(n):
            if not usable[q] or q in selected:
                continue
            proj, _ = project_onto_cone(train[:, q], G, mass=None)
            resid = float(np.linalg.norm(train[:, q] - proj))
            worst_resid = max(worst_resid, resid)
            score = resid / (norms[q] ** gamma) if gamma else resid
            if score > best_score:
                best, best_score, best_resid = q, score, resid
        if best is None:
            break
        history.append(worst_resid)
        # Stop on the ABSOLUTE worst residual, not on the weighted score: delta then
        # means the same thing at every gamma (module docstring).
        if tol is not None and worst_resid <= tol:
            break
        selected.append(best)
        _ = best_resid
    return selected, history


def fit_weighted_greedy(dataset: Dataset, *, delta: float | None = None,
                        R: int | None = None, gamma: float = 0.5) -> BasisResult:
    """Norm-weighted residual greedy at one ``gamma``. See the module docstring."""
    train = dataset.train()
    tol = None if R is not None else require_delta(delta) * dataset.scale

    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        selected, history = _select_weighted(train, float(gamma), max_R=R, tol=tol)
    seconds = time.perf_counter() - t0

    generators = (train[:, selected] if selected
                  else np.empty((dataset.dim, 0), dtype=float))
    pct = int(round(float(gamma) * 100))
    return BasisResult(
        method=f"adg_g{pct}",
        family="weighted",
        paper_tag="",
        generators=np.ascontiguousarray(generators, dtype=float),
        R=int(generators.shape[1]),
        selected_indices=[int(i) for i in selected],
        errors=[float(e) / dataset.scale for e in history],
        fit_seconds=seconds,
        solver_calls=summarize(counts),
        # Generators are the selected snapshots, unchanged -- as in CPG. The weighting
        # acts on the SELECTION, never on what is stored, so span_+ is untouched.
        normalized_generators=False,
        notes=(f"norm-weighted residual greedy, gamma={gamma:g}; "
               "first generator by [BEE20] Eq. (56)"),
    )


def _at(gamma: float):
    """A zero-argument-gamma fitter, for the registry."""
    def fit(dataset, *, delta=None, R=None):
        return fit_weighted_greedy(dataset, delta=delta, R=R, gamma=gamma)
    fit.__name__ = f"fit_weighted_g{int(round(gamma * 100))}"
    fit.__doc__ = f"Norm-weighted residual greedy at gamma={gamma:g}."
    return fit


#: The sweep the trade-off is read from: gamma from CPG's rule to ADG's, in even steps.
GAMMAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

FITTERS = {f"adg_g{int(round(g * 100))}": _at(g) for g in GAMMAS}
