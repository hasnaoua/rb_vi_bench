"""Exact enumeration of every cone the gamma family can produce on [0, 1].

The gamma family (``adapters.weighted``) selects

    q_r in argmax_q  e(theta_q) / ||theta_q||^gamma,        gamma in [0, 1]

with ``e`` the residual against the cone held so far. Sweeping gamma on a five-point
grid answers "what do these five members do"; it cannot answer "how many members are
there", and the report's Caveat 8 is exactly that admission. This module answers the
second question exactly, which retires the caveat rather than narrowing it.

**Why it is finite.** Taking logs,

    log s_gamma(q) = log e(theta_q) - gamma * log||theta_q||

is affine in gamma. The winner at a given gamma is therefore the upper envelope of ``n``
lines with slope ``-log||theta_q||`` and intercept ``log e(theta_q)`` -- equivalently the
upper convex hull of the points ``P_q = (-log||theta_q||, log e(theta_q))``. An envelope
of n lines has at most n-1 breakpoints, so one greedy step partitions [0, 1] into at most
n pieces, and only the candidates on the hull can ever win. Adjacent winners q, q' swap at

    gamma* = log( e(theta_q) / e(theta_q') ) / log( ||theta_q|| / ||theta_q'|| )

**Why it is free.** ``e`` and ``||.||`` at step r are exactly the quantities the greedy
has already computed to make its own choice. The envelope is arithmetic on those numbers:
no cone projection, hence no NNLS call, is issued to find the breakpoints.
``test_gamma_intervals`` pins this by comparing solver counts against the fixed-gamma
fitters.

**What the driver does.** Start with the single interval [0, 1] holding the Eq. (56)
first generator -- which is the first generator at *every* gamma, so the root is shared.
At each step, every live interval is split at its own breakpoints, and each piece carries
its own generator sequence forward. Pieces whose sequences coincide are merged again, so
the output is the minimal set of distinct trajectories, each tagged with the half-open
gamma-interval that produces it.

Intervals are half-open ``[lo, hi)`` with the last one closed at 1.0, so every gamma in
[0, 1] belongs to exactly one trajectory and ``trajectory_at(0.25)`` is unambiguous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _paths  # noqa: F401  -- sys.path side effect
from .adapters.weighted import ZERO_TOL
from .types import Dataset

from rb_vi_common.cone_projection import project_onto_cone

#: Two breakpoints closer than this are treated as one. The envelope is computed in log
#: space, so a crossing is only meaningful to about sqrt(eps) in gamma; splitting on
#: anything finer manufactures intervals that differ by round-off rather than by choice.
GAMMA_TOL = 1e-9


@dataclass
class Trajectory:
    """One distinct generator sequence, and the gamma-interval that produces it."""

    lo: float
    hi: float
    selected: list[int] = field(default_factory=list)
    #: Breakpoints interior to [0, 1] at which THIS trajectory was split off, by step.
    split_history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def contains(self, gamma: float) -> bool:
        """Half-open [lo, hi), closed at the right end of [0, 1]."""
        if self.hi >= 1.0 - GAMMA_TOL:
            return self.lo - GAMMA_TOL <= gamma <= self.hi + GAMMA_TOL
        return self.lo - GAMMA_TOL <= gamma < self.hi - GAMMA_TOL


def upper_envelope(resid: np.ndarray, norms: np.ndarray, live: np.ndarray,
                   lo: float = 0.0, hi: float = 1.0) -> list[tuple[float, float, int]]:
    """Partition ``[lo, hi]`` into ``(start, end, winner)`` pieces. No solver calls.

    ``winner`` is the index that ``adapters.weighted._select_weighted`` would choose for
    every gamma strictly inside the piece. Ties are broken toward the LOWEST index,
    because that loop tests ``score > best_score`` and so keeps the first of any tie --
    a different rule here would make the enumeration disagree with the fitter it exists
    to enumerate.

    Candidates with zero residual are dropped: their score is 0 at every gamma, so they
    can only win when every live candidate is exhausted, and that case ends the greedy
    rather than choosing among them.
    """
    idx = np.flatnonzero(live & (resid > 0.0) & (norms > ZERO_TOL))
    if idx.size == 0:
        return []
    # y_q(gamma) = a_q + gamma * b_q
    a = np.log(resid[idx])
    b = -np.log(norms[idx])

    pieces: list[tuple[float, float, int]] = []
    gamma = float(lo)
    guard = 0
    while gamma < hi - GAMMA_TOL:
        guard += 1
        if guard > idx.size + 2:                 # at most n-1 breakpoints; see docstring
            break
        y = a + gamma * b
        best = float(y.max())
        # Every piece boundary IS a crossing, so two lines are tied at ``gamma`` by
        # construction. The piece describes the OPEN interval to the right, and there
        # the larger slope wins -- breaking the tie by lowest index instead would
        # attribute each piece to the line that just lost it. Genuine duplicates (equal
        # residual AND equal norm, hence equal slope) fall through to lowest index,
        # which is what ``_select_weighted``'s ``score > best_score`` keeps.
        tied = np.flatnonzero(y >= best - 1e-12 * max(1.0, abs(best)))
        top_slope = float(b[tied].max())
        tied = tied[b[tied] >= top_slope - 1e-12]
        w_local = int(tied[np.argmin(idx[tied])])
        winner = int(idx[w_local])

        # Next crossing: the smallest gamma' > gamma at which some line overtakes.
        nxt = hi
        for j in range(idx.size):
            if b[j] <= b[w_local] + 1e-18:       # never overtakes going right
                continue
            cross = (a[w_local] - a[j]) / (b[j] - b[w_local])
            if gamma + GAMMA_TOL < cross < nxt:
                nxt = float(cross)
        pieces.append((gamma, min(nxt, hi), winner))
        if nxt <= gamma + GAMMA_TOL:
            break
        gamma = nxt
    if not pieces:                                # hi <= lo + tol: a degenerate interval
        y = a + lo * b
        tied = np.flatnonzero(y >= float(y.max()) - 1e-15)
        pieces = [(lo, hi, int(idx[int(tied[np.argmin(idx[tied])])]))]
    return pieces


def _residuals(train: np.ndarray, selected: list[int], norms: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Residual of every candidate against ``span_+{selected}``. One NNLS per candidate.

    This is the greedy's own work, issued once per interval per step -- exactly what a
    fixed-gamma run of the same length would issue. The envelope on top of it is free.
    """
    n = train.shape[1]
    resid = np.zeros(n)
    live = np.ones(n, bool)
    G = train[:, selected]
    for q in range(n):
        if q in selected or norms[q] <= ZERO_TOL:
            live[q] = False
            continue
        proj, _ = project_onto_cone(train[:, q], G, mass=None)
        resid[q] = float(np.linalg.norm(train[:, q] - proj))
    return resid, live


def enumerate_trajectories(dataset: Dataset, R: int, *,
                           max_intervals: int = 4096,
                           on_step=None) -> list[Trajectory]:
    """Every distinct generator sequence the family produces on [0, 1], up to ``R``.

    ``max_intervals`` is a guard, not a modelling choice: the split count is bounded by
    the data, but a pathological dataset could in principle split at every step, and a
    silent 2^R blow-up would look like a hang. Hitting it raises rather than truncating,
    because a truncated enumeration is not an enumeration.
    """
    train = dataset.train()
    norms = np.linalg.norm(train, axis=0)
    usable = norms > ZERO_TOL
    if not usable.any():
        return []

    # [BEE20] Eq. (56) at r=1, for every gamma -- so all trajectories share a root.
    first = int(np.argmax(np.where(usable, norms, -np.inf)))
    live_intervals = [Trajectory(0.0, 1.0, [first])]
    cap = min(int(R), int(usable.sum()))

    for step in range(2, cap + 1):
        nxt: list[Trajectory] = []
        for tr in live_intervals:
            resid, live = _residuals(train, tr.selected, norms)
            for (g0, g1, winner) in upper_envelope(resid, norms, live, tr.lo, tr.hi):
                if g1 - g0 <= GAMMA_TOL:
                    continue
                child = Trajectory(g0, g1, tr.selected + [winner],
                                   list(tr.split_history))
                if g0 > tr.lo + GAMMA_TOL:
                    child.split_history.append((step, g0))
                nxt.append(child)
        if not nxt:
            break
        live_intervals = _merge_adjacent(nxt)
        if on_step is not None:
            on_step(step, live_intervals)
        if len(live_intervals) > max_intervals:
            raise RuntimeError(
                f"{dataset.name}: gamma enumeration exceeded {max_intervals} intervals "
                f"at R={step}; a truncated enumeration would not be an enumeration"
            )
    return live_intervals


def _merge_adjacent(trs: list[Trajectory]) -> list[Trajectory]:
    """Fuse neighbouring intervals whose generator sequences are identical.

    Two pieces can be split apart at step r by a breakpoint and then make the same
    choice at every later step; left unmerged they would be reported as two trajectories
    that are in fact one, and the headline "number of distinct cones" would be wrong.
    """
    trs = sorted(trs, key=lambda t: t.lo)
    out: list[Trajectory] = []
    for tr in trs:
        if out and out[-1].selected == tr.selected and abs(out[-1].hi - tr.lo) <= GAMMA_TOL:
            out[-1].hi = tr.hi
        else:
            out.append(tr)
    return out


def trajectory_at(trajectories: list[Trajectory], gamma: float) -> Trajectory | None:
    """The trajectory whose interval contains ``gamma``."""
    for tr in trajectories:
        if tr.contains(gamma):
            return tr
    return None


def breakpoints_of(trajectories: list[Trajectory]) -> list[float]:
    """The interior breakpoints, i.e. the interval boundaries strictly inside (0, 1)."""
    return [tr.lo for tr in trajectories
            if GAMMA_TOL < tr.lo < 1.0 - GAMMA_TOL]
