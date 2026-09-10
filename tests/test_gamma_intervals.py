"""The exact gamma enumeration: does it agree with the fitter it enumerates?

The module's value rests entirely on three claims -- that the envelope is the same
argmax the greedy makes, that finding it costs no solver calls, and that the interval
containing a given gamma reproduces that gamma's fitted cone exactly. Each is asserted
here against the shipped fitters rather than against a re-derivation, because a
re-derivation would share any error in the module under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import datasets as ds_mod
from bench.adapters import METHODS
from bench.adapters.weighted import fit_weighted_greedy
from bench.instrument import count_solver_calls
from bench.gamma_intervals import (
    GAMMA_TOL, breakpoints_of, enumerate_trajectories, trajectory_at, upper_envelope,
)

GAMMAS = (("adg_g0", 0.0), ("adg_g25", 0.25), ("adg_g50", 0.5),
          ("adg_g75", 0.75), ("adg_g100", 1.0))


def _brute_force_winner(resid, norms, gamma):
    """What ``_select_weighted`` would pick: argmax with ties to the LOWEST index."""
    live = (resid > 0.0) & (norms > 1e-14)
    best, best_score = None, -np.inf
    for q in range(len(resid)):
        if not live[q]:
            continue
        score = resid[q] / (norms[q] ** gamma) if gamma else resid[q]
        if score > best_score:                       # strict: keeps the first of a tie
            best, best_score = q, score
    return best


def test_envelope_matches_brute_force_on_a_dense_gamma_sample():
    """The envelope must name the same winner the fitter's own loop would."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(3, 25))
        resid = rng.random(n) * rng.choice([1.0, 1e-3, 1e3])
        norms = np.exp(rng.normal(0.0, 2.0, n))
        live = np.ones(n, bool)
        pieces = upper_envelope(resid, norms, live)
        assert pieces, trial
        # Cover [0,1] exactly, in order, with no gap.
        assert pieces[0][0] == pytest.approx(0.0, abs=GAMMA_TOL)
        assert pieces[-1][1] == pytest.approx(1.0, abs=GAMMA_TOL)
        for (a0, a1, _), (b0, _b1, _w) in zip(pieces, pieces[1:]):
            assert a1 == pytest.approx(b0, abs=GAMMA_TOL)
        # Interior of every piece agrees with the fitter's rule.
        for (g0, g1, winner) in pieces:
            mid = 0.5 * (g0 + g1)
            assert _brute_force_winner(resid, norms, mid) == winner, (trial, g0, g1)


def test_at_most_n_minus_one_breakpoints_per_step():
    """An envelope of n lines has at most n-1 breakpoints. More would mean a bug."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(3, 30))
        resid = rng.random(n)
        norms = np.exp(rng.normal(0.0, 1.5, n))
        pieces = upper_envelope(resid, norms, np.ones(n, bool))
        assert len(pieces) - 1 <= n - 1


def test_zero_residual_candidates_never_win():
    """Score 0 at every gamma, so they can only 'win' when nothing is left."""
    resid = np.array([0.0, 0.0, 0.5])
    norms = np.array([1.0, 10.0, 2.0])
    pieces = upper_envelope(resid, norms, np.ones(3, bool))
    assert {w for _a, _b, w in pieces} == {2}


def test_enumeration_reproduces_every_shipped_gamma_member():
    """The interval containing gamma must give that member's cone, index for index."""
    for key in ("fem_lambda", "membrane_2d"):
        ds = ds_mod.load(key)
        trs = enumerate_trajectories(ds, 8)
        for method_key, gamma in GAMMAS:
            ref = METHODS[method_key].fit(ds, R=8).selected_indices
            tr = trajectory_at(trs, gamma)
            assert tr is not None, (key, gamma)
            assert list(tr.selected) == list(ref), (key, method_key)


def test_enumeration_adds_no_solver_calls_beyond_the_greedy_itself():
    """The breakpoints are arithmetic on residuals the greedy already computed.

    Asserted on a dataset whose whole interval is one trajectory, so the enumeration
    walks exactly one generator sequence and its solver count is directly comparable
    with a single fixed-gamma fit of the same length. With k trajectories the cost is
    k times that, which is the cost of walking k cones -- still no *extra* solve per
    cone, which is the claim.
    """
    ds = ds_mod.load("fem_lambda")
    R = 8
    with count_solver_calls() as fitted:
        fit_weighted_greedy(ds, R=R, gamma=0.5)
    with count_solver_calls() as enumerated:
        trs = enumerate_trajectories(ds, R)
    assert len(trs) == 1, "fixture assumes a single trajectory; pick another R"
    assert enumerated["nnls"] == fitted["nnls"], (dict(enumerated), dict(fitted))


def test_intervals_tile_the_unit_interval_without_gap_or_overlap():
    ds = ds_mod.load("membrane_2d")
    trs = enumerate_trajectories(ds, 10)
    assert trs[0].lo == pytest.approx(0.0, abs=GAMMA_TOL)
    assert trs[-1].hi == pytest.approx(1.0, abs=GAMMA_TOL)
    for a, b in zip(trs, trs[1:]):
        assert a.hi == pytest.approx(b.lo, abs=GAMMA_TOL)
    # Every gamma lands in exactly one trajectory.
    for g in np.linspace(0.0, 1.0, 51):
        hits = [t for t in trs if t.contains(float(g))]
        assert len(hits) == 1, (g, len(hits))


def test_trajectories_are_hierarchical():
    """A trajectory truncated at R is the first R of the longer one -- as for the fitter.

    The whole matched-cardinality protocol rests on this; an enumeration that broke it
    would describe cones the benchmark could not read.
    """
    ds = ds_mod.load("fem_lambda")
    short = enumerate_trajectories(ds, 6)
    long_ = enumerate_trajectories(ds, 12)
    for s in short:
        mid = s.midpoint
        l = trajectory_at(long_, mid)
        assert l is not None, mid
        assert list(l.selected[:len(s.selected)]) == list(s.selected), mid


def test_adjacent_identical_trajectories_are_merged():
    """Two pieces split apart then making identical later choices are one trajectory."""
    ds = ds_mod.load("fem_lambda")
    trs = enumerate_trajectories(ds, 10)
    for a, b in zip(trs, trs[1:]):
        assert a.selected != b.selected, "adjacent duplicates should have been merged"


def test_breakpoints_are_strictly_interior():
    ds = ds_mod.load("membrane_2d")
    bps = breakpoints_of(enumerate_trajectories(ds, 10))
    assert all(0.0 < b < 1.0 for b in bps)
    assert bps == sorted(bps)
