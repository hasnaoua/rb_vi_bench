"""The one synthetic dataset most tests run against.

Non-negative bumps whose support moves with the parameter -- structurally what a contact
multiplier looks like, at a size where a full method sweep costs milliseconds. Almost
every test uses it, so the suite stays fast enough to run on every edit.

It lives in its own module rather than in ``conftest.py`` because two things need it in
two different ways: ``conftest`` wraps it as the session ``bumps`` fixture, and a handful
of tests need to build a *variant* (a different dimension, a different seed) which a
fixture cannot express.
"""

from __future__ import annotations

import numpy as np

from bench.types import Dataset


def make_bumps(dim: int = 30, n: int = 18, seed: int = 0) -> Dataset:
    """A small, cheap, structurally realistic non-negative dataset."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, dim)
    cols = []
    for _ in range(n):
        c, w, a = rng.uniform(0.25, 0.75), rng.uniform(0.08, 0.25), rng.uniform(0.5, 2.0)
        cols.append(a * np.clip(1.0 - ((x - c) / w) ** 2, 0.0, None))
    S = np.column_stack(cols)
    idx = np.arange(n)
    test_idx = idx[::4]
    return Dataset(name="bumps", snapshots=S,
                   train_idx=np.setdiff1d(idx, test_idx), test_idx=test_idx)


def basis_of(dataset, generators):
    """A ``BasisResult`` wrapping explicit generators, for metric-level tests.

    The metrics take a fitted result, but several of them have to be pinned against a
    cone whose generators are *chosen*, not fitted -- the full training cone, a strict
    sub-cone, a deliberately scaled one. This wraps bare columns in the shape the
    metrics expect without going through an adapter.
    """
    from bench.types import BasisResult

    return BasisResult(
        method="explicit", family="test", paper_tag="", generators=np.asarray(generators),
        R=int(np.asarray(generators).shape[1]), selected_indices=[], errors=[],
        fit_seconds=0.0, solver_calls={}, normalized_generators=False, notes="",
    )


__all__ = ["basis_of", "make_bumps"]
