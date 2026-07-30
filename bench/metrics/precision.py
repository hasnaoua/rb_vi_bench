"""Precision: how well does the cone represent the multiplier manifold?

Everything here is a cone-projection error, ``||theta - Pi_K(theta)||``, where
``Pi_K`` is the NNLS projection onto the finitely generated cone. That projection is
**not linear** in ``theta`` -- there is no projection matrix to precompute -- which is
why every error below costs one NNLS solve per snapshot and why these numbers are the
expensive part of the benchmark.

Two comparisons are reported and they answer different questions:

* **At matched tolerance** -- each method is given the same ``delta`` and reaches its
  own ``R``. This is the interface [BEE20] §5 argues *for*: the user supplies an
  accuracy, not a cardinality. The interesting output is ``R``, not the error.
* **At matched cardinality** -- each method is given the same ``R``. This is the only
  comparison NMF can enter at all, and it is the one on which [BEE20]'s own notes
  record NMF beating CPG by 10-20%.

Reporting only one of the two is how a cone method can be made to look arbitrarily
good or bad, so the runner always does both.

**Train vs test.** Where a dataset has a split, generators are built on train only and
errors are reported separately on both. A cone fitted to a snapshot *contains* that
snapshot, so train error at full ``R`` is zero by construction and says nothing about
generalization -- only the test column does.
"""

from __future__ import annotations

import numpy as np

from .. import _paths  # noqa: F401
from ..types import BasisResult, Dataset

from rb_vi_common.cone_projection import project_onto_cone


def projection_errors(snapshots: np.ndarray, generators: np.ndarray,
                      mass: np.ndarray | None = None) -> np.ndarray:
    """``||theta_q - Pi_K(theta_q)||`` for each column of ``snapshots``.

    An empty cone projects to zero, so the error is the snapshot norm itself -- the
    ``n = 1`` case of [BEE20] Eq. (56).
    """
    S = np.asarray(snapshots, float)
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return np.linalg.norm(S, axis=0)
    out = np.empty(S.shape[1])
    for q in range(S.shape[1]):
        proj, _ = project_onto_cone(S[:, q], G, mass=mass)
        out[q] = np.linalg.norm(S[:, q] - proj)
    return out


def reconstruction_errors(snapshots: np.ndarray, generators: np.ndarray) -> np.ndarray:
    """Unconstrained least-squares error, for the POD negative control only.

    POD coefficients carry no sign constraint, so measuring POD with the *cone*
    projection would understate it -- it would be scored on a reconstruction it is not
    allowed to use. This is the honest floor: the best any basis of that cardinality
    can do without the non-negativity requirement.
    """
    S = np.asarray(snapshots, float)
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return np.linalg.norm(S, axis=0)
    coeffs, *_ = np.linalg.lstsq(G, S, rcond=None)
    return np.linalg.norm(S - G @ coeffs, axis=0)


def nonnegativity_violation(snapshots: np.ndarray, generators: np.ndarray,
                            cone: bool = True) -> dict[str, float]:
    """How far the reduced multiplier strays below zero.

    This is the property the whole cone construction exists to preserve ([BEE20] §5:
    POD "is not appropriate to build ``W_R^+``" because its modes have mixed signs).
    For a genuine cone method the violation is exactly zero -- a non-negative
    combination of non-negative generators cannot go negative -- so a non-zero value
    here is a bug, not a result. For the POD control it is *expected* to be non-zero,
    and that non-zero value is the control's entire purpose.
    """
    S = np.asarray(snapshots, float)
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return {"min_entry": 0.0, "max_violation": 0.0, "frac_violating": 0.0}

    approx = np.empty_like(S)
    if cone:
        for q in range(S.shape[1]):
            approx[:, q], _ = project_onto_cone(S[:, q], G, mass=None)
    else:
        coeffs, *_ = np.linalg.lstsq(G, S, rcond=None)
        approx = G @ coeffs

    scale = max(float(np.abs(S).max()), 1e-300)
    return {
        "min_entry": float(approx.min()),
        # Normalized so it is comparable across datasets of different physical scale.
        "max_violation": float(max(0.0, -approx.min()) / scale),
        "frac_violating": float(np.mean(approx.min(axis=0) < -1e-12 * scale)),
    }


def evaluate(dataset: Dataset, result: BasisResult) -> dict[str, float]:
    """Precision row for one (dataset, method) cell.

    Errors are reported relative to ``max_q ||theta_q||`` -- the same normalizer both
    relative tolerances use -- so an error column can be read directly against the
    ``delta`` that produced it.
    """
    scale = dataset.scale
    use_cone = result.family != "baseline" or result.method.startswith("nmf")
    err_fn = projection_errors if use_cone else reconstruction_errors

    train_err = err_fn(dataset.train(), result.generators)
    row: dict[str, float] = {
        "R": float(result.R),
        "train_max_rel_err": float(train_err.max() / scale) if train_err.size else 0.0,
        "train_mean_rel_err": float(train_err.mean() / scale) if train_err.size else 0.0,
    }

    test = dataset.test()
    if test is not None:
        test_err = err_fn(test, result.generators)
        row["test_max_rel_err"] = float(test_err.max() / scale)
        row["test_mean_rel_err"] = float(test_err.mean() / scale)
    else:
        # Absent, not zero: ``physics`` has no split by design, and a 0.0 here would
        # read as a perfect generalization result.
        row["test_max_rel_err"] = float("nan")
        row["test_mean_rel_err"] = float("nan")

    row.update(
        {f"nn_{k}": v
         for k, v in nonnegativity_violation(
             dataset.train(), result.generators, cone=use_cone).items()}
    )
    return row
