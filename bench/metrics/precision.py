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


def uses_cone_projection(result: BasisResult) -> bool:
    """Is this method scored by cone projection, or by unconstrained least squares?

    Every cone method and NMF (whose atoms are non-negative, so ``span_+`` is the space
    it actually offers) are scored with NNLS. POD is not: its coefficients carry no sign
    constraint, and measuring it through a cone projection would score it on a
    reconstruction it is not allowed to use. Shared so the error columns and the
    reconstruction figures cannot drift apart.
    """
    return result.family != "baseline" or result.method.startswith("nmf")


def reconstruct(snapshots: np.ndarray, generators: np.ndarray,
                cone: bool = True) -> np.ndarray:
    """The approximation each error column is measured against, as vectors.

    Same two conventions as ``projection_errors`` / ``reconstruction_errors``, so a
    reconstruction figure always shows exactly what the reported number scored.
    """
    S = np.asarray(snapshots, float)
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] == 0:
        return np.zeros_like(S)
    if not cone:
        coeffs, *_ = np.linalg.lstsq(G, S, rcond=None)
        return G @ coeffs
    out = np.empty_like(S)
    for q in range(S.shape[1]):
        out[:, q], _ = project_onto_cone(S[:, q], G, mass=None)
    return out


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


def per_snapshot_rel_errors(errors: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``||theta_q - Pi_K(theta_q)|| / ||theta_q||`` -- each snapshot against *itself*.

    The companion to the shared-denominator convention, not a replacement for it. The two
    answer different questions and the gap between them can be large:

    * **Shared** ``err_q / max_p ||theta_p||`` -- one global yardstick. Comparable across
      methods and readable against the tolerance that produced it, which is why it is the
      benchmark's primary column. But a large snapshot sets the yardstick, so a small one
      can be almost entirely unrepresented and still score well.
    * **Per-snapshot** ``err_q / ||theta_q||`` -- each snapshot judged on its own scale.
      Nothing hides behind anything.

    They diverge exactly where snapshot magnitudes spread. On ``physics`` (norms spanning
    604x) CPG at R=8 reads 0.0117 shared and **0.8137** per-snapshot: one snapshot is 81%
    unrepresented in its own terms while the headline says 1%.

    This is also the convention **ADG optimizes**. Normalizing to ``S_norm`` makes every
    snapshot unit-norm, so its ``epsilon`` bounds ``sin theta_K(x_hat) = e_K(x_hat)``
    per snapshot; CPG and mCPG bound the shared quantity. Reporting only the shared column
    grades ADG on an objective it is not pursuing -- on ``physics`` that inverts the
    ranking, from ADG 3.4x worse to ADG 9.4x better.

    Snapshots of zero norm are excluded rather than counted as 0/0. ``Dataset`` drops them
    at construction, so this should never fire, but the ratio must not invent a value.
    """
    err = np.asarray(errors, float)
    norms = np.linalg.norm(np.asarray(columns, float), axis=0)
    good = norms > 0.0
    if not good.any():
        return np.array([])
    return err[good] / norms[good]


def evaluate(dataset: Dataset, result: BasisResult) -> dict[str, float]:
    """Precision row for one (dataset, method) cell.

    Two normalizations are reported side by side, because they answer different questions
    and no single one is right for every method (see ``per_snapshot_rel_errors``):

    * ``*_rel_err`` divides by ``max_q ||theta_q||`` -- the same normalizer both relative
      tolerances use, so it can be read directly against the ``delta`` that produced it.
    * ``*_rel_err_persnap`` divides each snapshot by its own norm -- the quantity ADG's
      tolerance actually bounds.
    """
    scale = dataset.scale
    use_cone = uses_cone_projection(result)
    err_fn = projection_errors if use_cone else reconstruction_errors

    train_cols = dataset.train()
    train_err = err_fn(train_cols, result.generators)
    train_ps = per_snapshot_rel_errors(train_err, train_cols)
    row: dict[str, float] = {
        "R": float(result.R),
        "train_max_rel_err": float(train_err.max() / scale) if train_err.size else 0.0,
        "train_mean_rel_err": float(train_err.mean() / scale) if train_err.size else 0.0,
        "train_max_rel_err_persnap": float(train_ps.max()) if train_ps.size else 0.0,
        "train_mean_rel_err_persnap": float(train_ps.mean()) if train_ps.size else 0.0,
    }

    test = dataset.test()
    if test is not None:
        test_err = err_fn(test, result.generators)
        test_ps = per_snapshot_rel_errors(test_err, test)
        row["test_max_rel_err"] = float(test_err.max() / scale)
        row["test_mean_rel_err"] = float(test_err.mean() / scale)
        row["test_max_rel_err_persnap"] = float(test_ps.max()) if test_ps.size else 0.0
        row["test_mean_rel_err_persnap"] = float(test_ps.mean()) if test_ps.size else 0.0
    else:
        # Absent, not zero: ``physics`` has no split by design, and a 0.0 here would
        # read as a perfect generalization result.
        for key in ("test_max_rel_err", "test_mean_rel_err",
                    "test_max_rel_err_persnap", "test_mean_rel_err_persnap"):
            row[key] = float("nan")

    row.update(
        {f"nn_{k}": v
         for k, v in nonnegativity_violation(
             dataset.train(), result.generators, cone=use_cone).items()}
    )
    return row
