"""Stability: is the reduced pair usable, and is the cone well conditioned?

Three groups, in increasing order of how much the dataset must supply.

**1. Conditioning of the cone itself** (needs only the generators). Gram condition
number and ``e_orth``. [NDEE22] §4 is explicit that this is what mCPG exists to
improve: one wants "a basis such that the corresponding Gram matrix is as
well-conditioned as possible", and Gram-Schmidt is unavailable because it "would lead
to a departure from the positive cone ``W^+``". Its claims C5/C6 are exactly these two
numbers, so this group is a direct reproduction test.

**2. Determinism** (needs a re-run). CPG and ADG are deterministic given the training
set; NMF is not, and [BEE20] §5 lists that as a drawback. Measured, not assumed.

**3. Inf-sup constants** (needs ``A``, ``B(mu)`` and primal snapshots). ``beta_HF``
(Eq. 3), ``beta^dec`` (Eq. 14), ``beta^on`` (Eq. 18), ``beta^off`` (Eq. 30), plus
``sigma_S`` (Eq. 21), ``c_S`` (Eq. 22) and the Proposition 3.1 check. Only two datasets
here can support these, and the runner skips the rest rather than substituting an
identity for a constraint operator it does not have.

**Two caveats that are properties of the underlying implementation, not of this
harness, and that must travel with any number reported from here.**

``inf_sup`` returns an **upper bound** on ``beta``: minimizing a quotient over a cone
is non-convex, so the multi-start projected gradient can only overestimate. It is
therefore reliable for showing ``beta`` is *small* and merely suggestive when showing
``beta`` is *large*. What is exact is the separate ``beta = 0`` test (a convex QP), and
the Gordan certificate below is exact in the other direction -- ``t > 0`` *proves*
``beta^dec > 0``. So the honest reading is: trust ``gordan_t`` on the sign, treat the
``beta`` values as upper bounds.

``c_S`` is computed over the cone's **span** rather than the cone, which overestimates
it. Since ``c_S`` enters Proposition 3.1 only through the criterion Eq. (23) and the
bound Eq. (24), overestimating makes both *conservative* -- a verified Prop. 3.1 stays
verified.
"""

from __future__ import annotations

import numpy as np

from .. import _paths  # noqa: F401
from ..types import BasisResult, Dataset

from rb_vi_common.cone_greedy import e_orth
from rb_vi_common.reduction import (
    Whitener,
    boundedness_c_S,
    inf_sup,
    inf_sup_hf,
    orth_union,
    pod,
    sigma_S,
    supremizer_space,
)
from hertz_infsup_probe import gordan_certificate, orthant_meets_kernel


# ---------------------------------------------------------------------------
# 1. Conditioning of the cone
# ---------------------------------------------------------------------------

def gram_conditioning(generators: np.ndarray) -> dict[str, float]:
    """Condition number of the generator Gram matrix -- [NDEE22] claim C6.

    Computed on **column-normalized** generators. Without that, the number would be
    dominated by the scale spread of the snapshots rather than by the cone's aperture,
    and the two families would be incomparable: [BEE20]'s CPG keeps raw generators
    while [NDEE22]'s normalizes them. Normalizing here measures the geometry both
    conventions share.
    """
    G = np.asarray(generators, float)
    # A one-generator cone has a 1x1 Gram matrix whose condition number is exactly 1.
    # That is arithmetically true and comparatively meaningless -- placed next to a
    # 20-generator cone it reads as a perfect score. Datasets that saturate at R = 1
    # are common here (`fem_lambda` and `physics` are near-rank-1: s1/s0 = 0.026 and
    # 0.057), so this must be absent rather than flattering.
    if G.size == 0 or G.shape[1] < 2:
        return {"gram_cond": float("nan"), "gram_cond_raw": float("nan")}

    norms = np.linalg.norm(G, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    Gn = G / safe

    def cond(M):
        A = M.T @ M
        w = np.linalg.eigvalsh(A)
        w = w[w > 0]
        if w.size == 0:
            return float("nan")
        return float(w.max() / w.min())

    return {"gram_cond": cond(Gn), "gram_cond_raw": cond(G)}


def orthogonality_defect(generators: np.ndarray) -> dict[str, float]:
    """``e_orth`` of [NDEE22] Eq. (41) -- claim C5.

    ``e_orth(r) = ||(I - Pi_{K_r}) nu_{r+1}|| <= 1``; §5.1: "the larger ``e_orth(r)``
    (i.e. close to 1), the closer ``nu_{r+1}`` to being orthogonal to the cone". The
    paper reports ``e_orth^CPG <= e_orth^mCPG``, so mCPG should score *higher* here.

    Requires normalized generators for the bound ``<= 1`` to hold, so raw-generator
    methods are normalized first.
    """
    G = np.asarray(generators, float)
    if G.size == 0 or G.shape[1] < 2:
        return {"e_orth_mean": float("nan"), "e_orth_min": float("nan")}
    norms = np.linalg.norm(G, axis=0)
    Gn = G / np.where(norms > 0, norms, 1.0)
    vals = e_orth(Gn)
    if vals.size == 0:
        return {"e_orth_mean": float("nan"), "e_orth_min": float("nan")}
    return {"e_orth_mean": float(vals.mean()), "e_orth_min": float(vals.min())}


# ---------------------------------------------------------------------------
# 2. Determinism
# ---------------------------------------------------------------------------

def determinism(method, dataset, *, delta=None, R=None) -> dict[str, float]:
    """Re-run and compare: identical cone, or not?

    For the deterministic greedies this must return 0.0 -- a non-zero value is a bug
    (an unseeded RNG somewhere). For NMF it is expected to be non-zero, and the
    magnitude is the point: [BEE20] §5 counts non-determinism against NMF, since it
    "must be re-run from scratch for each R, and is non-deterministic when re-run".
    """
    a = method.fit(dataset, delta=delta, R=R)
    b = method.fit(dataset, delta=delta, R=R)
    same_R = a.R == b.R
    same_order = a.selected_indices == b.selected_indices
    if same_R and a.generators.shape == b.generators.shape:
        scale = max(float(np.abs(a.generators).max()), 1e-300)
        drift = float(np.abs(a.generators - b.generators).max() / scale)
    else:
        drift = float("nan")
    return {
        "rerun_same_R": float(same_R),
        "rerun_same_order": float(same_order),
        "rerun_generator_drift": drift,
    }


# ---------------------------------------------------------------------------
# 3. Inf-sup family
# ---------------------------------------------------------------------------

def infsup_report(dataset: Dataset, result: BasisResult, *,
                  delta_pod: float = 1e-2, delta_pga: float = 0.9,
                  n_eval: int = 5) -> dict[str, float]:
    """The [NDEE22] inf-sup constants for one fitted cone.

    Everything is computed in **whitened coordinates**, where the ``V`` and ``W`` inner
    products become Euclidean and the Riesz isomorphism of Eq. (15) folds into the
    change of basis -- so the formulas read as in the paper.

    [UNSPECIFIED] The inner products themselves are a choice neither paper states for
    these test problems: ``G_V = A`` (the energy norm, for which [NDEE22]'s ``beta_HF``
    is the natural constant) and ``G_W = I``. This mirrors item 8 of [BEE20]'s notes.

    ``beta^dec`` is the unstabilized pair (Eq. 14), ``beta^off`` the pair enriched with
    the PGA-compressed supremizer space (Eq. 30). Evaluated on the first ``n_eval``
    test parameters, or training ones when there is no split, and reported as
    min/median so a single favourable ``mu`` cannot carry the row.
    """
    out: dict[str, float] = {}
    if not dataset.supports_infsup or dataset.primal_snapshots is None:
        return out
    G = np.asarray(result.generators, float)
    if G.shape[1] == 0:
        return out

    wh = Whitener(dataset.A, None)                     # G_V = A (energy), G_W = I
    primal_tilde = np.column_stack([wh.v_to_tilde(dataset.primal_snapshots[:, i])
                                    for i in range(dataset.primal_snapshots.shape[1])])
    cone_tilde = np.column_stack([wh.w_to_tilde(G[:, r]) for r in range(G.shape[1])])
    # Normalize: beta is scale-invariant in each generator, but conditioning of the
    # intermediate SVDs is not.
    norms = np.linalg.norm(cone_tilde, axis=0)
    cone_tilde = cone_tilde / np.where(norms > 0, norms, 1.0)

    train_cols = dataset.train_idx if dataset.train_idx is not None else np.arange(dataset.n_snapshots)
    V_N = pod(primal_tilde[:, train_cols], delta_pod)

    eval_idx = (dataset.test_idx if dataset.test_idx is not None and len(dataset.test_idx)
                else train_cols)[:n_eval]

    betas_hf, betas_dec, betas_off, sigmas, c_ss, gordans = [], [], [], [], [], []
    kernel_hits = 0

    for i in eval_idx:
        B_hat = wh.B_hat(dataset.B_of_mu(int(i)))
        betas_hf.append(inf_sup_hf(B_hat))
        betas_dec.append(inf_sup(B_hat, V_N, cone_tilde))

        # Exact sign tests on C = Q^T B_hat^T X, the object beta^dec is built from.
        C = V_N.T @ B_hat.T @ cone_tilde
        gordans.append(gordan_certificate(C))
        meets, _ = orthant_meets_kernel(C)
        kernel_hits += int(meets)

        Z = supremizer_space(B_hat, cone_tilde)
        if Z.shape[1]:
            s, _ = sigma_S(Z, V_N)
            sigmas.append(s)
            V_off = orth_union(V_N, Z)
            betas_off.append(inf_sup(B_hat, V_off, cone_tilde))
            c_ss.append(boundedness_c_S(B_hat, V_off, cone_tilde))

    def stat(vals, name):
        arr = np.asarray([v for v in vals if np.isfinite(v)], float)
        if arr.size == 0:
            return
        out[f"{name}_min"] = float(arr.min())
        out[f"{name}_median"] = float(np.median(arr))

    stat(betas_hf, "beta_hf")
    stat(betas_dec, "beta_dec")
    stat(betas_off, "beta_off")
    stat(sigmas, "sigma_S")
    stat(c_ss, "c_S")
    stat(gordans, "gordan_t")

    out["dim_V_N"] = float(V_N.shape[1])
    out["n_infsup_mu"] = float(len(eval_idx))
    # The C1 question: does the kernel of C meet the non-negative orthant? A positive
    # Gordan t proves it does not. Both repositories report this never happening on
    # their test problems; recording it per cell is how that survives new datasets.
    out["kernel_meets_orthant"] = float(kernel_hits)
    # The monotonicity [NDEE22] reports as `beta^dec <= beta^off`: enriching the primal
    # space cannot lower the inf-sup constant. This is NOT Eq. (18) -- that compares
    # `beta^on` against `beta_HF`, and `beta^on` needs the *online* enrichment
    # `S_R(mu)` rebuilt per mu, which this offline-only report does not construct.
    if "beta_dec_min" in out and "beta_off_min" in out:
        out["ordering_holds"] = float(out["beta_off_min"] >= out["beta_dec_min"] - 1e-9)
    return out


def evaluate(dataset: Dataset, result: BasisResult, *,
             with_infsup: bool = True) -> dict[str, float]:
    """Stability row for one (dataset, method) cell."""
    row: dict[str, float] = {}
    row.update(gram_conditioning(result.generators))
    row.update(orthogonality_defect(result.generators))
    if with_infsup:
        row.update(infsup_report(dataset, result))
    return row
