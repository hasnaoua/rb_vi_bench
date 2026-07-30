"""End-to-end driver: HF snapshots -> CPG cone -> reduced solve, vs NMF,
with and without the [NDEE22] inf-sup stabilization.

Tag key: [BEE20] = Benaceur/Ern/Ehrlacher (this repository's paper, hal
preprint **v2** numbering); [NDEE22] = Niakh/Drouet/Ehrlacher/Ern, the sequel
implemented in ~/stable_model_reduction_vi. Both papers number an "Algorithm 1"
and an "Algorithm 2" differently, so tags are mandatory here. See
~/rb_vi_shared/README.md.

Runs on CPU in a few seconds. See toy_problem.py for what this does and does
not reproduce from [BEE20] §6.
"""

from __future__ import annotations

import numpy as np

import _shared_path  # noqa: F401  -- puts rb_vi_common on sys.path
from nmf_baseline import nmf
from rb_online import (infsup_report, online_enrichment, pga_enrichment,
                       pod_basis, solve_reduced)
from rb_vi_common import cone_projected_greedy, mcpg, norm_Lambda, project_onto_cone
from toy_problem import generate_snapshots, obstacle_gap, solve_hf


def cone_error(S, generators):
    """max_mu ||lambda(mu) - Pi_{K^+}(lambda(mu))||, the LHS of [BEE20] Eq. (58)."""
    errs = []
    for j in range(S.shape[1]):
        proj, _ = project_onto_cone(S[:, j], generators)
        errs.append(norm_Lambda(S[:, j] - proj))
    return max(errs)


def main():
    N, n_train = 60, 40
    S_pri, S_du, params, A, F = generate_snapshots(N=N, n_train=n_train)

    print(f"HF snapshots: primal {S_pri.shape}, dual {S_du.shape}")
    print(f"dual snapshots min entry: {S_du.min():.3e}  (must be >= 0)")
    active = (S_du > 1e-8).sum(axis=0)
    print(f"active contact nodes per parameter: {active.min()}-{active.max()}")

    # The tolerance eps_du of [BEE20] Eq. (58) is ABSOLUTE, so it must be read
    # against the scale of the snapshots. [BEE20] §5 notes the alternative: "One
    # can also consider a relative error criterion instead of an absolute one by
    # dividing the left-hand side of (58) by ||lambda(mu)||_Lambda." That
    # relative form is what [NDEE22] Eq. (13) uses; ``rb_vi_common.cpg``
    # implements it, and the two agree at matched tolerance (see
    # ~/rb_vi_shared/tests/test_equivalence.py).
    scale = max(norm_Lambda(S_du[:, j]) for j in range(S_du.shape[1]))
    print(f"max ||lambda(mu)||: {scale:.3f}  "
          f"(eps_du below is quoted as a fraction of this)")

    # --- Algorithm 2: how R responds to the tolerance -------------------
    print("\nCPG ([BEE20] Algorithm 2) -- tolerance sweep")
    print("  eps_du/scale     eps_du        R")
    for frac in (0.5, 0.3, 0.2, 0.1, 0.05):
        r = cone_projected_greedy(S_du, eps_du=frac * scale)
        print(f"  {frac:>10.2f}   {frac*scale:>9.3f}   {r.R:>4d}")

    # Fix a cardinality where the comparison is informative. At R = n_train the
    # cone contains every snapshot and CPG is exact by construction, which says
    # nothing about either method.
    R_target = 10
    res = cone_projected_greedy(S_du, eps_du=1e-12, max_R=R_target)
    print(f"\nCPG at fixed R = {res.R}")
    print(f"  selected mu indices: {res.selected_indices}")
    print(f"  residual r_n by iteration: "
          f"{', '.join(f'{r:.3e}' for r in res.residuals)}")

    # Hierarchy check: [BEE20] §5 calls these "nested dual RB spaces", so
    # truncating the generator list must reproduce the intermediate cones and
    # the residual must be non-increasing.
    nested_ok = all(
        res.residuals[k] <= res.residuals[k - 1] + 1e-12
        for k in range(1, len(res.residuals))
    )
    print(f"  residuals non-increasing (hierarchical): {nested_ok}")

    # --- NMF baseline at the SAME cardinality ---------------------------
    e_cpg = cone_error(S_du, res.generators)
    print(f"\nNMF baseline ([BEE20] §6.4) at matched R = {res.R}")
    print(f"  max cone-projection error, CPG: {e_cpg:.4e}")
    for seed in (0, 1, 2):
        W = nmf(S_du, R=res.R, seed=seed)
        print(f"  max cone-projection error, NMF (seed={seed}): "
              f"{cone_error(S_du, W):.4e}")

    # --- Reduced solve at unseen parameters -----------------------------
    # Two primal spaces:
    #   (a) V_N alone            -- [BEE20] Algorithm 1 line 4 as written, with
    #                               NO inf-sup stabilization;
    #   (b) V_N + S_R^red        -- [NDEE22] Eq. (29), S_R^red from PGA
    #                               ([NDEE22] Algorithm 1).
    # (a) is what this repository shipped before; (b) is the sequel's fix.
    Theta = pod_basis(S_pri, n_modes=8)
    cone = res.generators

    S_on = online_enrichment(A, cone)                       # [NDEE22] Eq. (17)
    print("\n" + "=" * 66)
    print("[NDEE22] stabilization of the [BEE20] reduced solve")
    print("=" * 66)
    print(f"  dim V_N = {Theta.shape[1]},  R = {cone.shape[1]},  "
          f"dim S_R = {S_on.shape[1]}   ([NDEE22] Eq. 17/20)")
    print("  PGA compression of S_R  ([NDEE22] Alg. 1, Eq. 27-28):")
    enrichments = {"none (as [BEE20])": None}
    for d in (0.9, 0.7, 0.5):
        S_red, info = pga_enrichment(A, cone, delta_pga=d, primal_basis=Theta)
        print(f"    delta_PGA = {d:.1f}  ->  dim S_R^red = {info['dim_S_red']:3d}"
              f"   (from dim S_R = {info['dim_S_R']}, {info['n_iter']} iterations)")
        enrichments[f"PGA delta={d:.1f}"] = S_red
    enrichments["full S_R ([NDEE22] Eq. 17)"] = S_on

    # Inf-sup constants for the decorrelated pair vs the enriched pair.
    # NOTE these are UPPER bounds (see infsup_report); the beta = 0 test inside
    # is exact, so a reported 0 is trustworthy and a large value is suggestive.
    b_dec, b_off = infsup_report(A, Theta, cone, enrichment=enrichments["PGA delta=0.5"])
    print(f"\n  beta^dec ([NDEE22] Eq. 14, the UNSTABILIZED pair): {b_dec:.4e}")
    print(f"  beta^off ([NDEE22] Eq. 30, V_N + S_R^red):         {b_off:.4e}")
    if b_dec > 1e-8:
        print("  -> beta^dec > 0 here, so this toy does NOT exhibit the")
        print("     instability [NDEE22] §3 repairs. B = I is parameter-")
        print("     independent and sign-definite; see rb_online.py's caveat")
        print("     and REPRODUCTION_NOTES.md. The wiring is exercised, but on")
        print("     this problem it has no instability to fix.")

    # Accuracy at several unseen parameters, both ways.
    rng = np.random.default_rng(123)
    mus = [np.array([rng.uniform(0.4, 1.2), rng.uniform(0.25, 0.75)])
           for _ in range(5)]

    print(f"\n  Reduced solve ([BEE20] Alg. 1 line 4) at {len(mus)} unseen mu")
    print(f"    {'primal space':<28}{'dim':>5}{'rel u err':>12}{'rel lam err':>13}"
          f"{'max viol':>11}")
    f = F[:, 0]
    for label, S in enrichments.items():
        eu, el, vi = [], [], []
        for mu_new in mus:
            gap = obstacle_gap(N, mu_new)
            u_hf, lam_hf = solve_hf(A, f, gap)
            u_rb, lam_rb = solve_reduced(A, f, gap, Theta, cone, enrichment=S)
            eu.append(np.linalg.norm(u_rb - u_hf) / np.linalg.norm(u_hf))
            el.append(np.linalg.norm(lam_rb - lam_hf) / np.linalg.norm(lam_hf))
            vi.append(max(0.0, float((u_rb - gap).max())))
            assert lam_rb.min() >= -1e-12, "reduced multiplier left the cone"
        dim = Theta.shape[1] if S is None else Theta.shape[1] + np.shape(S)[1]
        print(f"    {label:<28}{dim:>5}{np.mean(eu):>12.4e}{np.mean(el):>13.4e}"
              f"{np.max(vi):>11.3e}")
    print("    (means over the 5 mu; non-negativity of lambda_hat held in all"
          " cases)")


if __name__ == "__main__":
    main()
