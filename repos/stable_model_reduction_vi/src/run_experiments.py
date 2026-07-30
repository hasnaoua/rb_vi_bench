"""Reproduce the paper's qualitative claims on the 1-D obstacle test case.

Claims under test (all from Niakh/Drouet/Ehrlacher/Ern, ESAIM: M2AN 2022):

  C1  The decorrelated pair (V_N, W_R^+) need not be inf-sup stable. §5.1 finds
      beta^dec "almost zero (of the order of 10^-7)" for (N,R) = (89,50).
  C2  PGA recovers stability: beta^off > 0 (Eq. 30), via Prop. 3.1.
  C3  PGA's S_R^red is much smaller than S_R (§3).
  C4  sigma_{S^n}(mu_n) decreases with n and reaches 0 -- Lemma 3.2.
  C5  mCPG generators are more nearly orthogonal to the existing cone than
      CPG's: e_orth^CPG(r) <= e_orth^mCPG(r), Eq. (41) and Fig. 6/12.
  C6  mCPG's Gram matrix is better conditioned than CPG's (§4).
  C7  Proposition 3.1: where Eq. (23) holds, Eq. (24) is a valid positive lower
      bound on the inf-sup constant of the enriched pair.

These are qualitative reproductions on a different (1-D) problem. No number
here should match a number in §5.

C1 DOES NOT REPRODUCE HERE, for a structural reason worth stating up front:
on this test case beta^dec stays comfortably positive even when N < R. The
dimension count is necessary but not sufficient, because the inf in Eq. (14)
runs over a CONE, not a subspace: ker(C) has dimension R - N > 0, but it need
not meet the non-negative orthant. Here it provably never does -- Gordan's
theorem supplies a certificate y with C^T y > 0 componentwise. C2 is therefore
vacuous (PGA has no instability to repair), which is why C7 exists: it tests
the paper's actual theorem without depending on beta^dec vanishing.
See REPRODUCTION_NOTES.md.
"""

from __future__ import annotations

import numpy as np

import _shared_path  # noqa: F401  -- puts rb_vi_common on sys.path
from hf_model import ObstacleHF, training_set, validation_set
# The algorithms live in the shared package, tagged [NDEE22] there because it
# also holds [BEE20]'s CPG. Every equation number in this driver is [NDEE22]'s.
from rb_vi_common import (S_R_full, Whitener, boundedness_c_S, cpg, e_orth,
                          inf_sup, inf_sup_hf, mcpg, orth_union, pga, pod,
                          sigma_S, supremizer_space)


def main():
    np.set_printoptions(precision=3, suppress=True)

    hf = ObstacleHF(n_elem=200, n_dual=60)
    D_train = training_set(n_r=8, n_c=8)
    D_valid = validation_set(n=10)
    print(f"HF: {hf.n_v} primal dofs, {hf.n_w} dual dofs")
    print(f"D_train: {len(D_train)} params, D_valid: {len(D_valid)} params")

    # --- HF snapshots (§2.2) --------------------------------------------
    U, L = hf.snapshots(D_train)
    print(f"snapshots: primal {U.shape}, dual {L.shape}, "
          f"lambda >= 0: {L.min() >= 0}")
    active = (L > 1e-10).sum(axis=0)
    print(f"active constraints per mu: {active.min()}-{active.max()}")

    # Whiten once: V carries the energy inner product G_V = K, W carries l^2.
    wh = Whitener(hf.K, None)
    Ut = wh.L @ U                     # primal snapshots in whitened coords
    B_train = [wh.B_hat(hf.B(mu)) for mu in D_train]
    B_valid = [wh.B_hat(hf.B(mu)) for mu in D_valid]

    # --- Reduced bases: POD (Eq. 10-11) and the cone (Eq. 12-13) ---------
    # §5.1 picks tolerance pairs so as to "cover the two possible settings,
    # namely N < R where we are sure that for all mu ... b is not inf-sup
    # stable (beta^dec = 0) and N >= R where inf-sup stability cannot be
    # asserted a priori." We do the same with two pairs.
    delta_cpg = 2e-1
    res_cpg = cpg(L, delta=delta_cpg)
    res_mcpg = mcpg(L, delta=delta_cpg)
    R = res_mcpg.R
    print(f"\nCPG: R = {res_cpg.R}   mCPG: R = {res_mcpg.R}   "
          f"(delta_CPG = {delta_cpg})")

    # ================= C5, C6: mCPG vs CPG ==============================
    print("\n" + "=" * 62)
    print("C5/C6  mCPG vs CPG  (§4, Eq. 41)")
    eo_cpg, eo_mcpg = e_orth(res_cpg.generators), e_orth(res_mcpg.generators)
    m = min(len(eo_cpg), len(eo_mcpg))
    print(f"  mean e_orth  CPG {eo_cpg[:m].mean():.4f}   mCPG {eo_mcpg[:m].mean():.4f}")
    print(f"  C5 holds (e_orth^CPG <= e_orth^mCPG) on "
          f"{100*np.mean(eo_cpg[:m] <= eo_mcpg[:m] + 1e-12):.0f}% of iterations")

    def gram_cond(G):
        Gn = G / np.linalg.norm(G, axis=0, keepdims=True)
        return np.linalg.cond(Gn.T @ Gn)

    print(f"  Gram condition number  CPG {gram_cond(res_cpg.generators):.3e}"
          f"   mCPG {gram_cond(res_mcpg.generators):.3e}")

    # Use the mCPG cone downstream: §4 recommends it, and it is better
    # conditioned, which the cone projections in inf_sup depend on.
    cone = res_mcpg.generators

    S_full = S_R_full(B_train, cone)
    beta_hf = np.array([inf_sup_hf(Bh) for Bh in B_valid])       # Eq. (3)

    # Two regimes, as in §5.1: N < R and N >= R.
    for delta_pod in (1e-2, 1e-4):
        V_N = pod(Ut, delta=delta_pod)
        N = V_N.shape[1]
        regime = "N < R" if N < R else "N >= R"
        print("\n" + "=" * 62)
        print(f"(N, R) = ({N}, {R})   [{regime}]   delta_POD = {delta_pod:g}")
        print("=" * 62)

        # ---- C3: how much smaller is S_R^red than S_R? -----------------
        print("  C3  dim S_R^red vs dim S_R  (Eq. 20/28)")
        print(f"      dim S_R = {S_full.shape[1]}")
        runs = {}
        for d in (0.9, 0.7, 0.5):
            out = pga(B_train, V_N, cone, delta=d)
            runs[d] = out
            print(f"      delta_PGA = {d:.1f}  ->  dim S_R^red = {out.S.shape[1]:3d}"
                  f"   ({S_full.shape[1]/max(out.S.shape[1],1):.1f}x smaller)")

        # ---- C4: Lemma 3.2 ---------------------------------------------
        out = runs[0.5]
        s = np.array(out.sigmas)
        bound = min(len(D_train) * R, hf.n_v - N)
        print(f"\n  C4  Lemma 3.2 (delta_PGA = 0.5)")
        print(f"      iterations n = {out.n_iter};  sigma from {s[0]:.3f} "
              f"to {s[-1]:.3f}")
        print(f"      non-increasing: {bool(np.all(np.diff(s) <= 1e-12))}")
        print(f"      n <= min(m*R, dim V - dim V_N) = {bound}: "
              f"{out.n_iter <= bound}")

        # ---- C1, C2: inf-sup constants ---------------------------------
        V_off = orth_union(V_N, out.S) if out.S.shape[1] else V_N   # Eq. (29)
        rows = []
        for Bh in B_valid:
            b_dec = inf_sup(Bh, V_N, cone)                        # Eq. (14)
            Z = supremizer_space(Bh, cone)
            V_on = orth_union(V_N, Z)                             # Eq. (17)
            b_on = inf_sup(Bh, V_on, cone)                        # Eq. (18)
            b_off = inf_sup(Bh, V_off, cone)                      # Eq. (30)
            rows.append((b_dec, b_on, b_off))
        A = np.column_stack([beta_hf, np.array(rows)])

        print(f"\n  C1/C2  inf-sup on D_valid   (upper bounds -- see inf_sup docstring)")
        print(f"      {'':10s}{'min':>11s}{'median':>11s}{'max':>11s}")
        for name, col in zip(["beta_HF", "beta^dec", "beta^on", "beta^off"], A.T):
            print(f"      {name:10s}{col.min():11.3e}{np.median(col):11.3e}"
                  f"{col.max():11.3e}")
        print(f"      C1  beta^dec ~ 0 ?          max = {A[:,1].max():.3e}")
        print(f"      C2  beta^off > 0 for all ?  {bool((A[:,3] > 1e-8).all())}"
              f"   (min = {A[:,3].min():.3e})")
        print(f"      Eq.(18) beta^on >= beta_HF ? "
              f"{bool((A[:,2] >= A[:,0] - 1e-9).all())}")
        print(f"      monotone beta^dec <= beta^off <= ~beta^on ? "
              f"{bool((A[:,1] <= A[:,3] + 1e-9).all())}")

        # ---- C7: Proposition 3.1 itself --------------------------------
        # This is the paper's central theorem and, unlike C1/C2, it can be
        # checked whatever beta^dec happens to be. For each mu: if
        # sigma_S(mu) < beta^on/c_S (Eq. 23), then Eq. (24) asserts
        #     beta*_S = (beta^on - c_S sigma_S) / (1 + sigma_S)
        # is a valid POSITIVE lower bound for the inf-sup constant of the pair
        # (V_N + S, W_R^+) -- which is beta^off.
        print(f"\n  C7  Proposition 3.1  (Eq. 23 => Eq. 24)")
        n_appl, n_ok = 0, 0
        margins = []
        for j, Bh in enumerate(B_valid):
            Z = supremizer_space(Bh, cone)
            sig = sigma_S(Z, V_off)[0]
            c_S = boundedness_c_S(Bh, V_off, cone)
            b_on = A[j, 2]
            if sig < b_on / max(c_S, 1e-300):          # Eq. (23)
                n_appl += 1
                beta_star = (b_on - c_S * sig) / (1.0 + sig)   # Eq. (24)
                if beta_star > 0 and A[j, 3] >= beta_star - 1e-9:
                    n_ok += 1
                margins.append(A[j, 3] - beta_star)
        print(f"      criterion (23) holds for {n_appl}/{len(B_valid)} mu")
        if n_appl:
            print(f"      bound (24) valid (beta*_S > 0 and <= beta^off): "
                  f"{n_ok}/{n_appl}")
            print(f"      slack beta^off - beta*_S: min {min(margins):.3e}, "
                  f"median {np.median(margins):.3e}")


if __name__ == "__main__":
    main()
