"""Checks on the shared cone algorithms.

Run:  python3 tests/test_equivalence.py      (from ~/rb_vi_shared)

The point of the first test is the claim made in ``cone_greedy``'s module
docstring: the two CPG transcriptions -- [BEE20] Algorithm 2 and the [NDEE22]
Remark 4.3 baseline -- are the same algorithm in different conventions. That
claim is the justification for keeping both functions rather than deleting one,
so it is checked rather than asserted. See ``rb_vi_common/__init__.py`` for the
tag key.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rb_vi_common import (cone_projected_greedy, cpg, e_orth, mcpg,  # noqa: E402
                          project_onto_cone)


def _snapshots(seed=0, dim=40, n=25):
    """Non-negative snapshots with a moving, compactly supported bump.

    Shaped like a contact multiplier -- sparse, non-negative, with a support
    that translates and rescales with the parameter -- so the cone algorithms
    face the structure they are designed for rather than dense random noise.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, dim)
    cols = []
    for _ in range(n):
        c, w, a = rng.uniform(0.25, 0.75), rng.uniform(0.08, 0.25), rng.uniform(0.5, 2.0)
        cols.append(a * np.clip(1.0 - ((x - c) / w) ** 2, 0.0, None))
    return np.column_stack(cols)


def test_two_cpg_transcriptions_agree():
    """[BEE20] Alg. 2 and the [NDEE22] Rmk 4.3 baseline build the same cone.

    Differences that must NOT matter:
      (a) generators raw ([BEE20] line 6) vs normalized ([NDEE22] Rmk 4.3) --
          positive rescaling does not change span_+;
      (c) argmax over all of P_tr ([BEE20] line 5) vs over the unselected
          ([NDEE22] line 12) -- selected snapshots project exactly.

    Difference that DOES matter and is compensated here:
      (b) [BEE20] Eq. (58) is absolute, [NDEE22] Eq. (13) relative. Setting
          eps_du = delta * max_p ||lambda(mu_p)|| makes the two criteria the
          same inequality.
    """
    for seed in range(5):
        T = _snapshots(seed=seed)
        denom = np.linalg.norm(T, axis=0).max()
        for delta in (0.5, 0.3, 0.15, 0.05):
            a = cone_projected_greedy(T, eps_du=delta * denom)
            b = cpg(T, delta=delta)

            assert a.R == b.R, f"seed={seed} delta={delta}: R {a.R} vs {b.R}"
            assert a.selected_indices == b.order, (
                f"seed={seed} delta={delta}: selection order differs\n"
                f"  [BEE20]  {a.selected_indices}\n  [NDEE22] {b.order}")

            # Same cone as a SET: projecting arbitrary probes must agree, even
            # though the generator matrices differ by a column scaling.
            rng = np.random.default_rng(100 + seed)
            for _ in range(8):
                y = rng.random(T.shape[0])
                pa = project_onto_cone(y, a.generators)[0]
                pb = project_onto_cone(y, b.generators)[0]
                assert np.allclose(pa, pb, atol=1e-8), (
                    f"seed={seed} delta={delta}: cones differ as sets")

            # And the column scaling is exactly snapshot normalization.
            scale = np.linalg.norm(a.generators, axis=0)
            assert np.allclose(a.generators / scale, b.generators, atol=1e-10)
    print("PASS  two CPG transcriptions agree (cone, order, cardinality)")


def test_mcpg_stays_in_the_positive_cone():
    """[NDEE22] Algorithm 2 line 10 must yield nu_r in W^+.

    This is the property that distinguishes mCPG from Gram-Schmidt: [NDEE22] §4
    rejects orthogonalization because "this would lead to a departure from the
    positive cone W^+". Line 9's second constraint (Upsilon <= theta
    componentwise) is what enforces it, so a negative entry here means line 9 is
    wrong, not merely inaccurate.
    """
    for seed in range(5):
        T = _snapshots(seed=seed)
        res = mcpg(T, delta=0.1)
        assert res.generators.min() >= -1e-12, (
            f"seed={seed}: mCPG generator left W^+, min={res.generators.min():.3e}")
        assert np.allclose(np.linalg.norm(res.generators, axis=0), 1.0, atol=1e-10), (
            "line 10 normalizes nu_r; norms should be 1")
    print("PASS  mCPG generators stay in W^+ and are normalized")


def test_e_orth_bounded_and_mcpg_better_conditioned():
    """[NDEE22] Eq. (41): e_orth <= 1, and §4's conditioning claim.

    [NDEE22] reports e_orth^CPG(r) <= e_orth^mCPG(r) and a better-conditioned
    mCPG Gram matrix. Both are empirical claims about the paper's own 2-D test
    case, so this checks the bound (which is structural) and REPORTS the
    comparison rather than asserting it.
    """
    def gram_cond(G):
        Gn = G / np.linalg.norm(G, axis=0, keepdims=True)
        return np.linalg.cond(Gn.T @ Gn)

    wins = 0
    for seed in range(5):
        T = _snapshots(seed=seed)
        rc, rm = cpg(T, delta=0.1), mcpg(T, delta=0.1)
        eo_c, eo_m = e_orth(rc.generators), e_orth(rm.generators)
        assert eo_c.max() <= 1.0 + 1e-9 and eo_m.max() <= 1.0 + 1e-9, "Eq. (41) bound"
        if gram_cond(rm.generators) <= gram_cond(rc.generators):
            wins += 1
    print(f"PASS  e_orth <= 1 (Eq. 41); mCPG better conditioned on {wins}/5 seeds")


def test_cone_is_hierarchical():
    """[BEE20] §5 calls the cones "nested": residuals must not increase."""
    T = _snapshots(seed=3)
    res = cone_projected_greedy(T, eps_du=1e-12, max_R=12)
    d = np.diff(res.residuals)
    assert (d <= 1e-12).all(), f"residuals increased: {res.residuals}"
    print("PASS  CPG residuals non-increasing (hierarchical cones)")


def test_rejects_negative_snapshots():
    """[BEE20] §2: lambda(mu) in W_R^+. A negative entry is a broken HF solve."""
    T = _snapshots(seed=0)
    T[0, 0] = -1e-6
    try:
        cone_projected_greedy(T, eps_du=1.0)
    except ValueError:
        print("PASS  negative snapshots rejected")
        return
    raise AssertionError("negative snapshots were accepted")


if __name__ == "__main__":
    test_two_cpg_transcriptions_agree()
    test_mcpg_stays_in_the_positive_cone()
    test_e_orth_bounded_and_mcpg_better_conditioned()
    test_cone_is_hierarchical()
    test_rejects_negative_snapshots()
    print("\nall checks passed")
