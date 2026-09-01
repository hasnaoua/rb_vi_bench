"""[NDEE22] Algorithm 1 -- PGA: Projected Greedy Algorithm.

Origin: [NDEE22] §3. See ``rb_vi_common/__init__.py`` for the tag key.

CITATION CONVENTION FOR THIS MODULE
-----------------------------------
Every section, equation, lemma and proposition number in this module refers to
**[NDEE22]** (Niakh, Drouet, Ehrlacher, Ern, ESAIM: M2AN 2022). Nothing here
comes from [BEE20]. Note that [NDEE22] Algorithm 1 is PGA, whereas [BEE20]
Algorithm 1 is that paper's online stage -- a different thing entirely.

THE PROBLEM PGA SOLVES
----------------------
To make the reduced saddle-point problem inf-sup stable, §2.3 enriches the
primal basis with supremizers, giving (Eq. 17)

    V^on_{N,R}(mu) := V_N + S_R(mu),   S_R(mu) := Span{ T(mu) chi_r }.

This works -- Eq. (18) gives beta^on >= beta_HF >= beta_0 > 0 -- but when the
constraint b(mu; . , .) is PARAMETER-DEPENDENT, so is S_R(mu). The enriched
space then "has to be constructed in the online phase, which is computationally
inefficient" (§2.3).

PGA removes that cost. It builds ONE parameter-independent subspace
S_R^red of S_R := sum_{mu in D_train} S_R(mu) (Eq. 20), offline, such that
(Eq. 27)

    sup_{mu in D_train} sigma_{S_R^red}(mu)  <=  delta.

Proposition 3.1 is what makes this sufficient: if sigma_S(mu) < beta^on/c_S
(Eq. 23) then the pair (V_N + S, W_R^+) is inf-sup stable with the explicit
constant of Eq. (24). §3 notes that in practice "a simple choice is
delta_PGA < beta_0^-1 C_0".

Termination is guaranteed by Lemma 3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reduction import orth_union, sigma_S, supremizer_space


@dataclass
class PGAResult:
    S: np.ndarray                              # (dim, n) basis of S_R^red, Eq. (28)
    sigmas: list = field(default_factory=list)  # sigma_{S^n}(mu_n) per iteration
    mus: list = field(default_factory=list)     # the greedily selected mu_n
    n_iter: int = 0


def pga(B_hats, V_N, cone_tilde, delta, max_iter=None, verbose=False):
    """Run [NDEE22] Algorithm 1.

    Parameters
    ----------
    B_hats : list of (n_w, n_v) arrays
        Whitened constraint matrices B_hat(mu) for each mu in D_train.
    V_N : (n_v, N) array
        Orthonormal basis of the primal reduced space V_N, from POD (Eq. 10).
    cone_tilde : (n_w, R) array
        Whitened generators chi_r of the dual cone W_R^+ (Eq. 12).
    delta : float
        The tolerance delta_PGA of Eq. (27).
    max_iter : int, optional
        Not in the paper. Lemma 3.2 bounds the iteration count by
        min(#D_train * R, dim(V) - dim(V_N)) already; this is a guard for
        pathological inputs.

    Returns
    -------
    PGAResult with S := S^n, the subspace of Eq. (28).
    """
    # S_R(mu) for every training parameter, precomputed: [NDEE22] Algorithm 1 revisits
    # all of them at lines 2 and 9 on every iteration.
    Zs = [supremizer_space(Bh, cone_tilde) for Bh in B_hats]

    n_v = V_N.shape[0]
    cap = (n_v - V_N.shape[1]) if max_iter is None else max_iter

    # 1: S^0 := {0}
    gens: list[np.ndarray] = []

    def basis():
        """Orthonormal basis of V_N + S^n."""
        if not gens:
            return V_N
        return orth_union(V_N, np.column_stack(gens))

    def all_sigmas(Q):
        return np.array([sigma_S(Z, Q)[0] for Z in Zs])

    # 2: mu_0 := argmax_{mu in D_train} sigma_{S^0}(mu)
    Q = basis()
    sig = all_sigmas(Q)
    idx = int(np.argmax(sig))

    sigmas, mus = [], []
    # 3: n := 0
    n = 0

    # 4: while sigma_{S^n}(mu_n) > delta do
    while sig[idx] > delta and n < cap:
        sigmas.append(float(sig[idx]))
        mus.append(idx)
        if verbose:
            print(f"  PGA n={n:3d}  mu_n=#{idx:3d}  sigma={sig[idx]:.6e}")

        # 5: v^(1)_{n+1} := argmax_{v in S_R(mu_n), ||v||<=1} ||(I - Pi_{V_N+S^n}) v||
        # sigma_S returns the maximizing coefficient vector, so the argmax is
        # Z c_max. This is the "eigenvalue problem" of §3 -- concretely the
        # leading singular pair of (I - QQ^T) Z.
        _, c = sigma_S(Zs[idx], Q)
        # sigma_S hands back c=None only for an empty S_R(mu), whose sigma is 0.0 and
        # so cannot have passed this loop's own `sig[idx] > delta` test.
        assert c is not None
        v1 = Zs[idx] @ c

        # 6: v^(2)_{n+1} := (I - Pi_{V_N+S^n}) v^(1)_{n+1}
        v2 = v1 - Q @ (Q.T @ v1)

        # 7: v_{n+1} := v^(2) / ||v^(2)||
        nrm = np.linalg.norm(v2)
        if nrm <= 1e-14:
            # Lemma 3.2 shows v^(2) != 0 whenever sigma > 0, so this is
            # unreachable in exact arithmetic; stop rather than divide by ~0.
            break
        gens.append(v2 / nrm)

        # 8: S^{n+1} := S^n + Span{v_{n+1}}
        Q = basis()

        # 9: mu_{n+1} := argmax_{mu in D_train} sigma_{S^{n+1}}(mu)
        sig = all_sigmas(Q)
        idx = int(np.argmax(sig))

        # 10: n := n + 1
        n += 1

    # 12: return S := S^n
    S = np.column_stack(gens) if gens else np.zeros((n_v, 0))
    return PGAResult(S=S, sigmas=sigmas, mus=mus, n_iter=n)


def S_R_full(B_hats, cone_tilde):
    """S_R of Eq. (20): the sum over D_train of the S_R(mu).

    Computed only to measure how much smaller PGA's S_R^red is -- §3 predicts
    "a much smaller dimension than S_R", and Eq. (27) is achievable trivially by
    S_R itself since sigma_{S_R}(mu) = 0.
    """
    blocks = [supremizer_space(Bh, cone_tilde) for Bh in B_hats]
    return orth_union(*blocks)
