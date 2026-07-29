# Stable model reduction for linear variational inequalities

Implementation of the two algorithmic contributions of:

> I. Niakh, G. Drouet, V. Ehrlacher, A. Ern.
> *Stable model reduction for linear variational inequalities with
> parameter-dependent constraints.* ESAIM: M2AN (2022),
> [doi:10.1051/m2an/2022077](https://doi.org/10.1051/m2an/2022077).

Every non-trivial line cites the section, equation, or algorithm line it comes
from. Choices the paper leaves open are marked `[UNSPECIFIED]` at the point of
use and collected in [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

**Read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) first.** One of the paper's
claims does not reproduce on this test problem, for a reason that is proved
rather than guessed, and the notes say exactly what was and was not tested.

This is the sequel to Benaceur/Ern/Ehrlacher (reference [9]), implemented in
`~/rb_contact_cpg`. The CPG algorithm here is that paper's; Algorithm 2 modifies it.

## The problem

For a parameter `µ`, minimize an energy under an inequality constraint (Eq. 1),
dualized into a saddle-point problem (Eq. 6–7) with a multiplier `λ(µ) ∈ W⁺`.
Model reduction needs a primal basis `V_N` and a dual cone `W_R⁺`. Building them
independently — POD for the primal, a cone algorithm for the dual — gives no
guarantee that the reduced pair is **inf-sup stable** (Eq. 14).

The classical repair is to enrich the primal basis with **supremizers**
`T(µ)χ_r` (Eq. 15–17), which restores stability (Eq. 18). But when the
constraint is parameter-dependent, so is `S_R(µ)` — so the enrichment must be
redone **online**, for every new `µ`. §2.3: "computationally inefficient."

## Contribution 1 — PGA (Algorithm 1)

Build **one parameter-independent** subspace `S_R^red ⊂ S_R` offline, such that

```
sup_{µ ∈ D_train} σ_{S_R^red}(µ) ≤ δ            (Eq. 27)
```

**Proposition 3.1** is what makes this enough: if `σ_S(µ) < β^on/c_S` (Eq. 23),
the pair `(V_N + S, W_R⁺)` is inf-sup stable with the explicit constant

```
β*_S = (β^on − c_S·σ_S) / (1 + σ_S) > 0        (Eq. 24)
```

PGA greedily adds the direction of `S_R(µ_n)` worst captured by what it has so
far — a leading singular pair, the "eigenvalue problem" of §3 — until Eq. (27)
holds. **Lemma 3.2** guarantees termination.

## Contribution 2 — mCPG (Algorithm 2)

A cone with a **wider aperture**, so the reduced dual basis is better
conditioned. §4: one wants "a basis such that the corresponding Gram matrix is
as well-conditioned as possible", but Gram–Schmidt is unavailable because it
"would lead to a departure from the positive cone `W⁺`".

Remark 4.3 gives the difference in one line: CPG appends the normalized snapshot
`θ_q/‖θ_q‖`; mCPG first subtracts `Υ_r`, the closest point of the existing cone
that stays **below** `θ_q` in the cone order, and appends the normalized
residual. The constraint `Υ ∈ K_{r-1} ∩ (θ_q − W⁺)` is what keeps the new
generator inside `W⁺` — that is the trick that replaces orthogonalization.

## Layout

The algorithms live in a **shared library**, [`~/rb_vi_shared`](../rb_vi_shared),
together with this paper's reference **[9]** — Benaceur/Ern/Ehrlacher, implemented
in [`~/rb_contact_cpg`](../rb_contact_cpg). Read that library's README before
touching a citation: because both papers number an "Algorithm 1", an
"Algorithm 2" and an "Eq. (57)" differently, **every citation in shared code
carries a paper tag** — `[NDEE22]` for this paper, `[BEE20]` for reference [9].

```
~/rb_vi_shared/rb_vi_common/
    reduction.py         whitening, POD, supremizers, σ_S, c_S, inf-sup  [NDEE22] §2–§3
    cone_greedy.py       Algorithm 2 (mCPG), the Rmk 4.3 CPG baseline,   [NDEE22] §4
                         e_orth, plus [BEE20]'s own CPG transcription
    pga.py               Algorithm 1 (PGA)                               [NDEE22] §3
    cone_projection.py   Π_{K} via NNLS                                  [BEE20] §5

src/hf_model.py          1-D obstacle problem, parameter-dependent constraint
src/run_experiments.py   the claim-by-claim driver
src/_shared_path.py      locates rb_vi_common (set RB_VI_SHARED to relocate)
notebooks/walkthrough.ipynb
```

`hf_model.py` deliberately stays here: it is this repository's substitute for
§5.1's 2-D FreeFem++ problem, reproduces no number in the paper, and merging it
with the sibling's test problems would blur which paper a result speaks to.

Everything is computed in **whitened coordinates**, where the `V` and `W` inner
products become Euclidean and the Riesz isomorphism of Eq. (15) folds into the
change of basis — so the formulas read as in the paper. See
`rb_vi_common/reduction.py`.

### Why the shared library has two CPGs

`mcpg` is what this paper's §4 supersedes CPG with, and is the primary algorithm.
`cpg` is the Remark 4.3 baseline as *this* paper describes it (normalized
generators, relative tolerance Eq. 13). `cone_projected_greedy` is the same
algorithm transcribed from **[BEE20] itself** (raw generators, absolute tolerance
Eq. 58, explicit `‖·‖_Λ` Gram matrix). Both are kept rather than merged, so that
neither paper's conventions are silently attributed to the other; their
equivalence is *tested* in `~/rb_vi_shared/tests/test_equivalence.py`, which
checks they select the same parameters in the same order and generate the same
cone as a set.

## Run

```bash
cd src && python3 run_experiments.py
```

`numpy` + `scipy` only; about 10 s on CPU. Nothing needs installing for the
shared library — `_shared_path.py` finds it at `../../rb_vi_shared`.

## What reproduces

`S_R^red` is 3.9× smaller than `S_R`; Lemma 3.2 holds; mCPG's generators are
more nearly orthogonal to the existing cone than CPG's and its Gram matrix is
better conditioned; and **Proposition 3.1 verifies in every applicable case**.

What does **not** reproduce: `β^dec ≈ 0`. On this 1-D problem the decorrelated
pair stays inf-sup stable even when `N < R`, because the inf runs over a *cone*
and a non-trivial kernel need not meet it — Gordan's theorem gives a certificate
that it never does here. Full argument in
[REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).
