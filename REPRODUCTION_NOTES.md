# Reproduction notes

Paper: **Stable model reduction for linear variational inequalities with
parameter-dependent constraints** — Idrissa Niakh, Guillaume Drouet, Virginie
Ehrlacher, Alexandre Ern. ESAIM: M2AN (2022), doi
[10.1051/m2an/2022077](https://doi.org/10.1051/m2an/2022077). Dated 9 Sept 2022.

Source: local PDF, read directly (24 pages). Unlike the previous paper in this
series, the equations came through cleanly — every citation below was read from
the rendered page, not from a lossy text extraction.

**Official code: none found.** §5 states the authors used FreeFem++ [23] and
Python with `cvxopt` [4]; no repository is linked. Nothing here derives from
their code.

This paper is the direct sequel to Benaceur/Ern/Ehrlacher (its reference [9]),
which is implemented in `~/rb_contact_cpg`. The CPG algorithm here is that
paper's; Algorithm 2 modifies it.

---

## Restructuring: the algorithms now live in a shared library

The algorithms were moved to [`~/rb_vi_shared`](../rb_vi_shared), shared with
[`~/rb_contact_cpg`](../rb_contact_cpg), which implements this paper's reference
**[9]**. Nothing about the transcriptions changed; the files moved, and
`run_experiments.py` reproduces its previous output exactly.

**This created one hazard, and it is handled explicitly.** The two papers'
numbering collides: `Eq. (57)` means different things in each, and both have an
**Algorithm 1** and an **Algorithm 2** denoting different algorithms (here,
Algorithm 1 = PGA and Algorithm 2 = mCPG; in [9], Algorithm 1 = the online stage
and Algorithm 2 = CPG). Every citation in shared code therefore carries a paper
tag — `[NDEE22]` for this paper, `[BEE20]` for [9] — and a bare number in
`rb_vi_common/` is a bug. `reduction.py` and `pga.py` are `[NDEE22]` end to end
and say so in a module-level convention block; algorithm references there are
tagged anyway, since those are the collisions that actually mislead.

Note that `[BEE20]`-tagged code carries two caveats this repository's code does
not: its numbering is **hal preprint v2** numbering, not the published IJNME
article's, and its equations were recovered from a PDF with a shifted Type-1 font
encoding and were read by eye. Neither applies to anything tagged `[NDEE22]`.

| Was | Now |
|---|---|
| `src/cone_greedy.py` | `rb_vi_common/cone_greedy.py` |
| `src/pga.py` | `rb_vi_common/pga.py` |
| `src/reduction.py` | `rb_vi_common/reduction.py` |

`src/hf_model.py` stayed: it is this repository's substitute for §5.1 and
reproduces no number in the paper.

**Three CPG-family functions now coexist**, deliberately. `mcpg` is Algorithm 2
and the primary algorithm. `cpg` is the Remark 4.3 baseline as *this* paper
describes [9] — normalized generators, relative tolerance Eq. (13). And
`cone_projected_greedy` is the same algorithm transcribed from **[9] itself** —
raw generators, absolute tolerance its Eq. (58), explicit `‖·‖_Λ` Gram matrix.
Merging them would mean silently attributing one paper's conventions to the
other. Their equivalence is instead **tested**:
`~/rb_vi_shared/tests/test_equivalence.py` confirms that at matched tolerance
they select the same parameters in the same order and generate the same cone as
a set, with generator matrices differing by exactly the snapshot normalization.

---

## What is implemented

| Paper element | Status |
|---|---|
| **Algorithm 1 — PGA** | **Implemented**, line by line (`rb_vi_common/pga.py`) |
| **Algorithm 2 — mCPG** | **Implemented**, line by line (`rb_vi_common/cone_greedy.py`) |
| CPG (baseline, ref. [9]) | Implemented, per Remark 4.3 (same file) |
| Supremizer operator, Eq. (15)–(17) | Implemented (`rb_vi_common/reduction.py`) |
| σ_S(µ), Eq. (21) | Implemented |
| c_S(µ), Eq. (22) | Implemented (as an upper bound — see below) |
| β_HF, β^dec, β^on, β^off, Eq. (3)(14)(18)(30) | Implemented |
| Proposition 3.1, Eq. (23)–(24) | Implemented and **verified numerically** |
| Lemma 3.2 | Verified numerically |
| POD with tolerance, Eq. (10)–(11) | Implemented |
| e_orth, Eq. (41) | Implemented |
| §5.1 membrane obstacle (2-D, FreeFem++) | **Not** implemented — 1-D analogue |
| §5.2 Hertz contact between half-disks | **Not** implemented |
| Timing / benefit-threshold study (Tables 1–2) | **Not** implemented |

**No number here reproduces a number in §5.**

---

## The test problem

A 1-D obstacle problem with a **parameter-dependent constraint** — the one
structural feature the paper turns on. `µ = (r, c)` sets the position and width
of the obstacle window; the multiplier lives on a reference domain and is mapped
to physical collocation points `s_i(µ) = c + r·ŝ_i`, so `B(µ)` genuinely varies
with `µ` while `g` does not. That mirrors §5.1, where `ψ̂` is prescribed on the
reference domain and is parameter-independent.

**Retuning that was necessary.** §5.1 uses `ψ̂(s) = -1.25 s²`, whose peak is `0` —
exactly the Dirichlet level. In 1-D that makes `u - ψ` strictly convex, so the
contact set collapses to a **single point** and the multiplier carries no spatial
structure at all (2–4 active constraints, nothing for the cone algorithms to
compress). Matching value and slope at the free boundary gives the contact
half-width

```
d = [1 - sqrt(1 - 4(0.125 - offset)/(k + 0.5))]/2,     k = curv/r²
```

so `curv` must sit well below `r²`. With `offset = 0.060, curv = 0.005` the
contact patch covers roughly 55–70% of the window and moves with `µ`: 19–43
active constraints out of 60. The paper tunes its own coefficient for a
comparable reason ("chosen so that the constraints at the boundary of ω(µ) are
inactive in order to avoid oscillations").

Other deviations, all deliberate: 1-D not 2-D; collocation multipliers rather
than the P1/P1 pair of §5.1; `a(µ;·,·)` taken parameter-independent (in §5.1 it
inherits `µ` through the geometric mapping, Eq. 37 — this changes nothing
algorithmically, since `S_R(µ)` depends on `µ` only through `B(µ)`).

---

## Results

Run `python3 src/run_experiments.py`. Two regimes, as in §5.1.

| Claim | Result |
|---|---|
| **C3** `S_R^red ≪ S_R` | ✅ dim `S_R` = 125 → 32 at `δ_PGA = 0.9` (3.9× smaller) |
| **C4** Lemma 3.2 | ✅ `σ` non-increasing; iteration count within the bound |
| **C5** `e_orth^CPG ≤ e_orth^mCPG` | ✅ holds on 92% of iterations (mean 0.58 vs 0.61) |
| **C6** mCPG better conditioned | ✅ Gram cond. 7.7e2 (CPG) vs 6.4e2 (mCPG) |
| **C7** Proposition 3.1 | ✅ Eq. (23) holds for 8/10 and 9/10 µ; where it holds, Eq. (24) is valid in **every** case |
| Eq. (18) `β^on ≥ β_HF` | ✅ all µ |
| Ordering `β^dec ≤ β^off` | ✅ all µ |
| **C1** `β^dec ≈ 0` | ❌ **does not reproduce** — see below |
| **C2** PGA restores stability | ⚠️ vacuous here, since C1 does not occur |

---

## C1 does not reproduce, and the reason is structural

§5.1 states that for `N < R` "we are sure that for all µ ∈ 𝒟, the bilinear form
`b(µ;·,·)` is not inf-sup stable (`β^dec = 0`)". On this test problem `β^dec`
stays around 1.5 even at `(N,R) = (5,27)`.

This is **not** a bug in the implementation, and it is **not** a claim that the
paper is wrong. It is a property of the test problem, and it is provable.

Writing `C = Qᵀ B̂(µ)ᵀ X` (primal basis `Q`, cone generators `X`),

```
β^dec = min_{α ≥ 0, α ≠ 0} ‖Cα‖ / ‖Xα‖,     so     β^dec = 0  ⟺  ker(C) meets the non-negative orthant.
```

`N < R` guarantees `dim ker(C) = R - N > 0`. But the inf in Eq. (14) runs over a
**cone**, not a subspace, so a non-trivial kernel is necessary and **not
sufficient** — the kernel must additionally intersect the orthant.

Here it never does. With a single lower obstacle every constraint row is
`-φ(s_i)` with the P1 basis `φ ≥ 0`, so every supremizer lies in one halfspace
and no non-negative combination can cancel. Two independent checks confirm it at
`(N,R) = (5,27)`:

- the LP `{Cα = 0, Σα = 1, α ≥ 0}` is **infeasible**;
- **Gordan's theorem** supplies the certificate: `max t s.t. Cᵀy ≥ t·1, |y|≤1`
  returns `t = 2.56 > 0`, i.e. there exists `y` with `Cᵀy > 0` componentwise,
  which proves no non-negative vector lies in `ker(C)`.

I tried to induce the instability with a two-sided (channel) constraint, where
lower and upper blocks have opposite signs and cancellation becomes possible.
That code is retained (`ObstacleHF(channel_gap=...)`) and produces genuine
two-sided activity, but `β^dec` still stayed positive at every `(N,R)` tried.

### The "richer 2-D geometry" conjecture was tested, and does not hold

This note previously conjectured that producing the instability needs the richer
geometry of a 2-D contact surface — §5.2 has 280 dual dofs on a curved contact
manifold, where supremizers might genuinely oppose one another. **That conjecture
has since been tested and is not supported.**

`~/rb_contact_cpg/src/hertz_infsup_probe.py` builds the two cheapest versions of
that richer geometry on top of that repository's validated Hertz contact model:

- a **rotating contact normal** on a curved arc — vector-valued displacement,
  `µ`-dependent collocation, so `B(µ)` genuinely varies;
- **two deformable bodies** of different moduli, so the constraint acts on the
  displacement jump `[[u]]` and every row carries a positive and a negative block
  — the mechanism `channel_gap` reached for and could not realize in 1-D.

Across **40 configurations** — half-angles 0.05 to 1.30, `N` swept from 1 to
`R−1` so `dim ker(C)` reaches ~21 — **every one returns a positive Gordan
certificate**, so `β^dec > 0` stays *proven*. Neither curvature nor the two-body
jump is the missing ingredient.

The certificate does weaken monotonically as `N` falls and the arc curves, from
`t ≈ 1.7` at `N = R−1` to `t ≈ 5e-2` at `N = 2`, half-angle 1.30 — the
obstruction erodes with exactly the variables one would expect, without reaching
zero.

**A caution this repository's own C1 finding shares.** The prose argument above —
"every supremizer lies in one halfspace, so no non-negative combination can
cancel" — proves something *stronger than necessary*. `β^dec = 0` does not
require the supremizer `B̂ᵀη` to vanish; it requires only that it be **orthogonal
to `V_N`**. The halfspace argument is therefore sufficient but not necessary, and
what actually carries the C1 finding is the numerical Gordan certificate, which
is run on the correct object `C = Qᵀ B̂ᵀ X` (it includes `Q`). The conclusion
stands; the informal justification for it is weaker than it reads.

**Where to look next.** The variable not yet varied is the **dual
discretization**. §5.1 uses a P1/P1 pair and §6.1 of [9] uses collocation or LAC;
every implementation in these two repositories uses **collocation**, which keeps
constraint rows nearly independent. A P1 dual couples neighbouring multipliers
and is the most plausible remaining source of the cancellation C1 requires.

Because C2 is vacuous without C1, **C7 was added**: Proposition 3.1 is the
theorem the whole construction rests on, and it can be tested regardless of what
`β^dec` happens to be. It verifies.

---

## UNSPECIFIED items

| # | Item | Where | Choice | Alternatives |
|---|---|---|---|---|
| 1 | How `β` is computed anywhere in §5 | `rb_vi_common/reduction.inf_sup` | Exact convex-QP test for `β = 0`, then multi-start projected gradient for the value | An SOCP/QP sequence (the natural `cvxopt` route); face enumeration (exact, exponential in `R`) |
| 2 | Solver for Algorithm 2 line 9 | `rb_vi_common/cone_greedy._closest_in_cone_below` | SLSQP on the equivalent least-squares problem | Any QP solver; `cvxopt` is the authors' |
| 3 | `δ_PGA`, `δ_POD`, `δ_CPG` for a 1-D problem | `run_experiments.py` | Swept, not fixed | §5.1's values are calibrated to its own 2-D discretization |
| 4 | Tie-breaking in the `argmax` of Alg. 1 lines 2/9 and Alg. 2 line 12 | both | First maximum (`np.argmax`) | Random; other orderings |
| 5 | Inner product on `W` | `run_experiments.py` | `ℓ²` | §5.1 uses `L²(ω̂)`; with uniform collocation these differ by a constant factor |

**Item 1 matters most.** `inf_sup` returns an **upper bound** on `β`, because
minimizing a quotient over a cone is non-convex. It is therefore reliable when
showing `β` is *small* and only suggestive when showing `β` is *large*. The
`β = 0` test is separate and exact (a convex QP), which is what makes the C1
finding above trustworthy.

`c_S(µ)` (Eq. 22) is computed as a supremum over the cone's **span**, an upper
bound on the true `c_S`. Since `c_S` enters Prop. 3.1 only through the criterion
Eq. (23) and the bound Eq. (24), overestimating it makes both conservative — so
the C7 verification stays valid.

---

## Running

```bash
cd src && python3 run_experiments.py
```

`numpy` + `scipy` only; about 10 s on CPU.
