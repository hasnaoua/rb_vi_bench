# Reproduction notes

Paper: **A reduced basis method for parametrized variational inequalities applied to
contact mechanics** — Amina Benaceur, Alexandre Ern, Virginie Ehrlacher.
Preprint, HAL `hal-02081485v2`, submitted 27 Mar 2019 (v3 exists: 24 Sep 2019).
Source PDF: <https://hal.science/hal-02081485v2/file/contact.pdf>

**Official code: none found.** §6 states the authors used FreeFem++ [12] + Python
with `cvxopt` [1], but no repository is linked in the paper. Nothing here is derived
from their code; it is written from the text.

**This implementation was generated against v2. A v3 exists and was not consulted** —
numbered equations and algorithm content may differ between versions.

---

## Restructuring: the algorithms now live in a shared library

The cone algorithms were moved to [`~/rb_vi_shared`](../rb_vi_shared), shared with
[`~/stable_model_reduction_vi`](../stable_model_reduction_vi), which implements the
direct sequel (Niakh/Drouet/Ehrlacher/Ern, ESAIM: M2AN 2022 — this paper is its
reference **[9]**). Nothing about the transcriptions changed; the files moved.

**This created one hazard, and it is handled explicitly.** The two papers'
numbering collides: `Eq. (57)` means different things in each, and both have an
**Algorithm 1** and an **Algorithm 2** denoting different algorithms (here,
Algorithm 1 = the online stage and Algorithm 2 = CPG; there, Algorithm 1 = PGA
and Algorithm 2 = mCPG). Every citation in shared code therefore carries a paper
tag — `[BEE20]` for this paper, `[NDEE22]` for the sequel — and a bare number in
`rb_vi_common/` is a bug. **Every `[BEE20]` number still means v2 numbering**, and
the PDF-extraction hazard below still applies to all `[BEE20]`-tagged code.

Both CPG transcriptions were kept rather than merged. The sequel describes CPG in
its own conventions (normalized generators, relative tolerance) which differ from
this paper's Eq. (58); collapsing them would mean silently attributing one
paper's convention to the other. Their equivalence is instead **tested** —
`~/rb_vi_shared/tests/test_equivalence.py` confirms they select the same
parameters in the same order and generate the same cone as a set, with generator
matrices differing by exactly the snapshot normalization.

| Was | Now |
|---|---|
| `src/cpg.py` | `rb_vi_common/cone_greedy.py::cone_projected_greedy` |
| `src/cone_projection.py` | `rb_vi_common/cone_projection.py` |

`src/nmf_baseline.py`, `src/toy_problem.py` and `src/contact_dataset.py` stayed —
the first is §6.4 and unused by the sequel, the others are this repository's own
substitutes for §6 and reproduce no number in either paper.

---

## What is implemented

| Paper element | Status |
|---|---|
| Algorithm 2 — Cone-Projected Greedy | **Implemented**, line by line (`rb_vi_common/cone_greedy.py`) |
| Eq. (56), (57), (58) — selection + stopping | **Implemented** (same) |
| Cone projection `Π_{K⁺}` | **Implemented** as NNLS (`rb_vi_common/cone_projection.py`) |
| §6.4 / Eq. (66)–(69) — NMF baseline | **Implemented** (`src/nmf_baseline.py`) |
| POD primal basis (§5) | **Implemented** (`src/rb_online.py`) |
| Algorithm 1 lines 1, 4 — reduced saddle-point solve | **Implemented, linear-constraint case only** |
| Inf-sup stabilization of that solve | **Added, from the sequel** — see below |
| Algorithm 1 lines 2–3 — EIM of the constraint (§4.3) | **NOT implemented** |
| §3 — 2-D linear elasticity, non-interpenetration | **NOT implemented** |
| §6.1 — collocation / LAC constraint discretization | **NOT implemented** |
| §6.2–6.3 — Hertz half-disks, ring-on-block | **NOT implemented** |
| Kačanov iteration, Eq. (12)–(13) | **NOT exercised** (toy constraint is linear) |
| Non-matching meshes | **NOT implemented** |

The core contribution — the CPG algorithm — is implemented faithfully. The
application layer around it is not. `src/toy_problem.py` substitutes a 1-D obstacle
problem so the algorithm can be run and checked on CPU without a FEM stack.

**No number in this repository reproduces any number in §6.**

---

## The reduced solve now has an optional inf-sup stabilization

Algorithm 1 line 4 solves the reduced saddle-point problem over the pair
(`V_N`, `W_R⁺`) built independently — POD for the primal, CPG for the dual — with
**no inf-sup stabilization**, and this paper supplies none. The sequel is the
paper that closes that gap: its §2.3 enriches the primal space with supremizers
(its Eq. 15–17), and its Algorithm 1 (PGA) builds that enrichment offline
(its Eq. 27–29).

`rb_online.solve_reduced` therefore takes an optional `enrichment`. **Omitting it
reproduces Algorithm 1 line 4 bit for bit**, which is the default so the `[BEE20]`
transcription stays intact and testable; `rb_online.pga_enrichment` supplies the
`[NDEE22]` fix. `run_demo.py` reports both.

**What this does and does not show on the toy — read before quoting a number.**
`toy_problem.py` has `B = I`: the constraint is `u ≤ gap`, with no `µ`-dependence.
So `S_R(µ)` is constant, and PGA's headline benefit — removing an *online*
construction cost — **is vacuous here**. What PGA still does is real but smaller:
at `δ_PGA = 0.9` it compresses `dim S_R = 10` to `dim S_R^red = 3`.

Measured on five unseen `µ`, `N = 8`, `R = 10`:

| primal space | dim | rel. `u` err | rel. `λ` err | max violation |
|---|---|---|---|---|
| `V_N` only (as published) | 8 | 4.10e-02 | 1.62e-01 | 5.25e-02 |
| `V_N + S_R^red`, δ=0.9 | 11 | 4.32e-02 | 1.55e-01 | **2.86e-02** |
| `V_N + S_R` (Eq. 17) | 18 | 4.83e-02 | 1.60e-01 | 5.12e-02 |

Enrichment roughly halves the constraint violation and slightly improves the dual
error, while slightly *worsening* the primal error. That is expected and not a
bug: **`β^dec` = 2.37e-01 > 0 on this toy**, so the pair was already inf-sup
stable and there is no instability to repair. Enlarging the primal space merely
changes the discretization, and Galerkin optimality does not hold for the
constrained problem, so a larger space can be marginally worse. Reduced
multipliers stayed non-negative in every case, which is the property CPG exists
to preserve.

`β^dec > 0` here is the same finding the sequel's repository reports as its
non-reproducing claim C1, and for the same structural reason. See below.

---

## Extraction problems that affected this implementation

The paper was read from the PDF, and extraction was lossy in ways that matter:

1. **`pymupdf4llm` dropped every display equation.** The prose extracted cleanly, so
   an automated quality check passes while the entire mathematical content is
   missing. Equations were recovered with `pdfplumber` instead.
2. **The PDF uses a shifted Type-1 font encoding.** Delimiters are mapped onto
   letters (`p`→`(`, `q`→`)`, `t`→`{`, `u`→`}`, `r`→`[`, `s`→`]`) and operators
   arrive as `(cid:NN)` codes. This was decoded, but two collisions are **not**
   mechanically resolvable:
   - `u` is both the closing brace and the primal displacement variable;
   - `P` is both `∈` and the parameter set `𝒫`.

   Equations quoted in the source comments were therefore read by eye against
   surrounding prose, not trusted from the decoder. **Verify any equation you rely
   on against the PDF.**

---

## UNSPECIFIED items

Each is flagged at its use site in the source.

| # | Item | Where | Choice made | Alternatives |
|---|---|---|---|---|
| 1 | Which norm `‖·‖_Λ` produced the §6 results | `rb_vi_common/cone_projection.norm_Lambda` | L², the "most natural" choice named first in §5 | `ℓ^∞(Γ_c,tr)`. **Not a drop-in swap** — it turns Eq. (57) from an NNLS into a linear program |
| 2 | Tie-breaking in `argmax`, Algorithm 2 line 5 | `rb_vi_common/cone_greedy.cone_projected_greedy` | Lowest training index | Random; largest `‖λ(µ)‖` |
| 3 | NMF stopping rule / iteration count | `nmf_baseline.nmf` | 200 fixed iterations | Relative decrease of Eq. (67); relative change in `W` |
| 4 | NMF initialization | `nmf_baseline.nmf` | Uniform random, seeded | NNDSVD; columns sampled from `T`. §6.4 notes the factorization is non-unique, so seeds give genuinely different `W` |
| 5 | Division guard in Eq. (69) | `nmf_baseline.nmf` | Additive `eps` in denominators | Clipping factors away from zero |
| 6 | POD: centring, and how `N` is chosen | `rb_online.pod_basis` | No centring; `N` passed explicitly | Mean-subtraction; energy-fraction criterion |
| 7 | EIM ranks `M_k`, `M_g` and the greedy that builds them | not implemented | — | §5 declines to detail Task (T1): "Since Task (T₁) can be considered to be standard, we only discuss Task (T₂)". Faithful implementation requires Barrault et al. [3], not this paper |
| 8 | The `V` and `W` inner products for the toy | `rb_online.pga_enrichment` | `G_V = A` (energy), `G_W = I` | `G_V = I`; a mass matrix for `W`. **Neither paper states this** — the toy is in neither. The energy norm is the one for which the sequel's `β_HF` is the natural constant |
| 9 | `δ_PGA` for a 1-D problem | `run_demo.py` | Swept (0.9 / 0.7 / 0.5) | The sequel's §5.1 value is calibrated to its own 2-D discretization and does not transfer |

Item 7 is the reason Algorithm 1 lines 2–3 are absent: the paper deliberately
does not specify them. Items 8–9 arrived with the stabilization wiring and are
`[NDEE22]`-side choices, not `[BEE20]` ones.

---

## Empirical finding that runs against the paper's framing

On the toy problem, at matched cardinality `R = 10`:

```
max cone-projection error, CPG:            5.4539e+01
max cone-projection error, NMF (seed=0):   4.8176e+01
max cone-projection error, NMF (seed=1):   4.2447e+01
max cone-projection error, NMF (seed=2):   4.7606e+01
```

**NMF is more accurate than CPG here**, by roughly 10–20%.

This is not evidence against the paper, and should not be cited as such — the toy
is not their test case. But it is worth stating plainly, and it is consistent with
the structure of the two methods: CPG generators are *selected snapshots*, so the
cone is interpolatory and its generators are constrained to be points of `S_du`;
NMF generators are *optimized atoms* free to sit anywhere in the non-negative
orthant. At equal cardinality the freer parametrization should usually win on
reconstruction error.

§5's case for CPG is accordingly not mainly about accuracy at matched `R`. It is:

- the user supplies a **tolerance** (Eq. 58) rather than a cardinality — §5 calls
  the cardinality-only interface a drawback of NMF, since "it is often difficult to
  anticipate the approximation capacity of the dual RB cone from its cardinality";
- the cones are **hierarchical/nested**, so `R` can be increased incrementally
  without recomputing (verified in `run_demo.py`); NMF must be re-run from scratch
  for each `R`, and is non-deterministic when re-run;
- CPG is **deterministic** given the training set.

The `seed=0/1/2` spread above is itself an illustration of the last point.

---

## Can the Hertz contact model host the sequel's inf-sup experiments?

`src/contact_dataset.py` is a validated Hertzian contact model — it reproduces the
analytic semi-elliptical pressure profile to a median relative shape error of
~0.05. The natural question is whether it is a better test problem for the
sequel's PGA/inf-sup experiments than either repository's 1-D obstacle toy, and
in particular whether it can exhibit the sequel's claim **C1** (`β^dec = 0`),
which its own 1-D toy provably cannot.

**Answer: no, not as it stands.** `src/hertz_infsup_probe.py` runs the evaluation;
the reasons are structural rather than a matter of tuning or resolution.

**1 — The parametrization is of the wrong kind.** Matching the model's QP against
the saddle-point dual form identifies `B A⁻¹ Bᵀ = C`, the influence matrix: the
model supplies only the **Schur complement**, with the primal field eliminated
analytically. The consistent factorization is `B = I`, `A = C⁻¹`, and then all
five parameters (`radius, offset, wav_amp, wav_len, delta`) enter through the gap
— that is, through `g`, the **right-hand side**. `B` is parameter-**independent**.
The sequel is about parameter-dependent *constraints*; with `B` constant,
`S_R(µ)` is constant and PGA has no online cost to remove. This is the same
defect the 1-D toy has.

**2 — A validated pressure profile is not sufficient.** With real Hertz pressure
snapshots and `B = I`, `β^dec > 0` is **proven** (positive Gordan certificate) at
every POD tolerance from 5e-1 to 1e-3. There is a second, sharper reason visible
here: the model has no external load, so `u = -C p` is determined entirely by the
multiplier, and the primal snapshot space is *contained in the supremizer space*
by construction. POD then returns a `V_N` aligned with `S_R` — the most
favourable configuration for inf-sup stability that can exist. `V_N` and `W_R⁺`
are not independent spaces at all, which is what the sequel's §2.3 assumes.

**3 — Curvature is not the missing ingredient.** The sequel's REPRODUCTION_NOTES
conjectures that C1 needs "the richer geometry of a 2-D contact surface". Parts C
and D of the probe test the two cheapest versions of that: a rotating contact
normal on a curved arc (vector-valued displacement, `µ`-dependent collocation),
and contact between **two** bodies of different moduli, where the constraint acts
on the displacement jump `[[u]]` and every row acquires a positive and a negative
block. Across **40 configurations** — half-angles 0.05 to 1.30, `N` swept from 1
to `R−1` so `dim ker(C)` reaches ~21 — **every one returns a positive Gordan
certificate**. Neither mechanism reaches `β^dec = 0`.

There is a consistent trend: the certificate weakens monotonically as `N` falls
and the arc curves, from `t ≈ 1.7` at `N = R−1` down to `t ≈ 5e-2` at `N = 2`,
half-angle 1.30. The obstruction is eroded by exactly the variables one expects;
it simply does not reach zero.

**Two probe defects found and fixed during this evaluation**, recorded because
both would have produced a falsely confident result:

- Primal snapshots generated as `u = A⁻¹(−Bᵀλ)` (i.e. `f = 0`) lie *inside* the
  supremizer space, so POD returns `V_N` aligned with `S_R` and `β^dec > 0` is a
  foregone conclusion. Fixed by applying a genuine `µ`-dependent external load
  and solving the actual contact QP. (For part B this is not a choice but a
  property of the boundary-integral model — see reason 2.)
- Two **identical** bodies constrained on the jump reduce exactly to a one-body
  problem with doubled compliance — the standard Hertz `E*` reduction — so the
  symmetric two-body case tests nothing new. It was reproducing the one-body
  certificates times `√2` exactly. Fixed by giving the bodies different moduli.

**What the Hertz model *is* good for here.** It remains the best dual-snapshot
source either repository has: physically meaningful, non-negative by KKT
construction, with support that moves and splits. Both repositories currently
draw their multipliers from 1-D obstacle toys.

**Where to look next.** The variable *not* yet tested is the dual discretization.
Both papers use P1/P1 or LAC pairings (§6.1 here, §5.1 in the sequel); every
implementation in these two repositories uses **collocation**, which keeps the
constraint rows nearly independent. A P1 dual couples neighbouring multipliers and
is the most plausible remaining source of the cancellation C1 requires.

```bash
cd src && python3 hertz_infsup_probe.py
```

---

## Reproducing the demo

```bash
cd src && python3 run_demo.py
```

Requires `numpy` and `scipy` only (see `requirements.txt`). Runs in a few seconds
on CPU. `cvxopt` — which the authors used for the cone projection — is **not**
required: `scipy.optimize.nnls` solves the same problem for the L² norm.
