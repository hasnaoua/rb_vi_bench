# Contact-force datasets — Niakh–Drouet–Ehrlacher–Ern test cases

High-fidelity **contact forces** (the dual Lagrange multipliers `λ(µ) ≥ 0`) for the test
cases of *Stable model reduction for linear variational inequalities with
parameter-dependent constraints* (ESAIM:M2AN, 2022) — the dual snapshots that feed
CPG / mCPG.

Four datasets, in rough order of how hard they are for a cone greedy: `hertz_b` and
`hertz_a` (low-rank Hertz reconstructions, where only conditioning separates the
methods), `hertz` (the Sec 5.2 sweep, where mCPG's conditioning win is dramatic), and
`membrane` (the moving-support regime, where every method struggles).

## What "contact force" means here
For each problem the HF solve is the saddle-point system (7): `a(µ;u,v) + b(µ;v,λ) =
f(µ;v)`, `b(µ;u,η) ≤ g(µ;η)`. The multiplier `λ(µ) ≥ 0` **is** the contact force
(obstacle reaction / contact pressure). Each `λ(µ_p)` is a column of `Lambda`; together
they are the dual snapshot family a greedy in the cone `W+` selects from.

## Files
| file | what |
|---|---|
| `membrane_contact_forces.npz` | membrane obstacle (Sec 5.1), `Lambda` 885×125 |
| `hertz_contact_forces.npz`    | Hertz contact (Sec 5.2), `Lambda` 47×81 |
| `hertz_case_a_imposed_displacement.npz` | Hertz case (a), parametric imposed displacement, `snapshots` 31×51 |
| `hertz_case_b_parametric_geometry.npz`  | Hertz case (b), parametric geometry `R₂=µ`, `snapshots` 23×51 |
| `loader.py`     | `load_dataset`, `w_inner`, `wnorm`, `wpod_error` |
| `membrane_hf.py`, `hertz_hf.py` | the HF solvers (re-runnable) |
| `generate_datasets.py` | rebuilds both `.npz` |
| `make_figures.py` | rebuilds the three figures |
| `fig1…png`, `fig2…png`, `fig3…png` | contact-force fields + CPG-readiness |

### `.npz` keys
**membrane**: `Lambda` (n_c×P), `W_gram` (n_c×n_c, the L² mass on `ω̂` = W-inner-product
Gram), `mu_samples` (125×3 = (radius, cx, cy)), `node_coords` (n_c×2).
**hertz**: `Lambda`, `W_gram` (1-D arc mass), `mu_samples` (81 = R₂),
`contact_abscissa` (n_c, x along Γᶜ₁), `gap` (n_c×81, the initial gap g(x;µ)).

**hertz_a / hertz_b**: a *different schema* — `snapshots` **row-wise** (P×n_c, not
transposed), `params` (P,), `abscissas` (51, uniform over `[-1,1]`), `description`.
They carry **no `W_gram`**: their dual space is the same 1-D contact arc, so the loader
rebuilds `W` as the P1 arc mass matrix over `abscissas`
(`greedy.datasets.contact_forces.arc_mass_matrix`) — the identical construction
`hertz_hf.arc_mass_1d` used to build the `hertz` file's own `W_gram`. This is the one
assumption added when integrating them; without it the W-norm would silently degrade to a
mesh-dependent nodal norm. `CASES[...]["gram_source"] = "arc_mass"` marks it.

```python
from greedy.datasets.contact_forces import load_contact_force_dataset

ds = load_contact_force_dataset("membrane")   # or "hertz", "hertz_a", "hertz_b"
ds.snapshots    # (P, n_c) — row p is lambda(mu_p) >= 0, transposed for the greedies
ds.gram         # (n_c, n_c) — <a,b>_W = a.T @ gram @ b
ds.parameters   # scalar parameter per snapshot, for plots/splits
```

The `.npz` files live in `data/contact_forces/`; the HF solvers and the original
standalone `loader.py` moved to `greedy/synthetic_data/contact_forces/`.

## The two problems
**Membrane obstacle (5.1)** — `-Δu = -1` on `Ω=(-½,½)²`, `u=0` on `∂Ω`, `u ≥ ψ(µ)` on a
fixed reference disk `ω̂`, with `ψ(µ) = -1.25·((x-µ₂)²+(y-µ₃)²)/µ₁²`. `µ=(µ₁,µ₂,µ₃)`
over the 5×5×5 = 125 grid. `λ(µ)` is the obstacle reaction; its support is a central
disk that **shifts with (µ₂,µ₃) and resizes with µ₁**.

**Hertz contact (5.2)** — two quarter-disks (body 1 radius `R₁=1` fixed, body 2 radius
`R₂=µ`), plane-strain elasticity `E=15, ν=0.35`, initial gap `γ₀=1e-3`, imposed edge
displacement `d=0.09`. `µ=R₂` over 81 points in `[0.7,1.3]`. `λ(µ)` is the contact
pressure on body 1's arc — a Hertzian semi-ellipse that widens with `µ`.

**Hertz cases (a) and (b)** — two further reconstructions of the same Hertz physics on a
shared reference arc of 51 uniform nodes over `[-1,1]` (see
`docs/figures/hertz_cases_reconstruction_check.png`, cf. the paper's Fig. 4 / Fig. 6).
They separate the two ways `λ` can be driven:
* **(a) parametric imposed displacement** — `µ = d` over 31 points in `[0.15,0.45]`
  (exactly the `fem_sols_training_parameters()` grid, `0.15 + 0.01i`). Pressing harder
  both *scales up* the semi-ellipse and *widens* its support, so this is the amplitude
  ramp the other cases lack. Numerical rank 14, W-norm spread 3.9×.
* **(b) parametric geometry** — `µ = R₂` over 23 points in `[0.905,1.125]`. Same
  parametrization as `hertz`, but a much narrower range around `R₂≈1`, so the profile
  barely moves: numerical rank **8**, W-norm spread only 1.08×.

## How these were built, and where they differ from the paper
These are **faithful reconstructions**, not byte-exact copies of the paper's data. The
originals come from FreeFem++ meshes that aren't published, so the goal was a
self-consistent HF solve of the *same* physics with the structure CPG/mCPG needs.

- **Fixed dual space (identical DOF ordering across µ).** Membrane: the constraint is
  imposed on a fixed reference disk `ω̂`, so `dim(W)` is constant and only `ψ(µ)`/the gap
  vary — the "parameter-dependent constraint" is carried by `g(µ)`. Hertz: `λ` lives on
  body 1's arc, which is parameter-independent, so `W` is naturally fixed while the
  constraint operator and gap change with `R₂`. This is the invariant that lets you
  stack the `λ(µ_p)` into one matrix.
- **L²/mass-consistent multipliers.** The discrete cone condition is `[M_c(u_c-ψ_c)]_i ≥ 0`
  (membrane) / `[M_c(Bu-g)]_i ≤ 0` (Hertz), so `λ` is a genuine P1 pressure field whose
  norm is `W_gram` — not a raw nodal reaction. (Membrane peak `λ≈8–19`, matching Fig 4's
  ~14; Hertz profile fits `p₀√(1-(x/a)²)` to ~11% L² misfit.)
- **Hertz uses the paper's own small-deformation linearization** (constant normal
  `n=e_y`, Sec 5.2), turning curved–curved contact into a vertical gap constraint — no
  closest-point search.
- **Recovery via the condensed dual QP** (`cvxopt`, as in the paper): `min ½λᵀAλ − bᵀλ,
  λ≥0` with `A = C K⁻¹ Cᵀ`. `λ` comes out directly as the contact force.
- **Dual dimensions differ** from the paper's meshes (here 885 vs 467, and 47 vs 280),
  and the Hertz `λ` is P1 over the potential-contact zone rather than P0 over the whole
  manifold. Structure and behaviour match; exact counts don't.

## Using with CPG / mCPG
Feed `Lambda` as the dual snapshot matrix and `W_gram` as the inner-product Gram used in
cone projections and the greedy error. `fig3` / `wpod_error` give the **POD baseline**:
Hertz is very low-rank (e(R=6)≈2.5e-2), while the membrane decays slowly
(e(R=20)≈0.25) — the moving-support regime where cone-aware greedy is expected to help
over plain POD, and a good stress test for the mCPG aperture-widening / initialization
behaviour. Regenerate anytime with
`python -m greedy.synthetic_data.contact_forces.generate_datasets`.

## Running the greedies

```bash
python -m greedy.pipelines.contact_forces_compare --case hertz    --epsilon 0.01
python -m greedy.pipelines.contact_forces_compare --case membrane --epsilon 0.25

# compare ADG with and without normalization -> adg_variants.csv
python -m greedy.pipelines.contact_forces_compare --case hertz --epsilon 0.01 --adg-study

# the two reconstructed Hertz cases, swept over epsilon (both are low-rank, so
# epsilon=1e-3 is where the comparison is actually discriminating)
for case in hertz_a hertz_b; do
  for eps in 0.05 0.01 0.001; do
    python -m greedy.pipelines.contact_forces_compare --case $case --epsilon $eps \
      --adg-study --matched-r --output-dir results/contact_forces/${case}_eps${eps}
  done
done
```

ADG normalizes by default (standard ADG); pass `--no-adg-normalize` for the
global-scale stopping criterion.

Writes `comparison_metrics.csv`, `convergence_vs_pod.png`,
`residuals_by_parameter.png`, `generators.png` and the physical-space bases to
`results/contact_forces/<case>/`.

### How the W-inner product is honoured
The greedies in `greedy.core` are written against the Euclidean norm. Rather than
duplicate them, the pipeline factors `W = Uᵀ U` and runs them on `U`-transformed
snapshots: `‖x‖_W = ‖U x‖₂`, and a cone combination transforms as
`U(A c) = (U A) c` with the same `c ≥ 0`. So the plain algorithms operating on
transformed data *are* the W-norm greedies, with identical selections.

The one place this breaks is mCPG. Its cone-shift step constrains
`Υ ≤ θ` **elementwise** — that is what keeps `θ − Υ` inside `W⁺` — and `U` does
not preserve elementwise-nonnegativity. Left alone it would silently enforce
`U(θ − Υ) ≥ 0`, a different and meaningless feasible set. mCPG therefore takes a
`constraint_transform=inv(U)` and imposes that bound back in physical
coordinates. `tests/test_contact_forces.py` pins both halves of this.

### What the two cases show
Both splits are interpolation: Hertz holds out interior `R₂` values; the
membrane holds out the strict interior of its 5×5×5 grid (98 train / 27 test),
since splitting on `mu_1` alone is degenerate — it has only 5 distinct values.

* **Hertz** (ε=1e-2) is where mCPG earns its keep: it reaches the tolerance at
  **R=25 with κ≈9.8e2**, against **R=38 with κ≈5.5e9** for CPG and ADG — seven
  orders of magnitude better conditioning, matching the paper's Fig. 6 claim.
  CPG and ADG select the *same set* here (in a different order), so they produce
  the identical cone and identical errors.
* **Hertz cases (a) and (b)** are the *easy* end of the spectrum, and the honest
  summary is that they do not discriminate between methods on accuracy. Both are
  low-rank enough that at ε=1e-2 every method reaches the tolerance at R=7 (a) /
  R=3 (b) with **identical errors**; POD hits 1.8e-6 by R=4 on (b). Conditioning
  is the only axis that separates them — and mCPG wins it by 1–2 orders at every
  ε tested (see the matched-R table below). Use them as a conditioning benchmark
  and a regression fixture, not as evidence about accuracy.
* **Membrane** is the hard moving-support regime, and worth being blunt about:
  POD needs R=97/98 to resolve the training set at all (e(R=20)≈0.25), so any
  ε below ~0.17 is unreachable and all three greedies just memorize the training
  set (R=P). Even a cone built from *all* 98 training snapshots leaves a max
  held-out error of ≈0.35, versus ≈0.12 for an unconstrained POD subspace of the
  same data. Nonnegative combinations can only add mass, never cancel it, so a
  bump at an unseen `(cx, cy)` is not reachable from bumps elsewhere. Use ε≈0.25–0.4
  to get a meaningful R < P comparison.

### ADG's normalization on these datasets
Run with `--adg-study`; numbers below are at ε=1e-2 (Hertz) and ε=0.25 (membrane).
`normalize_snapshots` is ADG's only knob (standard ADG = normalization on), and on
these datasets it is the wrong setting:

* **`normalize_snapshots` costs R with no accuracy return here.** Hertz R=38→42
  (κ 5.5e9→4.6e11); membrane R=56→82. Held-out error is unchanged to 3 digits in
  both. Expected: it is a *stricter* criterion, and it pays off when snapshot
  norms are spread widely enough for a large one to mask a small one — but the
  W-norm spread is only 1.16× (Hertz) and 1.30× (membrane), so there is nothing
  for it to fix. It should earn its keep on datasets with a real loading ramp
  (e.g. the physics set, ratio 604), not these.
* **Normalization cannot change selection order, only where ADG stops.** The angle
  criterion is invariant to positive per-vector scaling, so the normalized and
  unnormalized runs pick the same generators in the same order; they differ only in
  which snapshots still count as unresolved, hence in R. This is why there is no
  separate "ADG-best" arm in `--matched-r`: a normalized arm would trace exactly
  the same prefix curve.
* The Theorem 3.5 certificate reports **0 violations** for every variant. With every
  round admitting a θ_max maximizer, the bound is guaranteed rather than incidental.

### Matched-R verdict: mCPG wins, and it is the only method that generalizes
Run with `--matched-r` (writes `matched_r.csv` / `matched_r.png`). Because all
three greedies are nested, prefix K_1..K_R is exactly the cone each would give at
a looser ε, so comparing at equal R separates "better cone" from "stopped sooner".

Hertz at **R=16** (the largest budget every arm reaches):

| method | κ | train max | test max |
|---|---|---|---|
| CPG | 4.17e2 | 3.495e-2 | 6.280e-2 |
| **mCPG** | **1.08e2** | **2.940e-2** | **6.035e-2** |
| ADG | 4.17e2 | 3.495e-2 | 6.280e-2 |

mCPG dominates on every axis. CPG and ADG are *identical at every R* (same
selections, different order).

**The held-out error floors.** On Hertz, CPG/ADG test error is pinned at 6.280e-2
from R=12 all the way to R=38 — the last 26 generators improve training error 4×
and held-out error by nothing. mCPG is the only arm that keeps improving out of
sample (6.10e-2 @R=12 → 6.04e-2 @R=16 → 4.82e-2 @R=25). The useful budget here is
R≈12; everything past it is training-set decoration.

The membrane says the same thing at R=52: mCPG best (κ=3.07e1, train 2.710e-1,
test 3.673e-1); ADG κ=4.25e1, test 3.686e-1; CPG worst on test (3.786e-1).

**hertz_a at ε=1e-3, R=8** — accuracy ties, conditioning does not:

| method | κ | train max | test max |
|---|---|---|---|
| CPG | 1.84e3 | 1.246e-3 | 1.438e-2 |
| **mCPG** | **3.71e1** | **9.491e-4** | 1.438e-2 |
| ADG | 1.84e3 | 1.246e-3 | 1.438e-2 |

**hertz_b at ε=1e-3, R=2** — the same shape, even more extreme: CPG/ADG both
κ=2.20e1, mCPG **κ=1.29** (a near-orthogonal cone), every arm at train
1.285e-2 / test 1.120e-2.

Two things these two cases add to the picture:

* **CPG ≡ ADG is not universal.** On `hertz` they select the same set at every R,
  and on `hertz_a` at ε=1e-3 they do too — both take `{0,3,6,10,15,20,25,30}`,
  differing only in order, hence identical κ and errors. But at **ε=0.05 on
  hertz_a they genuinely diverge, and ADG is strictly worse**: CPG takes
  `{0,10,20,30}` (κ=3.16e1, train 7.357e-2, test 3.931e-2) while ADG takes
  `{0,15,20,30}` (κ=3.76e1, train 1.029e-1, test 8.264e-2) — a 2.1× worse
  held-out error at the same R=4. The angular criterion prefers node 15 over node
  10, and that is the wrong call here. So "CPG and ADG are the same algorithm in
  disguise" is an artifact of the earlier datasets, not a general fact: it breaks
  as soon as ε is loose enough for the choice to bite.
* **The held-out floor reappears.** hertz_a's test error stops improving at R=7
  (1.438e-2 at both R=7 and R=8 on the prefix curve, for CPG and mCPG alike)
  while training error keeps falling — and every ADG variant that runs longer,
  out to R=11, reports the same 1.438e-2. As on `hertz`, extra generators past
  that point buy nothing out of sample, which points at the ordered-interior
  split rather than at any one greedy.
