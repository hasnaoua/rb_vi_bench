# rb_vi_bench

A benchmark over four merged repositories on reduced-basis model order reduction for
parametrized **variational inequalities** (contact problems), measuring **precision**,
**stability**, **performance**, and **cross-implementation agreement**.

The four source repositories are preserved intact under `repos/`, with full git history.
Nothing in them was edited to make the benchmark work.

```
rb_vi_bench/
├── bench/                  the benchmark harness (this is the new code)
├── repos/
│   ├── rb_vi_shared/               shared algorithm library  [BEE20] + [NDEE22]
│   ├── rb_contact_cpg/             [BEE20] Benaceur/Ern/Ehrlacher
│   ├── stable_model_reduction_vi/  [NDEE22] Niakh/Drouet/Ehrlacher/Ern
│   └── greedy_algos/               CPG / mCPG / ADG, installable as `greedy`
├── tests/                  tests for the harness
└── results/                generated output (gitignored)
```

Figures are grouped **per dataset**, since comparing methods only makes sense within one:

```
results/figures/
├── _overview/precision_all_datasets.png     the only cross-dataset figure
└── <dataset>/
    ├── panel.png                            four metrics in one grid
    ├── metrics/      precision.png  conditioning.png  orthogonality.png  offline_cost.png
    ├── decrement/    vs_cardinality.png  vs_tolerance.png
    └── reconstruction/
        ├── all_methods.png
        └── <method>/  best.png  worst.png
```

## The two papers, and the tag convention

Everything here is tagged with the paper it comes from, and the tags are load-bearing:

* **`[BEE20]`** — A. Benaceur, A. Ern, V. Ehrlacher, *A reduced basis method for
  parametrized variational inequalities applied to contact mechanics*, HAL
  `hal-02081485v2`.
* **`[NDEE22]`** — I. Niakh, G. Drouet, V. Ehrlacher, A. Ern, *Stable model reduction
  for linear variational inequalities with parameter-dependent constraints*,
  ESAIM: M2AN (2022). This is the direct sequel; `[BEE20]` is its reference **[9]**.

**The two papers' numbering collides.** Both have an "Algorithm 1", an "Algorithm 2"
and an "Eq. (57)", denoting different things. In `[BEE20]`, Algorithm 1 is the online
stage and Algorithm 2 is CPG; in `[NDEE22]`, Algorithm 1 is PGA and Algorithm 2 is mCPG.
Every citation in shared code therefore carries a paper tag, and **a bare equation
number in `repos/rb_vi_shared/` is a bug**.

Two caveats travel with every `[BEE20]`-tagged number: it uses **preprint v2**
numbering (a v3 exists and was not consulted), and its equations were recovered from a
PDF with a shifted Type-1 font encoding and read by eye. Neither applies to `[NDEE22]`.

## Why there are five CPG implementations, and why they were not merged

| implementation | family | convention |
|---|---|---|
| `cone_projected_greedy` | `rb_vi_common` | `[BEE20]` Alg. 2: raw generators, **absolute** tolerance Eq. (58) |
| `cpg` | `rb_vi_common` | `[NDEE22]` Rmk 4.3: normalized generators, **relative** tolerance Eq. (13) |
| `mcpg` | `rb_vi_common` | `[NDEE22]` Alg. 2: cone-constrained residual generators |
| `greedy.core.CPG` | `greedy.core` | independent implementation, relative tolerance |
| `greedy.core.mCPG` | `greedy.core` | independent implementation |
| `greedy.core.AngularDefectGreedy` | `greedy.core` | **ADG** — Batch Normalized Angular-Defect Greedy, in neither paper |

Collapsing these would mean silently attributing one paper's conventions to the other.
Both source repositories argue this explicitly in their `REPRODUCTION_NOTES.md`, and the
merge preserves that decision.

The cost is that nothing *guarantees* the duplicates agree. The benefit is that
agreement becomes **measurable** — which is what `bench/metrics/agreement.py` does,
generalizing `repos/rb_vi_shared/tests/test_equivalence.py` to real datasets and to the
second implementation family.

### What actually gets reported

Every table and figure carries **one canonical implementation per algorithm**
(`adapters.DEFAULT_METHODS`) — the transcription from the paper that *introduced* it:

| reported as | implementation |
|---|---|
| CPG | `cone_projected_greedy` — `[BEE20]` Algorithm 2 |
| mCPG | `mcpg` — `[NDEE22]` Algorithm 2 |
| ADG | `AngularDefectGreedy`, normalized (standard) form on `S_norm` |
| NMF | `[BEE20]` §6.4, one seed |
| POD | negative control |

The duplicates stay registered and reachable via `--methods`. They are not redundant —
they are the *input* to the agreement metric, and `agreement.csv` is built from
`CROSS_FAMILY_PAIRS` regardless of the reported set, so the cross-implementation check
keeps running. Having established what it shows (CPG bit-identical across families,
mCPG non-unique but accuracy-equivalent), carrying all eleven through every table only
obscured the algorithm comparison. NMF's other seeds likewise remain available; their
spread is measured by `stability.determinism`.

## Install and run

```bash
python3 -m venv .venv && .venv/bin/pip install -e repos/greedy_algos pytest
```

The two `heavy`-tier 2-D datasets additionally need `cvxopt` (`greedy_algos`' optional
`[qp]` extra), which they import at module scope:

```bash
.venv/bin/pip install cvxopt
```

Run the default grid (fast-tier datasets, all methods, four tolerances and four
cardinalities), then render it:

```bash
.venv/bin/python -m bench.runner --subsample 200 --out results
```

```bash
.venv/bin/python -m bench.report --results results
```

Figures (metric vs cardinality, one line per method). `--split` writes one standalone
PNG per metric under `<out>/<dataset>/`; without it you get a combined four-panel figure
per dataset plus a cross-dataset precision overview:

```bash
.venv/bin/python -m bench.figures --results results --out results/figures --split
```

Marginal decrement `e(n+1) − e(n)` — what the next generator actually buys, all methods
on one axis, against both `R` and `ε`:

```bash
.venv/bin/python -m bench.decrement --cardinality-results results/sweep_dense --tolerance-results results
```

The `R` axis needs a sweep run with **consecutive** `--cardinalities 1 2 3 … N`;
non-consecutive steps are skipped rather than divided through, since `e(R+4) − e(R)` is
four generators' worth. The decrement is taken on the **training** error — the quantity a
greedy monotonically drives down — because the test error plateaus and differencing it
yields mostly round-off. Two consequences are real data, not noise: ADG admits tied
snapshots in **batches**, so an intermediate `R` can add a generator that changes nothing
followed by one that drops the error sharply; and NMF is refitted at every `R`, so its
decrement goes genuinely positive — the non-hierarchy `[BEE20]` §5 objects to.

Reconstruction figures — best and worst represented snapshot per method, per dataset,
showing the profile rather than a scalar. These fit directly and need no grid CSV:

```bash
.venv/bin/python -m bench.reconstruction --R 8 --split --out results/figures
```

Best/worst are ranked by **per-snapshot** relative error `‖θ−Π_K θ‖/‖θ‖`, not by the
shared-denominator column the tables report — under a shared denominator a
small-magnitude snapshot looks well reconstructed merely because it is small, so "worst"
would just pick the largest snapshot. The two differ by 1.5× on `hertz_pressure` and 70×
on `physics`, whose snapshot norms span a factor of 604.

Metric figures read **matched-cardinality rows only**, so feed them a run with a dense
`--cardinalities` grid — the tolerance sweep is what makes the main grid slow, and
fixed-`R` fits skip it:

```bash
.venv/bin/python -m bench.runner --deltas --cardinalities 1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 40 --no-infsup --no-determinism --subsample 200 --out results/sweep
```

Useful flags: `--datasets`, `--methods`, `--deltas`, `--cardinalities`, `--no-infsup`,
`--no-determinism`, and `--subsample N` to cap the training set (it changes the numbers
and is recorded in `results/manifest.json`).

Harness tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

The original entrypoints still work unchanged, from inside their own directories:

```bash
cd repos/stable_model_reduction_vi/src && ../../../.venv/bin/python run_experiments.py
```

## The two comparison modes

These answer different questions and are **not** interchangeable:

* **Matched tolerance** — every method gets the same `delta` and reports the `R` it
  needed. This is the interface `[BEE20]` §5 argues *for*: the user supplies an
  accuracy, not a cardinality. The interesting output is `R`, not the error.
* **Matched cardinality** — every method gets the same `R`. The only mode NMF and POD
  can enter at all, and the one on which `[BEE20]`'s own notes record NMF beating CPG.

Reporting only one of the two is how a cone method can be made to look arbitrarily good
or bad, so the runner always does both.

## The four metric families

**Precision** (`metrics/precision.py`) — cone-projection error `||theta - Pi_K(theta)||`,
split train/test, at both modes; plus a non-negativity check. `Pi_K` is an NNLS solve
and is **not linear** in its argument, so there is no projection matrix to precompute.

**Stability** (`metrics/stability.py`) — Gram condition number and `e_orth` (`[NDEE22]`
Eq. 41, its claims C5/C6); re-run determinism; and the inf-sup family `beta_HF`,
`beta^dec`, `beta^off`, `sigma_S`, `c_S`, with a Gordan certificate for the
`beta^dec = 0` question.

**Performance** (`metrics/performance.py`) — solver-call counts (machine-independent,
and the load-bearing metric), offline wall-clock, `R` at a target tolerance, and online
projection cost per query.

**Agreement** (`metrics/agreement.py`) — do independent transcriptions build the same
cone? Four criteria: same `R`, same selection order, **same cone as a set** (the one
that matters — it is invariant to generator scaling and column order), and generators
equal up to normalization.

## Reading the output honestly

Four things must travel with any number taken from here.

**`beta` values are upper bounds.** Minimizing a quotient over a *cone* is non-convex,
so `inf_sup`'s multi-start projected gradient can only overestimate. It is reliable for
showing `beta` is *small* and merely suggestive when showing it is *large*. What is
exact is the separate `beta = 0` test (a convex QP) and the **Gordan certificate**:
`gordan_t > 0` *proves* `beta^dec > 0`. Trust `gordan_t` on the sign.

**`c_S` is an overestimate**, computed over the cone's span rather than the cone. Since
it enters Proposition 3.1 only through Eq. (23) and Eq. (24), overestimating makes both
conservative — a verified Prop. 3.1 stays verified.

**POD is a negative control, not a competitor.** `[BEE20]` §5 is explicit that POD
cannot build `W_R^+`, because its modes have mixed signs and would destroy
`lambda >= 0`. It is in the grid to supply the unconstrained error floor and to
*demonstrate* the sign violation. Its low error is unattainable by a cone, and its
non-zero `nn_violation` is the point. A run where POD passes the non-negativity check
means the dataset cannot discriminate between methods.

**Skips are recorded, never dropped.** A method absent from a table because it was
never measured must not look like one that scored badly, so every non-running cell
emits a row with a `skip_reason`.

## FEM_SOLS: nodal forces vs pressure

`fem_lambda` is [BEE20] §6.2's half-disk contact problem. The archive ships **no
coordinates and no connectivity**, so the node ordering had to be established from the
values (see `datasets.FEM_LAMBDA_ORDERING`):

* **Already ordered**, index 0 at the symmetry axis running outward — total variation
  7.81 against 84.0 for a random permutation, 91% of active steps strictly decreasing.
* **Node 0 carries a half-support shape function**: `λ₀/λ₁ = 0.5035 ± 0.019` across all
  50 snapshots. So these are nodal **forces** `∫ p φᵢ`, not pressures.
* **Spacing is uniform**: doubling node 0 and fitting the Hertz semi-ellipse against the
  node *index* gives `R² = 0.989`, half-width `a = 13.3` nodes. Index is a faithful
  abscissa; no coordinates are needed.
* **Seriation finds nothing better** — spectral (Fiedler) ordering scored `TV = 20.7`,
  worse by 2.7×, because the 43 near-zero tail nodes dominate the correlation.

`fem_lambda_pressure` is the same data with that weighting undone (`p = λ/hᵢ`, i.e. node
0 doubled up to a global constant). Both are carried because **the correction is not
cosmetic**: cone methods are invariant to rescaling a *snapshot* but not a *coordinate*,
so the two are genuinely different reduction problems. Measured at `δ = 0.02`, CPG and
mCPG select identically on both, while **ADG does not** — consistent with ADG selecting
on angle, which a coordinate rescaling rotates. A test asserts at least one method sees
the difference, so the pair can never silently become redundant.

The residual ±8% mesh grading is **not** corrected; that needs tributary lengths the
archive does not carry.

## What the datasets are, and what they are not

Eight sources, differing in exactly the ways that matter: whether the constraint
operator `B` is parameter-dependent (only `obstacle_ndee22` is), whether primal
snapshots and a stiffness matrix are available for the inf-sup metrics (only
`toy_bee20` and `obstacle_ndee22`), and the `dim` / `n` ratio (`physics` is 7676 × 96).
The two 2-D sources (`membrane_2d`, `hertz_2d`) are `heavy` tier and opt-in, because
each costs one FEM solve per parameter.

**No dataset here reproduces a number in either paper's results section.** The 1-D
toys are deliberate substitutes for `[BEE20]` §6 and `[NDEE22]` §5.1, chosen so the
algorithms can be exercised on CPU without a FEM stack. Each source repository's
`REPRODUCTION_NOTES.md` states what is and is not implemented; read those before
quoting anything.

`Dataset.paper` records **provenance**, not a citation — it says which repository a
source came from, not that it reproduces that paper.

## What the first full run found

8 datasets × 11 methods × (6 tolerances + 4 cardinalities) = 880 cells, 688 run,
144 agreement comparisons, ~25 min. Regenerate with the commands above.

**CPG is implementation-independent; mCPG is not.** Across all 144 comparisons:

| pair | verdict | n |
|---|---|---|
| `cpg_bee20` vs `cpg_greedy` | equivalent | 48/48 |
| `cpg_ndee22` vs `cpg_greedy` | equivalent | 48/48 |
| `mcpg_ndee22` vs `mcpg_greedy` | equivalent / within solver tol | 17 |
| | different cone, same accuracy | 28 |
| | genuinely divergent | 3 |

Both CPG transcriptions are **bit-identical** to the independent `greedy.core`
implementation at every tolerance on every dataset. mCPG's two implementations
progressively diverge as `R` grows — `5.7e-12` at `R=3` up to `2.2e-1` at `R=135` — but
**achieve the same accuracy**, matching to 3–5 significant figures.

The reason is structural. mCPG's generators are *residuals built on earlier generators*,
so a difference in the [UNSPECIFIED] line-9 solve at step `r` propagates into every later
generator and compounds with `R`. CPG has no such accumulation: each generator is an
independently selected snapshot. So mCPG's cone is **non-unique** given the solver
choice, and neither implementation is wrong. Only 3 of 48 comparisons — all at the two
tightest tolerances, where `R` is largest — show a real accuracy difference, the worst
being `fem_lambda` at `delta=0.01` (test error 0.056 vs 0.046).

**`[NDEE22]` claims C5/C6 reproduce** on every dataset where `R > 1`: mCPG's Gram
condition number is lower (by up to 87× on `toy_bee20`) and its `e_orth` is higher.

**ADG's tolerance is a stronger guarantee than CPG's, so `R` at equal `ε` is not a fair
comparison.** ADG stops when *every* snapshot in `S_norm` satisfies
`e_K(x̂) = sin θ_K(x̂) ≤ ε` — a per-snapshot relative bound. CPG and mCPG stop on one
shared absolute threshold `ε·max_q‖θ_q‖`, under which a small-magnitude snapshot can be
declared resolved because a large one set the bar. ADG therefore needs more generators
at the same nominal `ε` (71 vs 52 on `hertz_pressure` at `ε=0.1`) and delivers a
correspondingly better error (0.37959 vs 0.61521 at `ε=0.5`). Read the two modes
together, or compare at matched cardinality.

At **matched cardinality**, where the stopping rules do not apply, ADG leads the cone
methods at low `R` on several datasets — `gaussian_synth` `R=4` (0.2152 vs CPG 0.2574),
`hertz_pressure` `R=16` (0.4435 vs 0.5218), `membrane_2d` `R=4` (0.7575 vs 0.8032) —
while mCPG tends to lead at higher `R`. `adg` and `adg_raw` coincide exactly in this
mode, which is the expected consistency check: normalization changes the stopping rule
only, never selection.

**The two CPG transcriptions cost 3× different for a bit-identical answer.** On
`toy_bee20` (`n=30`, `R=20`) `cone_projected_greedy` issues 1170 NNLS solves against
`cpg`'s 390. Two compounding causes, and the closed forms match the measurements
exactly:

* `[BEE20]` Alg. 2 sweeps **all** `n` snapshots **twice** per iteration — once for the
  line-5 argmax, once for the line-8 `r_n` update: `2nR = 1200`.
* `[NDEE22]` Rmk 4.3 sweeps only the **unselected** ones, **once**:
  `sum_r (n-r) = 390`.

This is a property of how each paper states the algorithm, not of either
implementation, and it is invisible in wall-clock at this problem size.

## Two known findings that predate this benchmark

Both are documented in the source repositories and both reproduce here:

* **`[NDEE22]` claim C1 (`beta^dec ≈ 0`) does not reproduce.** On every test problem in
  the merge, the Gordan certificate stays positive, proving `beta^dec > 0`. The inf runs
  over a *cone*, so a non-trivial kernel is necessary but not sufficient — it must also
  meet the non-negative orthant, and here it never does. Neither 2-D curvature nor a
  two-body displacement jump changes this (40 configurations tested in
  `repos/rb_contact_cpg/src/hertz_infsup_probe.py`). The untested variable is the **dual
  discretization**: every implementation here uses collocation, and a P1 dual couples
  neighbouring multipliers.
* **NMF beats CPG at matched cardinality**, by roughly 10–20%. This is expected, not
  evidence against `[BEE20]`: CPG's generators are *selected snapshots* while NMF's are
  *optimized atoms* free to sit anywhere in the non-negative orthant, so at equal
  cardinality the freer parametrization should win on reconstruction error. `[BEE20]`
  §5's case for CPG is about the tolerance-driven interface, hierarchical/nested cones,
  and determinism — not accuracy at matched `R`.

## One harness pitfall worth knowing

The relative tolerances normalize by the largest snapshot norm **over the training
set** — `rb_vi_common.cpg` computes it from the matrix it is handed, and
`greedy.core.ConeGreedy` does the same. `Dataset.scale` must reproduce exactly that
denominator, because the `[BEE20]` adapter uses it to convert the canonical relative
`delta` into Eq. (58)'s absolute `eps_du`.

Normalizing by the full-set maximum instead inflates `eps_du` whenever the largest-norm
snapshot lands in the test split, which is enough to stop `[BEE20]`'s CPG one generator
early and make two transcriptions of the same algorithm disagree for a reason that has
nothing to do with either paper. `tests/test_bench.py` pins this.

## Layout constraint

`rb_contact_cpg` and `stable_model_reduction_vi` locate the shared library through their
own `_shared_path.py`, which resolves `parents[2] / "rb_vi_shared"` — `src/` → repo →
the repo's parent. Because all four repos sit under a common `repos/` parent, that
resolution keeps working **unmodified**. Do not flatten `repos/` without setting the
`RB_VI_SHARED` environment variable.
