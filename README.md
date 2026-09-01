# rb_vi_bench

A benchmark over four merged repositories on reduced-basis model order reduction for
parametrized **variational inequalities** (contact problems), measuring **precision**,
**stability**, **performance**, and **cross-implementation agreement**.

The four source repositories are preserved under `repos/`, with full git history. **No
source file in them was edited to make the benchmark work** — the harness absorbs their
convention differences in `bench/adapters/` instead, and the duplicate CPG/mCPG
transcriptions are retained on purpose because they are the *input* to the agreement
metric. One directory was relocated, not edited: `greedy_algos/theorical proves/` → the
top-level `proofs/`, for the reason given below.

```
rb_vi_bench/
├── bench/                  the benchmark harness (this is the new code)
│   ├── __main__.py           `python -m bench <command>` — dispatches the five below
│   ├── runner.py             the grid: methods × datasets × modes → grid.csv
│   ├── report.py             grid.csv → readable tables
│   ├── figures.py            metric-vs-cardinality figures
│   ├── reconstruction.py     best/worst reconstructed snapshot per method
│   ├── decrement.py          what the next generator actually buys
│   ├── datasets.py           the dataset registry; types.py  the two data contracts
│   ├── adapters/             one Method per algorithm, per implementation family
│   ├── metrics/              precision, cone geometry, stability, performance, online
│   ├── plotting.py           the method palette and how a figure is saved
│   ├── cli.py                the argument fragments the five commands share
│   ├── geometry.py           how a snapshot lays out in space (figures only)
│   └── layout.py, tabular.py, instrument.py, _paths.py
├── repos/                  the four source repositories — see repos/README.md
│   ├── rb_vi_shared/               shared algorithm library  [BEE20] + [NDEE22]
│   ├── rb_contact_cpg/             [BEE20] Benaceur/Ern/Ehrlacher
│   ├── stable_model_reduction_vi/  [NDEE22] Niakh/Drouet/Ehrlacher/Ern
│   └── greedy_algos/               CPG / mCPG / ADG, installable as `greedy`
├── data/                   raw inputs owned by no source repo — see data/README.md
│   └── 3D_cladding_split/          the pellet-cladding archive → the `physics` dataset
├── proofs/                 Lean 4 / Mathlib proofs of the ADG rate and termination
├── tests/                  one module per layer of bench/, plus shared fixtures
└── results/                generated output (gitignored)
```

Three placement rules, each with a reason a reader can check:

* **`repos/` is flat.** `rb_contact_cpg` and `stable_model_reduction_vi` locate the shared
  library through their own `_shared_path.py`, which resolves `parents[2] / "rb_vi_shared"`
  — `src/` → the repo → its parent. All four sitting under one parent is what keeps that
  walk working unmodified, and hence what keeps the original entrypoints runnable. Nesting
  or renaming any of them requires setting `RB_VI_SHARED`.
* **Data sits with whoever reads it.** A dataset a vendored pipeline addresses by a
  relative path stays inside that repo. `data/` holds what only `bench` reads —
  currently the pellet-cladding archive. `repos/README.md` has the full table.
* **`proofs/` is top-level.** The Lean tree is not part of the `greedy` Python package and
  nothing in `bench` or `greedy` references it; it lived under `greedy_algos/` as
  `theorical proves/`, a name with a space in it that no tool enjoys, and it is where the
  8 GB Mathlib build cache accumulates. Its `lakefile.toml` names the *package*, not the
  directory, so `lake build` is unaffected by the move.

Figures are grouped **per dataset**, since comparing methods only makes sense within one:

```
results/figures/
├── _overview/precision_all_datasets.png     the only cross-dataset figure
└── <dataset>/
    ├── panel.png                            four metrics in one grid
    ├── train_vs_test.png                    both sets, one shared y-axis per row
    ├── metrics/      precision.png  precision_train.png  (+ _persnap of each)
    │                 conditioning.png  orthogonality.png  offline_cost.png
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
| orthant `W⁺` | `span₊` of the R most active coordinates — the naive admissible baseline |

**POD is not reported.** It was carried as a negative control — the unattainable
least-squares floor — but it is not a dual basis at all: `[BEE20]` §5 rules it out because
its mixed-sign modes cannot build `W_R^+`. Scoring it beside methods bound by `λ ≥ 0`
compares different problems, and it cost an axis range in every figure without informing
the comparison. It stays registered and reachable via `--methods` for the sign-violation
checks in the test suite.

Its place is taken by the **orthant**, which is the reference that *is* pertinent.
Generators are standard basis vectors `eᵢ`, so non-negativity holds trivially, and at
`R = dim` it is the whole positive orthant `W⁺` — the largest cone any of these methods is
allowed to build. It uses no information about the snapshot manifold beyond which
coordinates carry multiplier mass, so **a cone method that cannot beat it is not earning
its offline cost**. It also anchors the cone-geometry axes from above: `W⁺` contains
`K_full` entirely, so it misses nothing and is maximally too large. Both its modes are
closed-form — projecting a non-negative vector onto `span₊{eᵢ : i ∈ S}` keeps `S` and drops
the rest, so the residual is the norm of the discarded coordinates and no NNLS is needed.

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

`physics` reads its snapshots, its parameters and its train/test split from the
pellet-cladding archive under `data/`, which is raw input and therefore not versioned —
see `data/README.md` for what the eight files are. Unpack it, or point
`RB_VI_CLADDING_SPLIT` at a copy held elsewhere:

```bash
unzip -d data data/3D_cladding_split.zip
```

Two datasets also need inputs that live inside their own source repository and are built,
not downloaded — `fem_lambda` wants `lambda_dataset.npz`, which
`python -m greedy.datasets.lambda_snapshots` produces from `data/FEM_SOLS.zip`. Each
dataset's build raises with the exact command if its input is missing.

The five commands live behind one entry point. `python -m bench` lists them, and each
takes `--help` of its own. The longer form — `python -m bench.runner`,
`python -m bench.figures`, and so on — still works, so saved invocations keep running.

```bash
.venv/bin/python -m bench          # run  report  figures  reconstruct  decrement
```

Run the default grid (fast-tier datasets, all methods, four tolerances and four
cardinalities), then render it:

```bash
.venv/bin/python -m bench run --subsample 200 --out results
```

```bash
.venv/bin/python -m bench report --results results --out results/report.txt
```

Figures (metric vs cardinality, one line per method). `--separate` writes one standalone
PNG per metric under `<out>/<dataset>/`; without it you get a combined four-panel figure
per dataset plus a cross-dataset precision overview:

```bash
.venv/bin/python -m bench figures --results results --out results/figures --separate
```

**Training and test are drawn both ways.** `train_vs_test.png` puts the two sets in
adjacent panels — one row per normalization, shared denominator above and per-snapshot
below — and **forces one y-range onto each row**, so the vertical offset between the
panels *is* the generalization gap rather than something you have to reconstruct from two
sets of tick labels. `--separate` additionally writes the training curves as standalone
panels (`precision_train.png`, `precision_train_persnap.png`) alongside the test ones,
because the training error answers a different question: it is what every greedy actually
minimizes and is monotone in `R` for a nested cone, so it shows whether a method is
converging at all — as opposed to converging *usefully*, which is the test curve's job.
In the combined `panel.png` the training error stays a faint dotted overlay, where its
role is a reference for the test curve rather than a subject.

On `physics` the two rows say opposite things, and both are real. Under the shared
denominator train and test nearly coincide — the archive's split interleaves parameters,
so every test snapshot interpolates between training ones. Per-snapshot the gap is wide:
ADG drives its training error to 0.03 by `R=16` while its test error sits flat at 0.52.
Per-snapshot error is the quantity ADG's tolerance actually bounds, so that is the
convention in which its optimization is visibly not generalizing.

Marginal decrement `e(n+1) − e(n)` — what the next generator actually buys, all methods
on one axis, against both `R` and `ε`:

```bash
.venv/bin/python -m bench decrement --cardinality-results results/sweep_dense --tolerance-results results
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
.venv/bin/python -m bench reconstruct --R 8 --separate --out results/figures
```

Best/worst are ranked by **per-snapshot** relative error `‖θ−Π_K θ‖/‖θ‖`, not by the
shared-denominator column the tables report — under a shared denominator a
small-magnitude snapshot looks well reconstructed merely because it is small, so "worst"
would just pick the largest snapshot. The two differ by 1.5× on `hertz_pressure` and 77×
on `physics`, whose snapshot norms span a factor of 604. Ranking is on the held-out half
wherever a source ships a split, which since the pellet-cladding archive landed includes
`physics` — reconstructing a *training* snapshot is the easy case, and for the
selection-based methods a chosen one lies in the cone exactly.

Metric figures read **matched-cardinality rows only**, so feed them a run with a dense
`--cardinalities` grid — the tolerance sweep is what makes the main grid slow, and
fixed-`R` fits skip it:

```bash
.venv/bin/python -m bench run --deltas --cardinalities $(seq 1 40) \
  --datasets fem_lambda physics membrane_2d \
  --methods cpg_bee20 mcpg_ndee22 adg adg_momentum adg_k0 nmf_s0 orthant \
  --no-infsup --no-determinism --subsample 200 --out results/sweep_dense
```

`--deltas` with no values switches the tolerance sweep off, which is what makes a
40-point cardinality grid affordable. The dataset list is spelled out because this run
includes the two `heavy` sources the default grid leaves out — the decrement figures want
every dataset, and at fixed `R` even the 2-D FEM models are tractable.

Useful flags: `--datasets`, `--methods`, `--deltas`, `--cardinalities`, `--no-infsup`,
`--no-determinism`, and `--subsample N` to cap the training set (it changes the numbers
and is recorded in `results/manifest.json`). `--separate` was called `--split` and still
answers to it; the old name collided with the train/test split every dataset now has,
which made it read as if it selected the evaluation set.

Harness tests — one module per layer of `bench/`, sharing `tests/_fixtures.py`:

```bash
.venv/bin/python -m pytest tests/ -q                    # all 144
.venv/bin/python -m pytest tests/test_datasets.py -q    # just the registry
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

**Cone geometry** (`metrics/cone_geometry.py`) — how the reduced cone `K_R` sits inside
the cone the snapshots actually generate, `K_full = span₊{θ₁…θ_Q}`. Every other metric
scores a cone against the *finite snapshot set*; this scores it against their whole conic
hull, which is the object a dual basis is really meant to represent. Three statistics:
**coverage** (relative residual of random points of `K_full` projected onto `K_R`),
**aperture** (pairwise angles between generators — the cone-level analogue of `e_orth`),
and **reach outside `K_full`**.

### mCPG's cone really is larger — and not only in aperture

CPG and ADG take snapshots as generators, so `K_R ⊆ K_full` by construction. mCPG does
not: its generators are `νᵣ = (θ_q − Υᵣ)/‖·‖`, and a *difference* of two elements of
`span₊{θ}` need not lie in `span₊{θ}`. Line 9's second constraint only guarantees
`νᵣ ≥ 0`, i.e. membership of `W⁺` — which is the property the method needs.

Measured: on `toy_bee20` at R=8, **two of eight** mCPG generators lie outside `span₊{θ}`
(relative distances 0.15 and 0.82); on `fem_lambda`, most do. Confirmed independently of
the NNLS residual by **LP feasibility** (`∃c ≥ 0 : Θc = νᵣ`), infeasible for exactly those
generators. So [NDEE22] Remark 4.3's parenthetical — that `νᵣ` lies "in fact of
`Span⁺({θ_qₙ})`" — does **not** hold as stated. What holds is the `W⁺` membership the same
sentence claims first, and that is what the construction rests on.

It is **data-dependent**: on smooth synthetic bumps every mCPG generator stays inside. So
it is a property of the algorithm–data interaction, which is why it is measured per
dataset rather than asserted once.

Whether the extra reach *helps* is what coverage answers, and on these datasets it barely
does — mCPG's coverage of `K_full` is within a fraction of a percent of CPG's despite a
much wider aperture (51.8° vs 42.0° on `toy_bee20`; 37.2° vs 3.5° on `fem_lambda`, whose
snapshots are nearly collinear). Reaching outside `K_full` only pays if the multipliers
met at run time also lie outside it.

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

> **Note on the findings below and above.** Several are quoted from measurements on
> datasets no longer in the registry — `toy_bee20`, `obstacle_ndee22`, `hertz_pressure`,
> `gaussian_synth`, `fem_lambda_pressure`, `hertz_2d`. The numbers were real when taken
> and are kept because they document *why* the code is shaped as it is, but they cannot
> be reproduced from a current run. Anything measured on `fem_lambda`, `physics` or
> `membrane_2d` still can be.

## What the datasets are, and what they are not

Three sources, all contact problems on real geometry:

| key | reported name | dim × n | tier |
|---|---|---|---|
| `fem_lambda` | Half-disks of Hertz | 57 × 50 | fast |
| `physics` | 3D Pellet-Cladding | 7676 × 94 | fast |
| `membrane_2d` | 2-D membrane | 497 × 125 | heavy (needs `cvxopt`) |

They differ in the way that matters most for a cone method — the `dim`/`n` ratio spans
57 × 50 to 7676 × 94 — and in geometry: a mirrored contact line, a 76 × 101 surface grid,
and a scattered 2-D patch. `membrane_2d` is `heavy` tier and opt-in, because it costs one
FEM solve per parameter.

**None of them carries a stiffness matrix or a constraint operator**, so the inf-sup
metrics ([NDEE22] Eq. 14/18/30) and the solved-error metric (`metrics.online`) report
nothing on any of them. Both remain implemented and generic: any `Dataset` supplied with
`A`, `B_of_mu`, `rhs_of_mu`, `gap_of_mu` and primal snapshots revives them, and their
tests build such a problem themselves rather than depending on the registry. The two
1-D obstacle toys that used to provide them were removed along with the other synthetic
and 1-D sources.

**Every source runs a train/test phase.** `fem_lambda` and
`physics` use the split their own archive ships; the rest get a deterministic stride, so
train and test span the same parameter range rather than the stride being a held-out
tail. Each cell records `n_train` and `n_test` in `grid.csv`, and `manifest.json` states
the partition every dataset ran under. `Dataset` still permits a split-less source — the
figures fall back to training error and label the panel — but no registered dataset is
in that state.

`physics` was the last one that was. It previously read
`greedy_algos/data/physics_data.txt`, which carries the 7676 × 99 matrix and nothing
else: no parameters and no split, so `greedy_algos`' pipeline built and evaluated on the
whole dataset and every `test_*` column came out `nan`. The pellet-cladding archive ships
the same matrix — verified elementwise — together with the 99 parameter values and a
50/49 partition of them. Two consequences worth stating, both pinned by tests:

* **The inferred parameter axis was off by one grid step.**
  `greedy.datasets.physics_dataset` hard-codes a 96-point displacement grid from the
  problem statement and reconciles it with 99 columns by dropping three. The shipped grid
  is 99 points — 0.16–0.30 mm by 0.005, then 0.31–1.00 mm by 0.01 — so the first column
  that builder kept is 0.175 mm, not the 0.18 mm it was labelled, and the error carries
  through to the last, 1.00 mm rather than 1.01. The snapshots were never wrong; their
  abscissa was.
* **The snapshot set did not change, but the training set halved.** The old path
  dropped three leading columns and `Dataset` then dropped two numerically-zero ones; the
  new path keeps all 99 and `Dataset` drops five — the same 94 snapshots, because columns
  0–4 are exactly the no-contact states before the imposed displacement closes the
  0.05 mm gap. Half of those 94 are now held out, so every method fits on 47 columns
  rather than 94 and its `R` at a given tolerance moves with them: at `δ=0.05` ADG's two
  variants now stop at `R=4` and `R=12` where they stopped at 6 and 13. **`physics`
  numbers from before the split are not comparable cell-for-cell.** What is preserved is
  the tolerance itself: the largest-norm snapshot falls in the training half, so
  `Dataset.scale` — the denominator both relative tolerances normalize by — is the same
  number, and a `δ` still means what it meant. And `test_max_rel_err` is now a real
  held-out number instead of `nan`.

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

That tally predates the pellet-cladding split, so its `physics` rows were fitted on all
94 columns and carry no test error. Nothing below is a `physics` number, so nothing below
moves; re-run the wide grid (`--methods` with all eleven) if you want `physics` cells
that match the rest of `results/`.

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
