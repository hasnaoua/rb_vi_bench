/-
# Normalized Angular-Defect Greedy (ADG): convergence-rate theory

This file formalizes the **speed / rate of convergence** of the *normalized*
Angular-Defect Greedy algorithm implemented in
`src/greedy/core/angle_defect_greedy.py`.

## What ADG does (recap of the reference implementation)

ADG grows a *positive cone* `K = span_+ {g_1, …, g_R}` one round at a time and
projects every training snapshot `x` onto the current cone by non-negative
least squares (NNLS), `Π_K(x) = argmin_{c ≥ 0} ‖A c − x‖₂`.  Each round it
admits **every** snapshot attaining the largest *angle*
`θ_K(x) = arcsin(‖x − Π_K(x)‖₂ / ‖x‖₂)` to the cone.  In the *normalized*
variant (`normalize_snapshots = True`) every snapshot is first divided by its
own norm, so the stopping test becomes a genuine per-snapshot **relative**
error check, while — as the reference code proves — the selection is unchanged
because the angle criterion is invariant to positive per-vector scaling.

## What this file proves

The rate analysis is split into a reusable *engine* and a *concrete model*.

* **Part I — Error-sequence engine.**  Purely on a real sequence `e : ℕ → ℝ`:
  a uniform *angular-defect contraction* `e (n+1) ≤ q · e n` with `0 ≤ q < 1`
  forces geometric decay `e n ≤ qⁿ · e 0`, convergence `e n → 0`, and a finite
  iteration count to reach any tolerance `ε > 0`.

* **Part II — The cone worst-case error.**  Modelling the NNLS residual by the
  best-approximation distance `infDist x K`, the worst-case error
  `worstError X K = max_{x ∈ X} infDist x K` is **monotone** under cone growth
  and vanishes exactly when the cone already contains every snapshot.

* **Part III — An ADG run and its rate.**  Packaging a nested family of cones
  as a `Run`, the worst-case error `err n` is antitone, obeys the geometric
  rate / convergence / tolerance-count theorems under the angular-defect
  contraction hypothesis, and hits `0` once the cone spans the data.

* **Part IV — The normalized certificate.**  For unit-norm snapshots the cone
  residual equals `sin` of the cone angle, so the normalized worst-case error is
  exactly `max_x sin θ_K(x)` — the quantity the reference "Theorem 3.5"
  certificate bounds.

The one deep ingredient deliberately left as a hypothesis is the contraction
factor `q < 1` itself: deriving it from the data geometry (a fill-distance /
covering-number bound, or a weak-greedy comparison to Kolmogorov cone widths)
is the research frontier flagged in `Part V`.
-/
import Mathlib

namespace ADG

open Filter Metric
open scoped Topology

/-! ## Part I. Abstract greedy error sequences (the rate engine)

Everything here is stated for a bare real sequence `e : ℕ → ℝ` that is
nonnegative and contracts by a fixed factor each step.  These are the rate
theorems; Parts II–III show the ADG worst-case error is such a sequence. -/

section ErrorSeq

variable {e : ℕ → ℝ} {q : ℝ}

/-- **Geometric rate.**  A nonnegative sequence contracting by a fixed factor
`q ≥ 0` each step is bounded by `qⁿ · e 0`.  For ADG, `q` is the uniform
angular-defect factor: each enrichment round shrinks the worst-case error by at
least `q`. -/
theorem geom_rate (hq : 0 ≤ q)
    (hstep : ∀ n, e (n + 1) ≤ q * e n) : ∀ n, e n ≤ q ^ n * e 0 := by
  intro n
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      e (n + 1) ≤ q * e n := hstep n
      _ ≤ q * (q ^ n * e 0) := by exact mul_le_mul_of_nonneg_left ih hq
      _ = q ^ (n + 1) * e 0 := by ring

/-- **Convergence.**  Under a strict contraction `0 ≤ q < 1` the error tends to
`0`: the algorithm converges. -/
theorem geom_tendsto (hnn : ∀ n, 0 ≤ e n) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hstep : ∀ n, e (n + 1) ≤ q * e n) : Tendsto e atTop (𝓝 0) := by
  have hb : ∀ n, e n ≤ q ^ n * e 0 := geom_rate hq0 hstep
  have hpow : Tendsto (fun n => q ^ n * e 0) atTop (𝓝 0) := by
    have h := tendsto_pow_atTop_nhds_zero_of_lt_one hq0 hq1
    simpa using h.mul_const (e 0)
  exact squeeze_zero hnn hb hpow

/-- **Iteration count.**  Under a strict contraction, for any tolerance `ε > 0`
there is a finite round `N` after which the worst-case error stays `≤ ε`.  This
is the "speed" statement: convergence in finitely many rounds to any accuracy. -/
theorem exists_steps_to_tol (hnn : ∀ n, 0 ≤ e n) (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hstep : ∀ n, e (n + 1) ≤ q * e n) {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∀ n ≥ N, e n ≤ ε := by
  obtain ⟨N, hN⟩ :=
    Metric.tendsto_atTop.1 (geom_tendsto hnn hq0 hq1 hstep) ε hε
  refine ⟨N, fun n hn => ?_⟩
  have h := hN n hn
  rw [Real.dist_eq, sub_zero, abs_of_nonneg (hnn n)] at h
  exact h.le

/-- **Polynomial rate engine.**  A companion to `geom_tendsto`: a nonnegative
error bounded by an algebraically decaying envelope `C · (n+1)^(−α)` (the shape
weak-greedy theory produces when the cone Kolmogorov widths decay
algebraically) also converges to `0`. -/
theorem tendsto_of_le_rpow (hnn : ∀ n, 0 ≤ e n) {C α : ℝ} (hα : 0 < α)
    (hb : ∀ n, e n ≤ C * ((n : ℝ) + 1) ^ (-α)) : Tendsto e atTop (𝓝 0) := by
  have hbase : Tendsto (fun n : ℕ => ((n : ℝ) + 1)) atTop atTop :=
    tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
  have hrpow : Tendsto (fun n : ℕ => ((n : ℝ) + 1) ^ (-α)) atTop (𝓝 0) :=
    (tendsto_rpow_neg_atTop hα).comp hbase
  have hpow : Tendsto (fun n : ℕ => C * ((n : ℝ) + 1) ^ (-α)) atTop (𝓝 0) := by
    simpa using hrpow.const_mul C
  exact squeeze_zero hnn hb hpow

end ErrorSeq

/-! ## Part II. The ADG worst-case error over a positive cone

The NNLS cone projection residual `‖x − Π_K(x)‖₂` is the distance from `x` to
the cone `K`; we model it by `Metric.infDist`, which is defined for every set
and — crucially — is monotone under set inclusion, exactly the "a bigger cone
can only help" property that drives greedy convergence. -/

section Cone

variable {E : Type*} [NormedAddCommGroup E]

/-- Residual of a snapshot `x` to a cone `K`, i.e. the best-approximation
distance `‖x − Π_K(x)‖₂` realized by the NNLS projection. -/
noncomputable def coneResidual (x : E) (K : Set E) : ℝ := Metric.infDist x K

lemma coneResidual_nonneg (x : E) (K : Set E) : 0 ≤ coneResidual x K :=
  Metric.infDist_nonneg

/-- Growing the cone never increases any snapshot's residual. -/
lemma coneResidual_antitone {K K' : Set E} (hK : K.Nonempty) (h : K ⊆ K')
    (x : E) : coneResidual x K' ≤ coneResidual x K :=
  Metric.infDist_le_infDist_of_subset h hK

/-- Residual to a cone containing `0` is at most `‖x‖`, so the relative residual
`coneResidual x K / ‖x‖` lives in `[0, 1]`. -/
lemma coneResidual_le_norm {x : E} {K : Set E} (h0 : (0 : E) ∈ K) :
    coneResidual x K ≤ ‖x‖ := by
  have h := Metric.infDist_le_dist_of_mem (x := x) h0
  simpa [coneResidual, dist_zero_right] using h

/-- Worst-case error of a cone over a nonempty finite snapshot set:
`max_{x ∈ X} ‖x − Π_K(x)‖₂`. -/
noncomputable def worstError (X : Finset E) (hX : X.Nonempty) (K : Set E) : ℝ :=
  X.sup' hX (fun x => coneResidual x K)

lemma worstError_nonneg (X : Finset E) (hX : X.Nonempty) (K : Set E) :
    0 ≤ worstError X hX K := by
  obtain ⟨x, hx⟩ := hX
  simp only [worstError]
  exact le_trans (coneResidual_nonneg x K) (Finset.le_sup' (fun x => coneResidual x K) hx)

/-- **Monotonicity of the worst-case error.**  A larger cone has a smaller (or
equal) worst-case error — the structural engine of greedy convergence. -/
lemma worstError_antitone {X : Finset E} (hX : X.Nonempty) {K K' : Set E}
    (hK : K.Nonempty) (h : K ⊆ K') : worstError X hX K' ≤ worstError X hX K := by
  simp only [worstError]
  refine Finset.sup'_le hX _ (fun x hx => ?_)
  exact le_trans (coneResidual_antitone hK h x) (Finset.le_sup' (fun x => coneResidual x K) hx)

/-- **Exact recovery.**  Once the cone contains every snapshot, the worst-case
error is exactly `0`. -/
lemma worstError_eq_zero_of_subset {X : Finset E} (hX : X.Nonempty) {K : Set E}
    (h : ∀ x ∈ X, x ∈ K) : worstError X hX K = 0 := by
  refine le_antisymm ?_ (worstError_nonneg X hX K)
  refine Finset.sup'_le hX _ (fun x hx => ?_)
  simp [coneResidual, Metric.infDist_zero_of_mem (h x hx)]

end Cone

/-! ## Part III. An ADG run and its convergence rate

A `Run` records the data (a nonempty finite snapshot set) and the nested family
of cones produced round by round.  `zero_mem` says each cone is a genuine
positive cone (it contains the origin), and `nested` records that ADG only ever
adds generators.  Its worst-case error `err` then inherits every rate theorem
from Part I. -/

/-- An ADG run: nested positive cones over a fixed finite snapshot set. -/
structure Run (E : Type*) [NormedAddCommGroup E] where
  /-- Training snapshots. -/
  X : Finset E
  /-- The snapshot set is nonempty. -/
  hX : X.Nonempty
  /-- The cone after round `n`. -/
  K : ℕ → Set E
  /-- Every cone contains the origin (it is a positive cone). -/
  zero_mem : ∀ n, (0 : E) ∈ K n
  /-- Cones only grow: ADG adds generators, never removes them. -/
  nested : ∀ n, K n ⊆ K (n + 1)

namespace Run

variable {E : Type*} [NormedAddCommGroup E] (R : Run E)

/-- The worst-case (per-snapshot) residual after round `n`. -/
noncomputable def err (n : ℕ) : ℝ := worstError R.X R.hX (R.K n)

lemma err_nonneg (n : ℕ) : 0 ≤ R.err n := worstError_nonneg R.X R.hX (R.K n)

lemma K_nonempty (n : ℕ) : (R.K n).Nonempty := ⟨0, R.zero_mem n⟩

lemma err_step_le (n : ℕ) : R.err (n + 1) ≤ R.err n :=
  worstError_antitone R.hX (R.K_nonempty n) (R.nested n)

/-- The worst-case error is antitone in the round index. -/
lemma err_antitone : Antitone R.err :=
  antitone_nat_of_succ_le R.err_step_le

/-- **Geometric convergence rate of ADG.**  Under a uniform angular-defect
contraction — each round shrinks the worst-case error by at least a factor
`q < 1` — the error decays geometrically, `err n ≤ qⁿ · err 0`. -/
theorem geometric_rate {q : ℝ} (hq0 : 0 ≤ q)
    (hdefect : ∀ n, R.err (n + 1) ≤ q * R.err n) :
    ∀ n, R.err n ≤ q ^ n * R.err 0 :=
  geom_rate hq0 hdefect

/-- **Convergence of ADG.**  Under a strict angular-defect contraction the
worst-case error tends to `0`. -/
theorem tendsto_zero {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hdefect : ∀ n, R.err (n + 1) ≤ q * R.err n) :
    Tendsto R.err atTop (𝓝 0) :=
  geom_tendsto R.err_nonneg hq0 hq1 hdefect

/-- **Finite iteration count.**  Under a strict angular-defect contraction, ADG
reaches any tolerance `ε > 0` after finitely many rounds. -/
theorem steps_to_tol {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hdefect : ∀ n, R.err (n + 1) ≤ q * R.err n) {ε : ℝ} (hε : 0 < ε) :
    ∃ N, ∀ n ≥ N, R.err n ≤ ε :=
  exists_steps_to_tol R.err_nonneg hq0 hq1 hdefect hε

/-- **Finite termination.**  Once round `N`'s cone contains every snapshot, the
worst-case error is `0` from then on. -/
theorem terminates {N : ℕ} (h : ∀ x ∈ R.X, x ∈ R.K N) :
    ∀ n ≥ N, R.err n = 0 := by
  intro n hn
  have hz : R.err N = 0 := worstError_eq_zero_of_subset R.hX h
  have hle : R.err n ≤ 0 := by rw [← hz]; exact R.err_antitone hn
  exact le_antisymm hle (R.err_nonneg n)

end Run

/-! ## Part IV. The normalized certificate: residual = sin(angle)

The reference code selects with the *angle* `θ_K(x) = arcsin(residual/‖x‖)` and,
in the normalized variant, measures a per-snapshot relative error.  For unit-norm
snapshots the cone residual is exactly `sin θ_K(x)`, so the normalized
worst-case error equals `max_x sin θ_K(x)` — the quantity bounded by the
angular-defect ("Theorem 3.5") certificate `err_p ≤ sin θ_p^max`. -/

section Angle

-- In applications `E` is a Euclidean / inner-product space, but the results
-- below need only its norm, so we state them over a `NormedAddCommGroup`.
variable {E : Type*} [NormedAddCommGroup E]

/-- The ADG selection angle `θ_K(x) = arcsin(‖x − Π_K(x)‖ / ‖x‖)`. -/
noncomputable def coneAngle (x : E) (K : Set E) : ℝ :=
  Real.arcsin (coneResidual x K / ‖x‖)

/-- For a unit-norm snapshot, the cone residual is `sin` of the cone angle. -/
theorem residual_eq_sin_coneAngle {x : E} {K : Set E} (h0 : (0 : E) ∈ K)
    (hx : ‖x‖ = 1) : coneResidual x K = Real.sin (coneAngle x K) := by
  have hle : coneResidual x K ≤ 1 := by
    have h := coneResidual_le_norm (x := x) (K := K) h0
    rwa [hx] at h
  have hge : (0 : ℝ) ≤ coneResidual x K := coneResidual_nonneg x K
  unfold coneAngle
  rw [hx, div_one, Real.sin_arcsin (by linarith) hle]

/-- **Normalized worst-case error = `max_x sin θ_K(x)`.**  On unit-norm data the
worst-case residual the algorithm tracks is literally the largest `sin` of a
selection angle, the left-hand side of the angular-defect certificate. -/
theorem worstError_eq_sup_sin {X : Finset E} (hX : X.Nonempty) {K : Set E}
    (h0 : (0 : E) ∈ K) (hnorm : ∀ x ∈ X, ‖x‖ = 1) :
    worstError X hX K = X.sup' hX (fun x => Real.sin (coneAngle x K)) := by
  refine Finset.sup'_congr hX rfl (fun x hx => ?_)
  exact residual_eq_sin_coneAngle h0 (hnorm x hx)

end Angle

/-! ## Part V. Research roadmap (next milestones, not yet formalized)

The results above reduce the **rate** of normalized ADG to the single
hypothesis `hdefect : err (n+1) ≤ q · err n` with `q < 1`.  Establishing that
hypothesis from the data geometry — rather than assuming it — is the frontier:

1. **Uniform angular defect from a fill-distance bound.**  Show that when the
   normalized snapshots are `δ`-dense on the unit sphere of their span, each ADG
   round attains an angle bounded below, giving an explicit `q = q(δ) < 1` and
   hence, via `Run.geometric_rate`, an explicit exponential rate.

2. **Weak-greedy comparison.**  Relate `err n` to the Kolmogorov *cone widths*
   of the snapshot family and transport the DeVore–Petrova–Wojtaszczyk
   weak-greedy rates (algebraic and sub-exponential); `tendsto_of_le_rpow` is
   the endpoint such a comparison would feed.

3. **Scale-invariance of selection.**  Formalize that the angle criterion and
   the selected index set are invariant under positive per-snapshot scaling,
   which is what makes "normalized ADG selects identically to raw ADG" — the
   claim the reference implementation relies on — a theorem rather than a
   convention. -/

end ADG
