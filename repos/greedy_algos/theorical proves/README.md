# Theorical proves — Lean 4 formalization of ADG theory

Machine-checked proofs (Lean 4 + Mathlib, pinned to `v4.32.0`) about the
Angular-Defect Greedy (ADG) reduced-cone method from `../src/greedy`.

## Current module

- [`Theorical proves/ADGRate.lean`](Theorical%20proves/ADGRate.lean) —
  **convergence speed / rate of normalized ADG**. Sorry-free. Contents:
  - *Part I* — error-sequence rate engine: geometric decay `e n ≤ qⁿ·e 0`,
    convergence `e n → 0`, and a finite iteration count to any tolerance, under
    a uniform angular-defect contraction `e (n+1) ≤ q·e n` (`0 ≤ q < 1`); plus a
    polynomial-envelope convergence lemma.
  - *Part II* — the worst-case cone error `worstError X K = maxₓ infDist x K`:
    monotone under cone growth, zero exactly when the cone spans the data.
  - *Part III* — an ADG `Run` (nested positive cones) whose worst-case error is
    antitone and inherits the Part I rate/convergence/tolerance theorems, and
    terminates once the cone contains every snapshot.
  - *Part IV* — the normalized certificate: for unit-norm snapshots the residual
    equals `sin` of the cone angle, so the normalized worst-case error is
    `maxₓ sin θ_K(x)` (the "Theorem 3.5" left-hand side).
  - *Part V* — research roadmap: deriving the contraction factor `q < 1` from
    data geometry (fill distance / weak-greedy comparison to cone widths) and
    formalizing scale-invariance of the selection.

- [`Theorical proves/ADGTermination.lean`](Theorical%20proves/ADGTermination.lean)
  — **unconditional** results a concrete run gives (sorry-free):
  - `FiniteRun.err_eq_zero_of_card_le` — finite exact termination: with a greedy
    run that keeps picking a not-yet-generator while the error is positive, the
    worst-case error is `0` for every `n ≥ |X|` (recovery in ≤ `|X|` rounds).
  - `FiniteRun.err_le_of_cover` — covering / δ-density bound: once the generators
    δ-cover the snapshots, the worst-case error is `≤ δ`.
  - `FiniteRun.tendsto_zero` — convergence, from eventual exactness.

  Note: a *per-step* geometric contraction `err(n+1) ≤ q·err n` (`q<1`) is **not**
  implied by density — greedy error can plateau then drop — so `Run.geometric_rate`
  stays conditional; the unconditional facts are the two above.

## Build

```bash
lake exe cache get   # first time: pull prebuilt Mathlib oleans
lake build
```
