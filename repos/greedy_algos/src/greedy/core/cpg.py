from __future__ import annotations

import numpy as np

from greedy.core.cone_greedy import ConeGreedy
from greedy.core.reduction_common import vector_projection_angle


class CPG(ConeGreedy):
    """
    Cone-Projected Greedy (CPG) algorithm.

    Builds a reduced positive cone  W^+ = span_+{λ(μ_1), ..., λ(μ_R)}
    such that every training snapshot λ(μ) ∈ S_DU is approximated within the
    relative tolerance ε_DU, using
    max_μ ||λ(μ) - Π_K(λ(μ))||_2 / max_μ ||λ(μ)||_2.

    Each generator is the selected snapshot itself; see ``mCPG`` for the
    variant that stores the cone-shifted, normalized residual instead.

    Shared machinery (cone projection, residual sweep, stopping scale,
    diagnostics) lives in ``ConeGreedy``.

    Parameters
    ----------
    snapshots : array-like, shape (N_train, d)
        The HF solution snapshots {λ(μ)}_{ μ ∈ P^tr }.
    epsilon : float
        Relative tolerance ε_DU >= 0.
    """

    _display_name = "CPG"

    def _init_state(self) -> None:
        # Angle (radians) between each accepted candidate and its projection
        # onto the cone it joined -- how much genuinely new direction it adds.
        # mCPG reports the same measure; ADG's angle_history is the analogue.
        self.candidate_angle_history: list[float] = []

    def compute_phases(self) -> None:
        """
        Run Algorithm 2 — Cone-Projected Greedy (CPG).

        After completion:
          - self.basis_matrix  : ndarray (R, d)  — the R selected snapshots
          - self.selected_indices : list[int]     — their positions in P^tr
          - self.residual_history : list[float]   — r_n values (diagnostic)
          - self.relative_residual_history : list[float] — r_n/max_q ||λ_q||
          - self.candidate_angle_history : list[float] — angle (radians) of
            each accepted snapshot to the cone it joined
        """
        self._reset_common_history()
        self.candidate_angle_history = []

        # r_0 := max_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_0}(λ(μ))‖  (K^+_0 = {0})
        residuals = self._compute_residuals()
        r_n = float(np.max(residuals)) if residuals.size else 0.0

        # A cone of every snapshot represents the training set exactly, so R can
        # never need to exceed N. The bound is not cosmetic: at epsilon=0 the
        # residual bottoms out at rounding noise (~1e-15) which never drops to
        # exactly 0, and without it the loop re-selects the same snapshot
        # forever, growing the basis without bound. mCPG has always had this
        # guard; CPG lacked it.
        max_iterations = self.snapshots.shape[0]

        while r_n > self.stopping_tolerance and len(self._basis) < max_iterations:
            # μ_n = argmax_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_{n-1}}(λ(μ))‖
            mu_n_idx = int(np.argmax(residuals))
            r_n = float(residuals[mu_n_idx])

            # Convergence check (avoids adding a near-zero vector)
            if r_n <= self.stopping_tolerance:
                break

            candidate = self.snapshots[mu_n_idx]
            # Angle between the candidate and its projection onto K^+_{n-1},
            # i.e. how much new information it adds, computed before the cone
            # is updated (diagnostic only; the selection above used the
            # residual, not the angle).
            projection = self._project_onto_cone(candidate)
            self.candidate_angle_history.append(
                vector_projection_angle(candidate, projection)
            )

            # K^+_n := span_+{λ(μ_1), ..., λ(μ_n)}
            self._basis.append(candidate.copy())
            self.selected_indices.append(mu_n_idx)

            # r_n := max_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_n}(λ(μ))‖
            #   (recomputed once with the updated cone; this doubles as the
            #   next iteration's starting residuals, so it costs one sweep per
            #   accepted candidate instead of two)
            residuals = self._compute_residuals()
            r_n = float(np.max(residuals)) if residuals.size else 0.0
            self.residual_history.append(r_n)
            self.residual_basis_sizes.append(len(self._basis))
            self.relative_residual_history.append(r_n / self.stopping_scale)

        self._finalize_basis()
