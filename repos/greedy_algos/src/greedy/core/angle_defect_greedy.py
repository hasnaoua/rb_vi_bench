from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from greedy.core.cone_greedy import ConeGreedy
from greedy.core.reduction_common import (
    row_map,
    solve_nonnegative_least_squares,
    vector_projection_angle,
)


class AngularDefectGreedy(ConeGreedy):
    """
    Angular-defect greedy construction of a reduced positive cone.

    Algorithm used here:
      1. initialize with the two snapshots that have the largest mutual angle;
      2. project all snapshots onto the current positive cone;
      3. among snapshots whose relative projection error is still above
         ``epsilon``,
         add every snapshot attaining the largest angle to the cone;
      4. stop once every unselected snapshot has relative projection error
         <= ``epsilon``.

    The cone projection is computed with NNLS, exactly as in CPG:

        min ||A c - v||_2,  c >= 0

    where the columns of A are the current basis snapshots.
    The stopping criterion is the normalized residual
    ``||snapshot - projection||_2 / max_q ||snapshot_q||_2``. Angles are
    normalized separately by their vector norms.

    ``normalize_snapshots=True`` changes what that shared scale ``max_q
    ||snapshot_q||_2`` is measured against: instead of one global scale
    over all snapshots (which lets a large-magnitude snapshot mask poor
    relative accuracy on a small-magnitude one, since the latter's absolute
    error can dip below the global tolerance long before its own relative
    error does), each snapshot is first divided by its own norm, making the
    stopping criterion a genuine per-snapshot relative-error check. This
    cannot change which snapshot is selected at a given round (the angle
    criterion is provably invariant to positive per-vector scaling), only
    which snapshots still count as unresolved. The returned basis vectors
    are always the original, physical-scale snapshots regardless of this
    flag, since span_+ is likewise invariant to positive per-generator
    scaling.

    "Standard ADG" is this algorithm with ``normalize_snapshots=True``; that
    flag is the only knob. Every round admits exactly the snapshots attaining
    theta_max, so the Theorem 3.5 certificate always holds (see
    ``verify_angular_defect_certificate``).
    """

    _display_name = "Angular defect greedy"
    _basis_xlabel = "stored lambda/contact index"
    _basis_ylabel = "value"
    _residual_ylabel = "max projection residual"
    _residual_curve_label = "max residual $r_p$"

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 1e-14,
        normalize_snapshots: bool = False,
    ):
        self.normalize_snapshots = bool(normalize_snapshots)
        super().__init__(snapshots, epsilon, zero_tol=zero_tol)

    @property
    def _scale_snapshots(self) -> np.ndarray:
        # ADG measures its stopping scale on the (possibly normalized) fit
        # snapshots, unlike CPG/mCPG which use the raw matrix.
        return self._fit_snapshots

    def _init_state(self) -> None:
        # Snapshots used to drive selection/stopping (candidate angle,
        # unresolved-set membership, stopping_scale/tolerance). When
        # normalize_snapshots is True this is a per-row unit-norm copy of
        # self.snapshots, which turns the stopping criterion into a genuine
        # per-snapshot relative-error check instead of one absolute
        # tolerance shared across all snapshots (epsilon * max_x ||x||).
        # Candidate SELECTION is unaffected either way: the angle criterion
        # theta_K(x) is provably invariant to positive per-vector scaling
        # (cone projection commutes with it), so normalizing only changes
        # which snapshots are still "unresolved" at a given round, not which
        # one wins the argmax among them. The cone itself, and every basis
        # vector actually stored in self._basis, always stay on the
        # original physical scale (self.snapshots) regardless of this flag,
        # since span_+ is invariant to positive per-generator scaling too.
        if self.normalize_snapshots:
            norms = np.linalg.norm(self.snapshots, axis=1)
            safe_norms = np.where(norms > self.zero_tol, norms, 1.0)
            self._fit_snapshots = self.snapshots / safe_norms[:, None]
        else:
            self._fit_snapshots = self.snapshots

        # Diagnostics for each accepted greedy candidate after initialization.
        self.angle_history: list[float] = []
        self.candidate_residual_history: list[float] = []

        # Number of snapshots admitted per enrichment round (one entry per
        # completed round) -- the tied-maximizer count.
        self.batch_size_history: list[int] = []

        # Theorem 3.5 (angular defect controls the global error) certificate,
        # one entry per completed enrichment round p: theta_max_history[p] is
        # the largest angle among the unresolved set at that round, and
        # certificate_history[p] = max(stopping_tolerance, stopping_scale *
        # sin(theta_max_history[p])) is a computable a priori bound on
        # residual_history[p] (the max error of the cone *before* that
        # round's enrichment).
        self.theta_max_history: list[float] = []
        self.sigma_history: list[float] = []
        self.certificate_history: list[float] = []

        self.initial_pair: tuple[int, int] | None = None
        self.initial_pair_angle: float | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _angle_between_vector_and_projection(
        self,
        v: np.ndarray,
        projection: np.ndarray,
    ) -> float:
        # Shared with CPG.candidate_angle_history and
        # mCPG.candidate_angle_history so all three report the same
        # "how novel is this candidate direction" diagnostic.
        return vector_projection_angle(v, projection, zero_tol=self.zero_tol)

    @staticmethod
    def _pairwise_largest_angle(
        snapshots: np.ndarray,
        zero_tol: float,
    ) -> tuple[int, int, float] | None:
        norms = np.linalg.norm(snapshots, axis=1)
        valid = norms > zero_tol
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            return None

        normalised = np.zeros_like(snapshots, dtype=float)
        normalised[valid] = snapshots[valid] / norms[valid, None]

        cosine = np.clip(normalised @ normalised.T, -1.0, 1.0)
        cosine[~valid, :] = np.inf
        cosine[:, ~valid] = np.inf
        cosine[np.tril_indices_from(cosine)] = np.inf

        flat_index = int(np.argmin(cosine))
        i, j = np.unravel_index(flat_index, cosine.shape)
        if not np.isfinite(cosine[i, j]):
            return None

        angle = float(np.arccos(cosine[i, j]))
        return int(i), int(j), angle

    def _init_basis(self) -> None:
        """Initialize with the two snapshots forming the largest mutual angle."""
        self._basis = []
        self.selected_indices = []
        self.initial_pair = None
        self.initial_pair_angle = None

        n_snapshots = self.snapshots.shape[0]
        if n_snapshots == 0:
            return
        if n_snapshots == 1:
            if np.linalg.norm(self._fit_snapshots[0]) > self.stopping_tolerance:
                self._basis.append(self.snapshots[0].copy())
                self.selected_indices.append(0)
            return

        pair = self._pairwise_largest_angle(self._fit_snapshots, self.zero_tol)
        if pair is None:
            # Degenerate all-zero data: no useful cone generator is needed.
            return

        i, j, angle = pair
        self._basis = [self.snapshots[i].copy(), self.snapshots[j].copy()]
        self.selected_indices = [i, j]
        self.initial_pair = (i, j)
        self.initial_pair_angle = angle

    # ------------------------------------------------------------------
    # Cone projection and metrics
    # ------------------------------------------------------------------

    def measure_angle(self, v: np.ndarray) -> float:
        """Return the angle between ``v`` and its projection onto the current cone."""
        v = np.asarray(v, dtype=float)
        projection = self._project_onto_cone(v)
        return self._angle_between_vector_and_projection(v, projection)

    def _compute_projection_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return residuals and angles from every snapshot to the current cone.

        Returns
        -------
        residuals:
            ``||snapshot - projection||_2`` for each snapshot, computed on
            ``self._fit_snapshots``. When ``normalize_snapshots=True`` these
            are therefore per-snapshot-relative residuals (each snapshot has
            unit norm going in), not absolute physical-scale residuals; this
            also makes residual_history/relative_residual_history report in
            that same normalized space. angles are unaffected either way,
            since theta_K is invariant to positive per-vector scaling.
        angles:
            angle in radians between each snapshot and its cone projection.
        """
        S = self._fit_snapshots
        norms = np.linalg.norm(S, axis=1)

        if len(self._basis) == 0:
            residuals = norms
            angles = np.where(norms > self.zero_tol, np.pi / 2.0, 0.0)
            return residuals.astype(float), angles.astype(float)

        if len(self._basis) == 1:
            b = self._basis[0]
            bb = float(b @ b)
            if bb <= self.zero_tol:
                residuals = norms
                angles = np.where(norms > self.zero_tol, np.pi / 2.0, 0.0)
                return residuals.astype(float), angles.astype(float)

            coeffs = np.maximum(0.0, S @ b / bb)
            projections = np.outer(coeffs, b)
            residuals = np.linalg.norm(S - projections, axis=1)
            projection_norms = np.linalg.norm(projections, axis=1)

            angles = np.zeros(S.shape[0], dtype=float)
            nonzero = norms > self.zero_tol
            no_projection = nonzero & (projection_norms <= self.zero_tol)
            with_projection = nonzero & ~no_projection
            angles[no_projection] = np.pi / 2.0
            cos_theta = np.zeros(S.shape[0], dtype=float)
            cos_theta[with_projection] = np.einsum(
                "ij,ij->i",
                S[with_projection],
                projections[with_projection],
            ) / (norms[with_projection] * projection_norms[with_projection])
            angles[with_projection] = np.arccos(
                np.clip(cos_theta[with_projection], -1.0, 1.0)
            )
            return residuals.astype(float), angles.astype(float)

        A = self._as_basis_matrix()

        def _metrics(v: np.ndarray) -> tuple[float, float]:
            coeffs = solve_nonnegative_least_squares(A, v)
            projection = A @ coeffs
            residual = float(np.linalg.norm(v - projection))
            angle = self._angle_between_vector_and_projection(v, projection)
            return residual, angle

        metrics = row_map(_metrics, S)
        residuals = np.array([item[0] for item in metrics], dtype=float)
        angles = np.array([item[1] for item in metrics], dtype=float)
        return residuals, angles

    def _next_angle_candidates(
        self,
        residuals: np.ndarray,
        angles: np.ndarray,
    ) -> list[int]:
        unresolved = residuals > self.stopping_tolerance
        if self.selected_indices:
            unresolved[np.array(self.selected_indices, dtype=int)] = False
        if not np.any(unresolved):
            return []

        scores = np.where(unresolved, angles, -np.inf)
        max_score = float(np.max(scores))
        if not np.isfinite(max_score):
            return []

        tied = unresolved & np.isclose(
            angles,
            max_score,
            rtol=1e-12,
            atol=max(self.zero_tol, 1e-14),
        )
        return [int(index) for index in np.flatnonzero(tied)]

    def _exact_max_candidates(
        self,
        eligible: np.ndarray,
        angles: np.ndarray,
    ) -> list[int]:
        """Default per-round selection: the snapshots at exactly theta_max."""
        if eligible.size == 0:
            return []
        theta_max = float(np.max(angles[eligible]))
        tied = eligible[
            np.isclose(
                angles[eligible],
                theta_max,
                rtol=1e-12,
                atol=max(self.zero_tol, 1e-14),
            )
        ]
        return [int(index) for index in tied]

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def compute_phases(self) -> None:
        self._reset_common_history()
        self.angle_history = []
        self.candidate_residual_history = []
        self.batch_size_history = []
        self.theta_max_history = []
        self.sigma_history = []
        self.certificate_history = []
        # Seeds _basis/selected_indices with the widest-angle pair, so it must
        # run after _reset_common_history() has cleared them.
        self._init_basis()

        if not self._basis:
            self._finalize_basis()
            return

        residuals, angles = self._compute_projection_metrics()
        max_residual = float(np.max(residuals))
        self.residual_history.append(max_residual)
        self.residual_basis_sizes.append(len(self._basis))
        self.relative_residual_history.append(max_residual / self.stopping_scale)

        while True:
            unresolved_mask = residuals > self.stopping_tolerance
            if self.selected_indices:
                unresolved_mask[np.array(self.selected_indices, dtype=int)] = False
            eligible = np.flatnonzero(unresolved_mask)
            if eligible.size == 0:
                break  # every snapshot resolved -> normal stop

            # theta_max is the largest angle over the still-unresolved set (the
            # worst remaining direction), used for the certificate histories.
            # Every admitted candidate attains it, so the Theorem 3.5 bound holds.
            theta_max = float(np.max(angles[eligible]))

            candidates = self._exact_max_candidates(eligible, angles)
            if not candidates:
                break

            sigma = float(np.sin(theta_max))
            self.theta_max_history.append(theta_max)
            self.sigma_history.append(sigma)
            self.certificate_history.append(
                max(self.stopping_tolerance, self.stopping_scale * sigma)
            )

            accepted_count = 0
            # S_norm is a SET: snapshots on the same positive ray collapse to one
            # element, so card(S_norm) <= N. That matters exactly here. A batch admits
            # EVERY snapshot attaining theta_max, and coincident rays necessarily tie --
            # so without this guard a k-fold repeated direction would enter the cone as
            # k separate generators. They add nothing to span_+, inflate n*, and make
            # the Gram matrix singular. Duplicates of an ALREADY-stored generator cannot
            # reach here (they project exactly, so their residual is 0 and they are not
            # eligible); only within-batch repeats need catching.
            accepted_units: list[np.ndarray] = []
            for candidate in candidates:
                candidate_residual = float(residuals[candidate])
                candidate_angle = float(angles[candidate])
                if candidate_residual <= self.stopping_tolerance:
                    continue

                row = self._fit_snapshots[candidate]
                row_norm = float(np.linalg.norm(row))
                if row_norm > self.zero_tol:
                    unit = row / row_norm
                    if any(float(unit @ other) > 1.0 - 1e-12 for other in accepted_units):
                        continue
                    accepted_units.append(unit)

                self._basis.append(self.snapshots[candidate].copy())
                self.selected_indices.append(candidate)
                self.angle_history.append(candidate_angle)
                self.candidate_residual_history.append(candidate_residual)
                accepted_count += 1

            if accepted_count == 0:
                break
            self.batch_size_history.append(accepted_count)

            residuals, angles = self._compute_projection_metrics()
            max_residual = float(np.max(residuals))
            self.residual_history.append(max_residual)
            self.residual_basis_sizes.append(len(self._basis))
            self.relative_residual_history.append(max_residual / self.stopping_scale)

        self.basis_matrix = np.array(self._basis, dtype=float)

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def verify_angular_defect_certificate(self, *, atol: float = 1e-9) -> dict:
        """
        Check Theorem 3.5 against the recorded convergence history:

            residual_history[p] <= max(stopping_tolerance,
                                        stopping_scale * sin(theta_max_history[p]))

        for every completed enrichment round p. Requires ``compute_phases()``
        to have been called first.
        """
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")

        observed = np.asarray(self.residual_history[: len(self.theta_max_history)], dtype=float)
        bound = np.asarray(self.certificate_history, dtype=float)
        slack = observed - bound
        tolerance = atol * max(1.0, self.stopping_scale)
        violations = np.flatnonzero(slack > tolerance)

        return {
            "checked": int(observed.size),
            "violations": int(violations.size),
            "max_violation": float(np.max(slack)) if slack.size else 0.0,
            "violation_rounds": violations.tolist(),
        }

    def final_residuals(self) -> np.ndarray:
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        residuals, _ = self._compute_projection_metrics()
        return residuals

    def plot_convergence(
        self,
        ax: plt.Axes | None = None,
        *,
        show_certificate: bool = False,
    ) -> plt.Axes:
        """The shared convergence plot, plus ADG's Theorem 3.5 bound."""
        ax = super().plot_convergence(ax)
        if show_certificate and self.certificate_history:
            sizes = list(self.residual_basis_sizes)[: len(self.certificate_history)]
            ax.semilogy(
                sizes,
                self.certificate_history,
                marker="^",
                linestyle=":",
                color="#7c3aed",
                label=r"Thm 3.5 bound $\max(\varepsilon, C\sin\theta_p^{max})$",
            )
            ax.legend()  # refresh so the bound appears
        return ax
