from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt

from greedy.core.reduction_common import solve_nonnegative_least_squares, vector_projection_angle

try:
    from joblib import Parallel, delayed
except ImportError:  # joblib is optional; fall back to serial residuals.
    Parallel = None
    delayed = None


class AngularDefectGreedy:
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

    Two independent angular knobs act on each round (both default ``None`` =
    original behaviour; both take a value in ``[0, 1]``; they may be combined):

    * ``angle_redundancy_tol`` (a *diminish-R* filter). An unresolved snapshot is
      eligible only if its pairwise angle to *every already-selected generator*
      exceeds ``arcsin(angle_redundancy_tol)``; those nearly parallel to an
      existing generator are dropped as redundant, and enrichment stops once every
      remaining unresolved snapshot is redundant. R and the basis condition number
      shrink at a controlled accuracy cost (worst-case error may rise). Because a
      maximizer can be dropped, the Theorem 3.5 certificate is guaranteed only when
      this is ``None``.
    * ``angle_batch_tol`` (a *fewer-rounds* filter, R-neutral). Within each round
      it takes the band of almost-max-angle eligible candidates (angle within
      ``arcsin(angle_batch_tol)`` of theta_max) and admits a subset that is
      pairwise separated *from each other* by more than ``arcsin(angle_batch_tol)``
      -- several non-redundant directions enter at once, cutting the number of
      greedy rounds without changing the accuracy or (materially) R.

    Pairwise angles use snapshot *directions* and are scale-invariant, so both
    filters compose with ``normalize_snapshots``.
    """

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 1e-14,
        normalize_snapshots: bool = False,
        angle_redundancy_tol: float | None = None,
        angle_batch_tol: float | None = None,
    ):
        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots must be 2-D (N_train, d), got shape {snapshots.shape}"
            )
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if angle_redundancy_tol is not None and not (0.0 <= angle_redundancy_tol <= 1.0):
            raise ValueError("angle_redundancy_tol (epsilon_angle) must be in [0, 1]")
        if angle_batch_tol is not None and not (0.0 <= angle_batch_tol <= 1.0):
            raise ValueError("angle_batch_tol (epsilon_angle) must be in [0, 1]")

        self.snapshots = snapshots
        self.epsilon = float(epsilon)
        self.zero_tol = float(zero_tol)
        self.normalize_snapshots = bool(normalize_snapshots)
        self.angle_redundancy_tol = (
            None if angle_redundancy_tol is None else float(angle_redundancy_tol)
        )
        self.angle_batch_tol = (
            None if angle_batch_tol is None else float(angle_batch_tol)
        )

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
            norms = np.linalg.norm(snapshots, axis=1)
            safe_norms = np.where(norms > self.zero_tol, norms, 1.0)
            self._fit_snapshots = snapshots / safe_norms[:, None]
        else:
            self._fit_snapshots = snapshots

        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

        self._basis: list[np.ndarray] = []

        self.basis_matrix: np.ndarray | None = None
        self.selected_indices: list[int] = []

        # Max raw residual after initialization and after each accepted batch.
        self.residual_history: list[float] = []
        self.residual_basis_sizes: list[int] = []
        self.relative_residual_history: list[float] = []

        # Diagnostics for each accepted greedy candidate after initialization.
        self.angle_history: list[float] = []
        self.candidate_residual_history: list[float] = []

        # Number of snapshots admitted per enrichment round (one entry per
        # completed round). With angle_redundancy_tol=None this is the
        # tied-maximizer count; with the redundancy filter each round admits one
        # non-redundant generator, so entries are 1.
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

    def _as_basis_matrix(self) -> np.ndarray:
        if not self._basis:
            return np.empty((self.snapshots.shape[1], 0), dtype=float)
        return np.column_stack(self._basis)

    def _compute_stopping_scale(self) -> float:
        if self._fit_snapshots.shape[0] == 0:
            return 1.0
        scale = float(np.max(np.linalg.norm(self._fit_snapshots, axis=1)))
        return scale if scale > 0.0 else 1.0

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

    def _project_onto_cone(self, v: np.ndarray) -> np.ndarray:
        if len(self._basis) == 0:
            return np.zeros_like(v)

        A = self._as_basis_matrix()
        coeffs = solve_nonnegative_least_squares(A, v)
        return A @ coeffs

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

        if Parallel is None:
            metrics = [_metrics(v) for v in S]
        else:
            metrics = Parallel(n_jobs=-1)(delayed(_metrics)(v) for v in S)

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

    def _drop_redundant(self, indices: np.ndarray) -> np.ndarray:
        """
        angle_redundancy_tol filter (diminish-R): drop candidates nearly parallel
        to an existing generator.

        Return the subset of ``indices`` whose pairwise angle to *every*
        already-selected generator exceeds ``arcsin(angle_redundancy_tol)``. An
        empty result means every remaining candidate is redundant, which signals
        ``compute_phases`` to stop enriching. Pairwise angles use snapshot
        directions and are scale-invariant.
        """
        if not self.selected_indices or indices.size == 0:
            return indices
        min_separation = float(np.arcsin(np.clip(self.angle_redundancy_tol, 0.0, 1.0)))
        selected = self._fit_snapshots[np.array(self.selected_indices, dtype=int)]
        selected_norms = np.linalg.norm(selected, axis=1, keepdims=True)
        selected_units = np.divide(
            selected,
            selected_norms,
            out=np.zeros_like(selected),
            where=selected_norms > self.zero_tol,
        )
        kept: list[int] = []
        for index in indices:
            v = self._fit_snapshots[index]
            norm = float(np.linalg.norm(v))
            if norm <= self.zero_tol:
                continue
            unit = v / norm
            cosines = np.clip(selected_units @ unit, -1.0, 1.0)
            if float(np.min(np.arccos(cosines))) > min_separation:
                kept.append(int(index))
        return np.asarray(kept, dtype=int)

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

    def _band_pairwise_candidates(
        self,
        eligible: np.ndarray,
        angles: np.ndarray,
    ) -> list[int]:
        """
        angle_batch_tol selection (fewer-rounds): from the eligible pool take the
        band of almost-max-angle candidates (angle within
        ``arcsin(angle_batch_tol)`` of theta_max) and admit a subset that is
        pairwise separated FROM EACH OTHER by more than ``arcsin(angle_batch_tol)``,
        traversed max-angle first. This adds several non-redundant directions per
        round (fewer rounds) without inflating R materially.
        """
        if eligible.size == 0:
            return []
        sep = float(np.arcsin(np.clip(self.angle_batch_tol, 0.0, 1.0)))
        theta_max = float(np.max(angles[eligible]))
        band = eligible[angles[eligible] >= theta_max - sep]
        order = band[np.argsort(angles[band], kind="stable")[::-1]]
        admitted: list[int] = []
        admitted_units: list[np.ndarray] = []
        for candidate in order:
            v = self._fit_snapshots[candidate]
            norm = float(np.linalg.norm(v))
            if norm <= self.zero_tol:
                continue
            unit = v / norm
            if admitted:
                cosines = np.clip(np.asarray(admitted_units) @ unit, -1.0, 1.0)
                if float(np.min(np.arccos(cosines))) <= sep:
                    continue
            admitted.append(int(candidate))
            admitted_units.append(unit)
        return admitted

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def compute_phases(self) -> None:
        self.residual_history = []
        self.residual_basis_sizes = []
        self.relative_residual_history = []
        self.angle_history = []
        self.candidate_residual_history = []
        self.batch_size_history = []
        self.theta_max_history = []
        self.sigma_history = []
        self.certificate_history = []
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale
        self._init_basis()

        if not self._basis:
            self.basis_matrix = np.empty((0, self.snapshots.shape[1]), dtype=float)
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
            # worst remaining direction), used for the certificate histories. In
            # redundancy mode the admitted candidate need not attain it (a
            # redundant maximizer may be dropped), so the Theorem 3.5 certificate
            # is guaranteed only when angle_redundancy_tol is None.
            theta_max = float(np.max(angles[eligible]))

            # (1) diminish-R knob: keep only candidates that are not redundant
            # w.r.t. the existing generators; stop if all remaining are redundant.
            if self.angle_redundancy_tol is not None:
                eligible = self._drop_redundant(eligible)
                if eligible.size == 0:
                    break

            # (2) fewer-rounds knob: per-round selection from the eligible pool.
            if self.angle_batch_tol is None:
                candidates = self._exact_max_candidates(eligible, angles)
            else:
                candidates = self._band_pairwise_candidates(eligible, angles)
            if not candidates:
                break

            sigma = float(np.sin(theta_max))
            self.theta_max_history.append(theta_max)
            self.sigma_history.append(sigma)
            self.certificate_history.append(
                max(self.stopping_tolerance, self.stopping_scale * sigma)
            )

            accepted_count = 0
            for candidate in candidates:
                candidate_residual = float(residuals[candidate])
                candidate_angle = float(angles[candidate])
                if candidate_residual <= self.stopping_tolerance:
                    continue

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

    def project(self, v: np.ndarray) -> np.ndarray:
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        if self.basis_matrix.size == 0:
            return np.zeros_like(v, dtype=float)
        A = self.basis_matrix.T
        coeffs = solve_nonnegative_least_squares(A, np.asarray(v, dtype=float))
        return A @ coeffs

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

    @staticmethod
    def _snapshot_label(
        index: int,
        selected_indices: list[int],
        snapshot_labels: list[int] | np.ndarray | None = None,
    ) -> int:
        labels = selected_indices if snapshot_labels is None else snapshot_labels
        return int(labels[index]) if index < len(labels) else index

    def plot_basis(
        self,
        ax: plt.Axes | None = None,
        snapshot_labels: list[int] | np.ndarray | None = None,
    ) -> plt.Axes:
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for k, vec in enumerate(self.basis_matrix):
            label = self._snapshot_label(k, self.selected_indices, snapshot_labels)
            ax.plot(vec, label=f"basis {k + 1} (snap #{label})")

        ax.set_title(
            f"Angular defect greedy basis "
            f"(R = {len(self.basis_matrix)}, epsilon = {self.epsilon:g})"
        )
        ax.set_xlabel("stored lambda/contact index")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_convergence(
        self,
        ax: plt.Axes | None = None,
        *,
        show_certificate: bool = False,
    ) -> plt.Axes:
        if not self.residual_history:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        iterations = (
            self.residual_basis_sizes
            if self.residual_basis_sizes
            else range(len(self.residual_history))
        )
        ax.semilogy(iterations, self.residual_history, marker="o", label="max residual $r_p$")
        ax.axhline(
            self.stopping_tolerance,
            color="r",
            linestyle="--",
            label=f"relative epsilon = {self.epsilon:g}",
        )
        if show_certificate and self.certificate_history:
            ax.semilogy(
                list(iterations)[: len(self.certificate_history)],
                self.certificate_history,
                marker="^",
                linestyle=":",
                color="#7c3aed",
                label=r"Thm 3.5 bound $\max(\varepsilon, C\sin\theta_p^{max})$",
            )
        ax.set_title("Angular defect greedy convergence")
        ax.set_xlabel("cone size n")
        ax.set_ylabel("max projection residual")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        return ax

    def __repr__(self) -> str:
        R = len(self.basis_matrix) if self.basis_matrix is not None else len(self._basis)
        status = f"R={R}" if self.basis_matrix is not None else "not yet run"
        return (
            f"AngularDefectGreedy(epsilon={self.epsilon:g}, "
            f"N_train={len(self.snapshots)}, {status})"
        )
