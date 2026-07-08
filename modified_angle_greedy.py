from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt

from reduction_common import solve_nonnegative_least_squares

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
      3. among snapshots whose projection error is still above ``epsilon``,
         add every snapshot attaining the largest angle to the cone;
      4. stop once every unselected snapshot has projection error <= ``epsilon``.

    The cone projection is computed with NNLS, exactly as in CPG:

        min ||A c - v||_2,  c >= 0

    where the columns of A are the current basis snapshots.
    The stopping/error criterion is the raw-vector residual
    ``||snapshot - projection||_2``; normalization is used only for angles.
    """

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 1e-14,
    ):
        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots must be 2-D (N_train, d), got shape {snapshots.shape}"
            )
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")

        self.snapshots = snapshots
        self.epsilon = float(epsilon)
        self.zero_tol = float(zero_tol)

        self._basis: list[np.ndarray] = []

        self.basis_matrix: np.ndarray | None = None
        self.selected_indices: list[int] = []

        # Max raw residual after initialization and after each accepted batch.
        self.residual_history: list[float] = []
        self.residual_basis_sizes: list[int] = []

        # Diagnostics for each accepted greedy candidate after initialization.
        self.angle_history: list[float] = []
        self.candidate_residual_history: list[float] = []

        self.initial_pair: tuple[int, int] | None = None
        self.initial_pair_angle: float | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _as_basis_matrix(self) -> np.ndarray:
        if not self._basis:
            return np.empty((self.snapshots.shape[1], 0), dtype=float)
        return np.column_stack(self._basis)

    def _angle_between_vector_and_projection(
        self,
        v: np.ndarray,
        projection: np.ndarray,
    ) -> float:
        norm_v = float(np.linalg.norm(v))
        if norm_v <= self.zero_tol:
            return 0.0

        norm_projection = float(np.linalg.norm(projection))
        if norm_projection <= self.zero_tol:
            return float(np.pi / 2.0)

        cos_theta = float(np.dot(v, projection) / (norm_v * norm_projection))
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

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
            if np.linalg.norm(self.snapshots[0]) > self.epsilon:
                self._basis.append(self.snapshots[0].copy())
                self.selected_indices.append(0)
            return

        pair = self._pairwise_largest_angle(self.snapshots, self.zero_tol)
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

    def mesure_angle(self, v: np.ndarray) -> float:
        """Backward-compatible alias for the old misspelled method name."""
        return self.measure_angle(v)

    def _compute_projection_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return residuals and angles from every snapshot to the current cone.

        Returns
        -------
        residuals:
            ``||snapshot - projection||_2`` for each snapshot.
        angles:
            angle in radians between each snapshot and its cone projection.
        """
        S = self.snapshots
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
        unresolved = residuals > self.epsilon
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

    def _next_angle_candidate(
        self,
        residuals: np.ndarray,
        angles: np.ndarray,
    ) -> int | None:
        """Backward-compatible single-candidate view of the batch argmax."""
        candidates = self._next_angle_candidates(residuals, angles)
        return candidates[0] if candidates else None

    def farthest_snapshot(self, *, only_unresolved: bool = True) -> int | None:
        """
        Return the snapshot with the largest angle to the current cone.

        By default, snapshots already approximated within ``epsilon`` and already
        selected snapshots are ignored. Set ``only_unresolved=False`` to select
        by angle alone.
        """
        residuals, angles = self._compute_projection_metrics()
        if only_unresolved:
            return self._next_angle_candidate(residuals, angles)
        if self.selected_indices:
            angles = angles.copy()
            angles[np.array(self.selected_indices, dtype=int)] = -np.inf
        if not np.any(np.isfinite(angles)):
            return None
        return int(np.argmax(angles))

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------

    def compute_phases(self) -> None:
        self.residual_history = []
        self.residual_basis_sizes = []
        self.angle_history = []
        self.candidate_residual_history = []
        self._init_basis()

        if not self._basis:
            self.basis_matrix = np.empty((0, self.snapshots.shape[1]), dtype=float)
            return

        residuals, angles = self._compute_projection_metrics()
        self.residual_history.append(float(np.max(residuals)))
        self.residual_basis_sizes.append(len(self._basis))

        while True:
            candidates = self._next_angle_candidates(residuals, angles)
            if not candidates:
                break

            accepted_any = False
            for candidate in candidates:
                candidate_residual = float(residuals[candidate])
                candidate_angle = float(angles[candidate])
                if candidate_residual <= self.epsilon:
                    continue

                self._basis.append(self.snapshots[candidate].copy())
                self.selected_indices.append(candidate)
                self.angle_history.append(candidate_angle)
                self.candidate_residual_history.append(candidate_residual)
                accepted_any = True

            if not accepted_any:
                break

            residuals, angles = self._compute_projection_metrics()
            self.residual_history.append(float(np.max(residuals)))
            self.residual_basis_sizes.append(len(self._basis))

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

    def plot_convergence(self, ax: plt.Axes | None = None) -> plt.Axes:
        if not self.residual_history:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        iterations = (
            self.residual_basis_sizes
            if self.residual_basis_sizes
            else range(len(self.residual_history))
        )
        ax.semilogy(iterations, self.residual_history, marker="o")
        ax.axhline(
            self.epsilon,
            color="r",
            linestyle="--",
            label=f"epsilon = {self.epsilon:g}",
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
