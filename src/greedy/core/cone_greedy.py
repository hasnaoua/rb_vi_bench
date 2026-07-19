"""
Shared scaffolding for the cone greedies (CPG, mCPG, AngularDefectGreedy).

The three algorithms differ only in how they *choose* the next snapshot and what
they *store* as a generator. Everything around that choice --- validating the
snapshot matrix, the NNLS cone projection, the stopping scale, the residual
sweep and the diagnostic plots --- is common, and used to exist as three
near-identical copies. It lives here once.

Subclasses implement ``compute_phases()`` and override the hooks they need:

* ``_init_state()`` --- extra per-algorithm state, set up before the stopping
  scale is computed (this ordering is what lets ADG normalize its snapshots
  first and still have the scale measured on them);
* ``_scale_snapshots`` --- the snapshots the stopping scale is measured on
  (ADG measures it on its normalized copy, the others on the raw matrix);
* the ``_display_name`` / label attributes, which only affect plot text.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from greedy.core.reduction_common import row_map, solve_nonnegative_least_squares


class ConeGreedy:
    """
    Base for greedies that grow a positive cone K = span_+{g_1, ..., g_R}.

    Parameters
    ----------
    snapshots : array-like, shape (N_train, d)
        Training snapshots, one per row.
    epsilon : float
        Relative stopping tolerance, applied as ``epsilon * stopping_scale``.
    zero_tol : float
        Magnitude below which a norm is treated as zero.
    """

    # Plot text only; no effect on the algorithms.
    _display_name: str = "cone greedy"
    _basis_xlabel: str = "Component index"
    _basis_ylabel: str = "Value"
    _residual_ylabel: str = "Residual $r_n$"
    _residual_curve_label: str | None = None

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 0.0,
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

        # Generators of the cone, kept as a list while growing and stacked into
        # basis_matrix (R, d) at the end of compute_phases().
        self._basis: list[np.ndarray] = []
        self.basis_matrix: np.ndarray | None = None
        self.selected_indices: list[int] = []

        # Diagnostics shared by all three algorithms. Per-algorithm histories
        # (mCPG's shift norms, ADG's angles) are declared in _init_state().
        self.residual_history: list[float] = []
        self.residual_basis_sizes: list[int] = []
        self.relative_residual_history: list[float] = []

        self._init_state()
        self._refresh_stopping_scale()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _init_state(self) -> None:
        """Per-algorithm state, set up before the stopping scale is computed."""

    @property
    def _scale_snapshots(self) -> np.ndarray:
        """The snapshots the stopping scale is measured on."""
        return self.snapshots

    # ------------------------------------------------------------------
    # Stopping scale
    # ------------------------------------------------------------------

    def _compute_stopping_scale(self) -> float:
        snapshots = self._scale_snapshots
        if snapshots.shape[0] == 0:
            return 1.0
        scale = float(np.max(np.linalg.norm(snapshots, axis=1)))
        return scale if scale > 0.0 else 1.0

    def _refresh_stopping_scale(self) -> None:
        """(Re)compute the scale and the absolute tolerance derived from it."""
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

    def _reset_common_history(self) -> None:
        """
        Clear the shared outputs so compute_phases() is idempotent.

        Subclasses clear their own extra histories on top of this.
        """
        self._basis = []
        self.selected_indices = []
        self.residual_history = []
        self.residual_basis_sizes = []
        self.relative_residual_history = []
        self._refresh_stopping_scale()

    # ------------------------------------------------------------------
    # Cone geometry
    # ------------------------------------------------------------------

    def _as_basis_matrix(self) -> np.ndarray:
        """Current generators as columns, shape (d, r)."""
        if not self._basis:
            return np.empty((self.snapshots.shape[1], 0), dtype=float)
        return np.column_stack(self._basis)

    def _project_onto_cone(self, v: np.ndarray) -> np.ndarray:
        """
        Pi_K(v) via NNLS: min ||A c - v||_2 subject to c >= 0, with the current
        generators as the columns of A. The projection onto the empty cone {0}
        is the zero vector.
        """
        if not self._basis:
            return np.zeros_like(v)
        A = self._as_basis_matrix()
        return A @ solve_nonnegative_least_squares(A, v)

    def _compute_residuals(self, snapshots: np.ndarray | None = None) -> np.ndarray:
        """
        ||x - Pi_K(x)|| for every row of ``snapshots`` (default: self.snapshots).

        Note the name is load-bearing beyond this package:
        ``pipelines.component_sweep`` drives the greedies incrementally and
        calls this directly.

        Three paths, purely for speed --- they agree to floating-point exactness:
        an empty cone leaves each snapshot untouched; a single generator has a
        closed-form nonnegative projection; beyond that it is NNLS per row.
        """
        S = self.snapshots if snapshots is None else snapshots

        if not self._basis:
            return np.linalg.norm(S, axis=1)

        if len(self._basis) == 1:
            b = self._basis[0]
            bb = float(b @ b)
            if bb == 0.0:
                return np.linalg.norm(S, axis=1)
            coeffs = np.maximum(0.0, S @ b / bb)
            projections = np.outer(coeffs, b)
            return np.linalg.norm(S - projections, axis=1)

        A = self._as_basis_matrix()

        def distance(v: np.ndarray) -> float:
            return float(np.linalg.norm(v - A @ solve_nonnegative_least_squares(A, v)))

        return np.array(row_map(distance, S), dtype=float)

    def _finalize_basis(self) -> None:
        """Stack the grown generators into basis_matrix (R, d)."""
        self.basis_matrix = (
            np.array(self._basis, dtype=float)
            if self._basis
            else np.empty((0, self.snapshots.shape[1]), dtype=float)
        )

    def compute_phases(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project an arbitrary vector onto the computed cone."""
        self._require_fitted()
        v = np.asarray(v, dtype=float)
        if self.basis_matrix.size == 0:
            return np.zeros_like(v)
        A = self.basis_matrix.T
        return A @ solve_nonnegative_least_squares(A, v)

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
        """Plot each generator of the computed cone."""
        self._require_fitted()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for k, vector in enumerate(self.basis_matrix):
            label = self._snapshot_label(k, self.selected_indices, snapshot_labels)
            ax.plot(vector, label=f"basis {k + 1}  (snap #{label})")

        ax.set_title(
            f"{self._display_name} basis  "
            f"(R = {len(self.basis_matrix)}, epsilon = {self.epsilon:g})"
        )
        ax.set_xlabel(self._basis_xlabel)
        ax.set_ylabel(self._basis_ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_convergence(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the max residual r_n against cone size."""
        if not self.residual_history:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        sizes = self.residual_basis_sizes or range(1, len(self.residual_history) + 1)
        ax.semilogy(
            sizes, self.residual_history, marker="o", label=self._residual_curve_label
        )
        ax.axhline(
            self.stopping_tolerance,
            color="r",
            linestyle="--",
            label=f"relative epsilon = {self.epsilon:g}",
        )
        ax.set_title(f"{self._display_name} convergence")
        ax.set_xlabel("cone size n")
        ax.set_ylabel(self._residual_ylabel)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        return ax

    def __repr__(self) -> str:
        fitted = self.basis_matrix is not None
        R = len(self.basis_matrix) if fitted else len(self._basis)
        status = f"R={R}" if fitted else "not yet run"
        return (
            f"{type(self).__name__}(epsilon={self.epsilon:g}, "
            f"N_train={len(self.snapshots)}, {status})"
        )
