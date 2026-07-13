from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib.pyplot as plt

from greedy.core.reduction_common import (
    solve_cone_shift_projection,
    solve_nonnegative_least_squares,
    vector_projection_angle,
)

try:
    from joblib import Parallel, delayed
except ImportError:  # joblib is optional; fall back to serial residuals.
    Parallel = None
    delayed = None


class mCPG:
    """
    Modified Cone-Projected Greedy (mCPG) algorithm.

    Builds a reduced positive cone  W^+_R = span_+{nu_1, ..., nu_R}  such that
    every training snapshot theta(mu) is approximated within a relative
    residual tolerance ``epsilon``, following Algorithm 2 of Niakh, Drouet,
    Ehrlacher & Ern (2022), "Stable model reduction for linear variational
    inequalities with parameter-dependent constraints".

    Unlike CPG, which stores the raw selected snapshot theta_qr as the r-th
    generator, mCPG first removes from theta_qr everything already
    representable by the current cone K_{r-1} while staying inside the
    positive cone: it solves

        Upsilon_r = argmin_{Upsilon in K_{r-1} \\cap (theta_qr - W^+)} ||theta_qr - Upsilon||,

    then sets nu_r = (theta_qr - Upsilon_r) / ||theta_qr - Upsilon_r||. The
    extra elementwise constraint theta_qr - Upsilon >= 0 (so that
    Upsilon in theta_qr - W^+) is what keeps the shift itself a valid,
    physically meaningful cone element; without it this would just be the
    plain CPG-style unconstrained cone projection. Normalizing the residual
    shift instead of the raw snapshot yields a generator that is closer to
    orthogonal to K_{r-1}, which the paper shows produces a
    better-conditioned Gram matrix than plain CPG (see ``orth_defect_history``
    below, matching eq. (41) / Figure 6 of the paper).

    ``epsilon`` is the paper's normalized tolerance δ: the loop stops when
    max_q ||theta_q - Pi_K(theta_q)||_2 / ||theta_q1||_2 <= δ, with q1 the
    first selected maximum-norm snapshot. CPG and AngularDefectGreedy use the
    same normalization in this repo so epsilon sweeps compare like with like.

    Parameters
    ----------
    snapshots : array-like, shape (N_train, d)
        The HF solution snapshots {theta(mu)}_{mu in P^tr}. Expected to be
        elementwise non-negative, i.e. already valid elements of the
        positive cone W^+ (e.g. contact pressures / Lagrange multipliers).
    epsilon : float
        Relative stopping tolerance δ.
    zero_tol : float
        Numerical tolerance below which a shift/norm is treated as zero.
    upper_bound_tol : float
        Slack added to the elementwise upper bound in the constrained
        projection, to absorb floating-point noise.
    """

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 1e-12,
        upper_bound_tol: float = 1e-9,
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
        self.upper_bound_tol = float(upper_bound_tol)
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

        # Basis vectors nu_1, ..., nu_r of the reduced cone K^+_r — shape (r, d).
        self._basis: list[np.ndarray] = []
        self.selected_indices: list[int] = []  # rows q_1, ..., q_r in snapshots

        # Public outputs (set after compute_phases())
        self.basis_matrix: np.ndarray | None = None
        self.residual_history: list[float] = []          # raw max residual r_n, like CPG
        self.residual_basis_sizes: list[int] = []
        self.relative_residual_history: list[float] = []  # eq. (34): normalized by ||theta_q1||
        self.shift_norm_history: list[float] = []          # ||theta_qr - Upsilon_r|| per step
        self.orth_defect_history: list[float] = []          # eq. (41): e_orth(r-1)
        # Angle (radians) between the *raw* candidate theta_qr and its plain
        # cone projection onto K_{r-1}, before the constrained shift is
        # applied. Directly comparable to CPG.candidate_angle_history and
        # AngularDefectGreedy.angle_history, since all three use the same
        # vector_projection_angle formula on the same "candidate vs.
        # pre-update cone" pair.
        self.candidate_angle_history: list[float] = []

    # ------------------------------------------------------------------
    # Cone projection and residual helpers (mirrors CPG._project_onto_cone)
    # ------------------------------------------------------------------

    def _as_basis_matrix(self) -> np.ndarray:
        if not self._basis:
            return np.empty((self.snapshots.shape[1], 0), dtype=float)
        return np.column_stack(self._basis)

    def _compute_stopping_scale(self) -> float:
        if self.snapshots.shape[0] == 0:
            return 1.0
        scale = float(np.max(np.linalg.norm(self.snapshots, axis=1)))
        return scale if scale > 0.0 else 1.0

    def _project_onto_cone(self, v: np.ndarray) -> np.ndarray:
        """
        Plain nonnegative cone projection Pi_{K_r}(v) (no upper bound),
        exactly as in CPG. Used both for residual/candidate selection and
        for the orthogonality diagnostic.
        """
        if len(self._basis) == 0:
            return np.zeros_like(v)
        A = self._as_basis_matrix()
        x = solve_nonnegative_least_squares(A, v)
        return A @ x

    def _compute_residuals(self) -> np.ndarray:
        """||theta - Pi_{K_r}(theta)|| for every training snapshot."""
        S = self.snapshots

        if len(self._basis) == 0:
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

        def _dist(v: np.ndarray) -> float:
            x = solve_nonnegative_least_squares(A, v)
            return float(np.linalg.norm(v - A @ x))

        if Parallel is None:
            return np.array([_dist(v) for v in S])
        return np.array(Parallel(n_jobs=-1)(delayed(_dist)(v) for v in S))

    # ------------------------------------------------------------------
    # Growth primitives, exposed so fixed-component sweeps (see
    # component_sweep_comparison.fit_mcpg_fixed_components) can drive the
    # algorithm incrementally, the same way they already reach into CPG's
    # and AngularDefectGreedy's internals.
    # ------------------------------------------------------------------

    def _bootstrap(self) -> bool:
        """
        Select q_1 = argmax_q ||theta_q|| and build nu_1 through the exact
        same primitive as every later generator (``_grow_by_one``), instead
        of special-casing it: with K_0 = {0}, Upsilon_1 is trivially 0 (the
        only element of K_0, and the QP's k=0 branch already returns it), so
        this reduces to nu_1 = theta_q1 / ||theta_q1|| as required by
        Algorithm 2, while also populating shift_norm/orth_defect/angle
        history for step 1 instead of leaving it blank.
        """
        if self.snapshots.shape[0] == 0:
            return False
        norms = np.linalg.norm(self.snapshots, axis=1)
        q1 = int(np.argmax(norms))
        if norms[q1] <= self.zero_tol:
            return False
        return self._grow_by_one(q1)

    def _next_candidate(self, residuals: np.ndarray) -> int | None:
        scores = residuals.copy()
        if self.selected_indices:
            scores[np.array(self.selected_indices, dtype=int)] = -np.inf
        candidate = int(np.argmax(scores))
        if not np.isfinite(scores[candidate]):
            return None
        return candidate

    def _grow_by_one(self, candidate: int) -> bool:
        """
        Add the generator built from ``candidate`` (lines 9-11 of Algo. 2):
        solve for Upsilon_r, shift theta_qr by it, and normalize.
        """
        theta = self.snapshots[candidate]
        A = self._as_basis_matrix()

        # Angle of the raw candidate to K_{r-1} (plain cone projection),
        # before the constrained shift — the same "how novel is this
        # direction" diagnostic CPG and AngularDefectGreedy report.
        raw_projection = self._project_onto_cone(theta)
        candidate_angle = vector_projection_angle(theta, raw_projection)

        coeffs = solve_cone_shift_projection(A, theta, upper_tol=self.upper_bound_tol)
        upsilon = A @ coeffs
        shift = theta - upsilon
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm <= self.zero_tol:
            # See Remark 4.2: this can only happen if theta_qr was already in
            # K_{r-1}, in which case the residual gate should already have
            # stopped the loop. Guard against it defensively.
            return False

        nu = shift / shift_norm

        # Orthogonality defect of nu_r w.r.t. K_{r-1}, eq. (41): computed
        # against the *plain* cone projection, before nu_r is appended.
        plain_projection = self._project_onto_cone(nu)
        orth_defect = float(np.linalg.norm(nu - plain_projection))

        self._basis.append(nu)
        self.selected_indices.append(candidate)
        self.shift_norm_history.append(shift_norm)
        self.orth_defect_history.append(orth_defect)
        self.candidate_angle_history.append(candidate_angle)
        return True

    # ------------------------------------------------------------------
    # Main algorithm (Algorithm 2)
    # ------------------------------------------------------------------

    def compute_phases(self) -> None:
        """
        Run Algorithm 2 — modified Cone-Projected Greedy (mCPG).

        After completion:
          - self.basis_matrix          : ndarray (R, d) — the R generators nu_1..nu_R
          - self.selected_indices      : list[int]       — snapshot rows q_1..q_R
          - self.residual_history      : list[float]     — r_n values (diagnostic)
          - self.orth_defect_history   : list[float]     — eq. (41) e_orth(r)
        """
        self._basis = []
        self.selected_indices = []
        self.residual_history = []
        self.residual_basis_sizes = []
        self.relative_residual_history = []
        self.shift_norm_history = []
        self.orth_defect_history = []
        self.candidate_angle_history = []
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

        if not self._bootstrap():
            self.basis_matrix = np.empty((0, self.snapshots.shape[1]), dtype=float)
            return

        first_norm = self.stopping_scale
        residuals = self._compute_residuals()
        r_n = float(np.max(residuals))
        self.residual_history.append(r_n)
        self.residual_basis_sizes.append(len(self._basis))
        self.relative_residual_history.append(r_n / first_norm if first_norm > 0.0 else 0.0)

        max_iterations = self.snapshots.shape[0]
        while r_n > self.stopping_tolerance and len(self._basis) < max_iterations:
            candidate = self._next_candidate(residuals)
            if candidate is None:
                break
            if not self._grow_by_one(candidate):
                break

            residuals = self._compute_residuals()
            r_n = float(np.max(residuals))
            self.residual_history.append(r_n)
            self.residual_basis_sizes.append(len(self._basis))
            self.relative_residual_history.append(
                r_n / first_norm if first_norm > 0.0 else 0.0
            )

        self.basis_matrix = np.array(self._basis, dtype=float)

    # ------------------------------------------------------------------
    # Public diagnostics (mirrors CPG / AngularDefectGreedy)
    # ------------------------------------------------------------------

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project an arbitrary vector onto the computed cone W^+_R (plain cone projection)."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        if self.basis_matrix.size == 0:
            return np.zeros_like(np.asarray(v, dtype=float))
        A = self.basis_matrix.T
        coeffs = solve_nonnegative_least_squares(A, np.asarray(v, dtype=float))
        return A @ coeffs

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
        """Plot each basis generator nu_r."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for k, vec in enumerate(self.basis_matrix):
            label = self._snapshot_label(k, self.selected_indices, snapshot_labels)
            ax.plot(vec, label=f"basis {k + 1}  (snap #{label})")

        ax.set_title(f"mCPG basis  (R = {len(self.basis_matrix)},  epsilon = {self.epsilon:g})")
        ax.set_xlabel("Component index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        return ax

    def plot_convergence(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the residual r_n vs. iteration."""
        if not self.residual_history:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        iterations = (
            self.residual_basis_sizes
            if self.residual_basis_sizes
            else range(1, len(self.residual_history) + 1)
        )
        ax.semilogy(iterations, self.residual_history, marker="o")
        ax.axhline(
            self.stopping_tolerance,
            color="r",
            linestyle="--",
            label=f"relative epsilon = {self.epsilon:g}",
        )
        ax.set_title("mCPG convergence")
        ax.set_xlabel("cone size n")
        ax.set_ylabel("Residual $r_n$")
        ax.legend()
        ax.grid(True)
        return ax

    def __repr__(self) -> str:
        R = len(self.basis_matrix) if self.basis_matrix is not None else len(self._basis)
        status = f"R={R}" if self.basis_matrix is not None else "not yet run"
        return f"mCPG(epsilon={self.epsilon:g}, N_train={len(self.snapshots)}, {status})"
