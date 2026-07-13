import numpy as np
import matplotlib.pyplot as plt

from greedy.core.reduction_common import solve_nonnegative_least_squares, vector_projection_angle

try:
    from joblib import Parallel, delayed
except ImportError:  # joblib is optional; fall back to serial residuals.
    Parallel = None
    delayed = None


class CPG:
    """
    Cone-Projected Greedy (CPG) algorithm.

    Builds a reduced positive cone  W^+ = span_+{λ(μ_1), ..., λ(μ_R)}
    such that every training snapshot λ(μ) ∈ S_DU is approximated within the
    relative tolerance ε_DU, using
    max_μ ||λ(μ) - Π_K(λ(μ))||_2 / max_μ ||λ(μ)||_2.

    Parameters
    ----------
    snapshots : array-like, shape (N_train, d)
        The HF solution snapshots {λ(μ)}_{ μ ∈ P^tr }.
    epsilon : float
        Relative tolerance ε_DU >= 0.
    """

    def __init__(self, snapshots: np.ndarray, epsilon: float = 0.1):
        snapshots = np.asarray(snapshots, dtype=float)
        if snapshots.ndim != 2:
            raise ValueError(
                f"snapshots must be 2-D (N_train * d), got shape {snapshots.shape}"
            )
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        self.snapshots = snapshots          # shape (N_train, d)
        self.epsilon = float(epsilon)
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

        # Basis vectors of the reduced cone  K^+_n  — shape (n, d)
        # Kept as a Python list during construction, converted at the end.
        self._basis: list[np.ndarray] = []

        # Public outputs (set after compute_phases())
        self.basis_matrix: np.ndarray | None = None   # shape (R, d)
        self.selected_indices: list[int] = []          # indices in snapshots
        self.residual_history: list[float] = []        # r_n at each iteration
        self.residual_basis_sizes: list[int] = []      # cone size for each r_n
        self.relative_residual_history: list[float] = []
        # Angle (radians) between each accepted candidate and its projection
        # onto the cone it is being added to, i.e. how much genuinely new
        # information it contributes. Same diagnostic as
        # AngularDefectGreedy.angle_history and mCPG.candidate_angle_history,
        # via reduction_common.vector_projection_angle.
        self.candidate_angle_history: list[float] = []

    # ------------------------------------------------------------------
    # Core subroutines
    # ------------------------------------------------------------------

    def _compute_stopping_scale(self) -> float:
        if self.snapshots.shape[0] == 0:
            return 1.0
        scale = float(np.max(np.linalg.norm(self.snapshots, axis=1)))
        return scale if scale > 0.0 else 1.0

    def _project_onto_cone(self, v: np.ndarray) -> np.ndarray:
        """
        Compute  Π_{K^+_{n-1}}(v)  via NNLS:
            min  ‖A x - v‖  s.t.  x ≥ 0
        where the columns of A are the current basis vectors.

        If the basis is empty, the projection onto {0} is the zero vector.
        """
        if len(self._basis) == 0:
            return np.zeros_like(v)

        # A : shape (d, n)  — each column is a basis vector
        A = np.column_stack(self._basis)
        x = solve_nonnegative_least_squares(A, v)
        return A @ x

    def _compute_residuals(self) -> np.ndarray:
        """
        Compute  r(μ) = ‖λ(μ) - Π_{K^+_{n-1}}(λ(μ))‖  for all μ ∈ P^tr.

        - Empty basis  → residual is simply the norm of each snapshot.
        - Single vector → closed-form non-negative projection (no NNLS call).
        - General case  → parallel NNLS via joblib.

        Returns
        -------
        residuals : ndarray, shape (N_train,)
        """
        S = self.snapshots

        # ── fast path: empty cone ──────────────────────────────────────────
        if len(self._basis) == 0:
            return np.linalg.norm(S, axis=1)

        # ── fast path: single basis vector (n=1) ──────────────────────────
        # Π(v) = max(0, <v,b> / <b,b>) · b  (closed form)
        if len(self._basis) == 1:
            b = self._basis[0]
            bb = float(b @ b)
            if bb == 0.0:
                return np.linalg.norm(S, axis=1)
            coeffs = np.maximum(0.0, S @ b / bb)          # (N,)
            projs  = np.outer(coeffs, b)                   # (N, d)
            return np.linalg.norm(S - projs, axis=1)

        # ── general case: parallel NNLS ────────────────────────────────────
        A = np.column_stack(self._basis)                   # (d, k)

        def _dist(v: np.ndarray) -> float:
            x = solve_nonnegative_least_squares(A, v)
            return float(np.linalg.norm(v - A @ x))

        if Parallel is None:
            return np.array([_dist(v) for v in S])

        return np.array(Parallel(n_jobs=-1)(delayed(_dist)(v) for v in S))

    # ------------------------------------------------------------------
    # Algorithm 2  (faithful to the pseudocode)
    # ------------------------------------------------------------------

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
        # Reset state so the method is idempotent
        self._basis = []
        self.selected_indices = []
        self.residual_history = []
        self.residual_basis_sizes = []
        self.relative_residual_history = []
        self.candidate_angle_history = []
        self.stopping_scale = self._compute_stopping_scale()
        self.stopping_tolerance = self.epsilon * self.stopping_scale

        # r_0 := max_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_0}(λ(μ))‖  (K^+_0 = {0})
        residuals = self._compute_residuals()
        r_n = float(np.max(residuals)) if residuals.size else 0.0

        # main loop
        while r_n > self.stopping_tolerance:
            # μ_n = argmax_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_{n-1}}(λ(μ))‖
            mu_n_idx = int(np.argmax(residuals))
            r_n = float(residuals[mu_n_idx])

            # Convergence check (avoids adding a near-zero vector)
            if r_n <= self.stopping_tolerance:
                break

            candidate = self.snapshots[mu_n_idx]
            # Angle between the candidate and its projection onto K^+_{n-1},
            # i.e. how much new information it adds, computed before the
            # cone is updated (diagnostic only; selection above already
            # used the residual, not the angle).
            projection = self._project_onto_cone(candidate)
            self.candidate_angle_history.append(
                vector_projection_angle(candidate, projection)
            )

            # K^+_n := span_+{λ(μ_1), ..., λ(μ_n)}
            self._basis.append(candidate.copy())
            self.selected_indices.append(mu_n_idx)

            # r_n := max_{μ ∈ P^tr} ‖λ(μ) - Π_{K^+_n}(λ(μ))‖
            #   (recompute once, with the updated cone; this doubles as the
            #   next iteration's starting residuals, so it is only computed
            #   once per accepted candidate instead of twice)
            residuals = self._compute_residuals()
            r_n = float(np.max(residuals)) if residuals.size else 0.0
            self.residual_history.append(r_n)
            self.residual_basis_sizes.append(len(self._basis))
            self.relative_residual_history.append(r_n / self.stopping_scale)

        # results
        self.basis_matrix = (
            np.array(self._basis, dtype=float)
            if self._basis
            else np.empty((0, self.snapshots.shape[1]), dtype=float)
        )

    # ------------------------------------------------------------------
    # Diagnostics & visualisation
    # ------------------------------------------------------------------

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project an arbitrary vector onto the computed cone W^+_R."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        A = self.basis_matrix.T          # shape (d, R)
        x = solve_nonnegative_least_squares(A, v)
        return A @ x

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
        """Plot each basis vector (selected snapshot)."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        for k, vec in enumerate(self.basis_matrix):
            label = self._snapshot_label(k, self.selected_indices, snapshot_labels)
            ax.plot(vec, label=f"basis {k+1}  (snap #{label})")

        ax.set_title(
            f"CPG basis  (R = {len(self.basis_matrix)},  ε = {self.epsilon})"
        )
        ax.set_xlabel("Component index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        return ax

    @staticmethod
    def _heatmap_values(
        matrix: np.ndarray,
        scale: str = "log_abs",
    ) -> tuple[np.ndarray, str, str]:
        """
        Transform contact-force values for display.

        scale="linear" shows raw values.
        scale="log_abs" shows log10(abs(value)) for wide-range positive data.
        scale="signed_log" preserves sign while compressing large magnitudes.
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")

        tiny = np.finfo(float).tiny
        if scale == "linear":
            return matrix, "viridis", "Nodal value"
        if scale == "log_abs":
            return (
                np.log10(np.abs(matrix) + tiny),
                "viridis",
                r"$\log_{10}(|value|)$",
            )
        if scale == "signed_log":
            values = np.sign(matrix) * np.log10(1.0 + np.abs(matrix))
            return values, "coolwarm", r"$sign(value)\log_{10}(1+|value|)$"
        raise ValueError("scale must be 'linear', 'log_abs', or 'signed_log'")

    @staticmethod
    def plot_nodal_heatmap(
        matrix: np.ndarray,
        ax: plt.Axes | None = None,
        *,
        row_labels: list[int] | np.ndarray | None = None,
        title: str = "Nodal values",
        ylabel: str = "Row",
        scale: str = "log_abs",
    ) -> plt.Axes:
        """Plot a 2-D matrix whose columns are node values."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        matrix = np.asarray(matrix, dtype=float)
        values, cmap, cbar_label = CPG._heatmap_values(matrix, scale)

        im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        n_rows, n_nodes = matrix.shape
        ax.set_title(f"{title}  ({n_rows} x {n_nodes})")
        ax.set_xlabel("Node index")
        ax.set_ylabel(ylabel)

        max_ticks = min(n_rows, 12)
        tick_positions = np.unique(
            np.linspace(0, n_rows - 1, max_ticks, dtype=int)
        )
        ax.set_yticks(tick_positions)
        if row_labels is None:
            tick_labels = [str(pos) for pos in tick_positions]
        else:
            tick_labels = [str(int(row_labels[pos])) for pos in tick_positions]
        ax.set_yticklabels(tick_labels, fontsize=8)
        return ax

    def plot_basis_heatmap(
        self,
        ax: plt.Axes | None = None,
        *,
        snapshot_labels: list[int] | np.ndarray | None = None,
        scale: str = "log_abs",
    ) -> plt.Axes:
        """Plot the computed CPG basis as basis-vector index vs node index."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")

        return self.plot_nodal_heatmap(
            self.basis_matrix,
            ax,
            row_labels=snapshot_labels,
            title="CPG basis contact-force values",
            ylabel="Selected snapshot",
            scale=scale,
        )

    @staticmethod
    def plot_snapshot_maps_2d(
        matrix: np.ndarray,
        grid_shape: tuple[int, int],
        *,
        row_labels: list[int] | np.ndarray | None = None,
        title: str = "2D snapshot maps",
        scale: str = "log_abs",
        ncols: int | None = None,
        percentile_clip: tuple[float, float] = (1.0, 99.5),
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot each flattened snapshot as a 2-D grid map."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")

        n_maps, n_nodes = matrix.shape
        grid_rows, grid_cols = grid_shape
        if grid_rows * grid_cols != n_nodes:
            raise ValueError(
                f"grid_shape={grid_shape} is incompatible with {n_nodes} nodes"
            )

        values, cmap, cbar_label = CPG._heatmap_values(matrix, scale)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(finite_values, percentile_clip)
            if vmin == vmax:
                vmin, vmax = float(np.min(finite_values)), float(np.max(finite_values))
            if vmin == vmax:
                vmax = vmin + 1.0

        if ncols is None:
            ncols = min(10, max(1, int(np.ceil(np.sqrt(n_maps)))))
        ncols = max(1, min(ncols, n_maps))
        nrows = int(np.ceil(n_maps / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(2.3 * ncols, 2.0 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        last_im = None
        flat_axes = axes.ravel()
        for k, ax in enumerate(flat_axes):
            if k >= n_maps:
                ax.axis("off")
                continue

            grid_values = values[k].reshape(grid_shape)
            last_im = ax.imshow(
                grid_values,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            label = int(row_labels[k]) if row_labels is not None else k
            ax.set_title(f"basis {k + 1}\nsnap #{label}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{title}  ({n_maps} maps, grid {grid_rows} x {grid_cols})")
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=flat_axes.tolist(), shrink=0.72)
            cbar.set_label(cbar_label)
        return fig, axes

    def plot_basis_maps_2d(
        self,
        grid_shape: tuple[int, int],
        *,
        snapshot_labels: list[int] | np.ndarray | None = None,
        scale: str = "log_abs",
        ncols: int | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot the computed CPG basis snapshots as 2-D grid maps."""
        if self.basis_matrix is None:
            raise RuntimeError("Call compute_phases() first.")

        return self.plot_snapshot_maps_2d(
            self.basis_matrix,
            grid_shape,
            row_labels=snapshot_labels,
            title="CPG output snapshots",
            scale=scale,
            ncols=ncols,
        )

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
            label=f"relative ε = {self.epsilon:g}",
        )
        ax.set_title("CPG convergence")
        ax.set_xlabel("cone size n")
        ax.set_ylabel("Residual $r_n$")
        ax.legend()
        ax.grid(True)
        return ax

    def __repr__(self) -> str:
        R = (
            len(self.basis_matrix)
            if self.basis_matrix is not None
            else len(self._basis)
        )
        status = f"R={R}" if self.basis_matrix is not None else "not yet run"
        return f"CPG(ε={self.epsilon}, N_train={len(self.snapshots)}, {status})"


# -------------------------------------------------------------------------
# Physics dataset  —  contact-force nodal values
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import os, time

    DATA_PATH = "data/physics_data.txt"     # adjust path if needed
    EPSILON = 10e5             # tolerance in Euclidean norm
    OUT_DIR = "."                           # output directory
    GRID_SHAPE = (76, 101)                  # flattened node grid for d = 7676

    # ── 1. Load & orient ──────────────────────────────────────────────────
    # File layout: rows = nodes (d = 7676), cols = snapshots (N = 99).
    # Transpose so snapshots[i] is the i-th contact-force vector λ(μ_i).
    print("Loading data …")
    raw       = np.loadtxt(DATA_PATH)        # (7676, 99)
    snapshots = raw.T                        # (99, 7676)
    N, d      = snapshots.shape
    print(f"  {N} snapshots,  d = {d} nodes")
    if GRID_SHAPE[0] * GRID_SHAPE[1] != d:
        raise ValueError(f"GRID_SHAPE={GRID_SHAPE} is incompatible with d={d}")

    # ── 2. Discard numerically near-zero snapshots (unloaded states)
    norms        = np.linalg.norm(snapshots, axis=1)
    active_mask  = norms > 0.1             # threshold well below any physical value
    active_snaps = snapshots[active_mask]
    active_idx   = np.where(active_mask)[0] # original indices kept for reference
    print(f"  {active_mask.sum()} physically active snapshots retained "
          f"(norm > 0.1),  {(~active_mask).sum()} near-zero discarded")

    # ── 3. Run CPG ────────────────────────────────────────────────────────
    print(f"\nRunning CPG  (ε = {EPSILON:.2e}) …")
    t0  = time.perf_counter()
    cpg = CPG(snapshots=active_snaps, epsilon=EPSILON)
    cpg.compute_phases()
    elapsed = time.perf_counter() - t0

    # Map selected indices back to the full dataset numbering
    global_selected = [int(active_idx[i]) for i in cpg.selected_indices]
    print(cpg)
    print(f"  Selected (global indices): {global_selected}")
    print(f"  Wall time: {elapsed:.1f} s")

    # ── 4. Save basis & indices ───────────────────────────────────────────
    basis_path   = os.path.join(OUT_DIR, "cpg_basis_contact.npy")
    indices_path = os.path.join(OUT_DIR, "cpg_selected_contact.txt")
    np.save(basis_path, cpg.basis_matrix)
    np.savetxt(
        indices_path,
        np.array(global_selected, dtype=int),
        fmt="%d",
        header=(f"CPG selected snapshot indices (0-based, full dataset)\n"
                f"epsilon={EPSILON:.2e}  R={len(cpg._basis)}  d={d}"),
    )
    print(f"\nBasis saved  → {basis_path}")
    print(f"Indices saved → {indices_path}")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # 5a. Snapshot norms — shows the loading ramp
    ax = axes[0, 0]
    ax.semilogy(np.arange(N), norms, "o-", ms=3, color="steelblue")
    ax.fill_between(np.arange(N), norms,
                    where=~active_mask, alpha=0.25, color="red",
                    label="near-zero (excluded)")
    for gi in global_selected:
        ax.axvline(gi, color="orange", lw=1.2, alpha=0.8)
    ax.axvline(global_selected[0], color="orange", lw=1.2, alpha=0.8,
               label="selected snapshots")
    ax.set_xlabel("Snapshot index $i$")
    ax.set_ylabel(r"$\|\lambda(\mu_i)\|$")
    ax.set_title("Snapshot norms — loading ramp")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)

    # 5b. Convergence
    cpg.plot_convergence(axes[0, 1])

    # 5c. Active snapshots as a 2D nodal contact-force matrix.
    # Without node coordinates/connectivity, this is an index-space surface plot.
    CPG.plot_nodal_heatmap(
        active_snaps,
        axes[1, 0],
        row_labels=active_idx,
        title="Active snapshot contact-force values",
        ylabel="Snapshot index",
        scale="log_abs",
    )

    # 5d. CPG result as a 2D matrix: selected basis vs node index.
    cpg.plot_basis_heatmap(
        axes[1, 1],
        snapshot_labels=global_selected,
        scale="log_abs",
    )

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "cpg_contact_force.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved   → {plot_path}")

    # 5e. Output snapshots selected by CPG as true 2D maps.
    fig_basis_2d, _ = cpg.plot_basis_maps_2d(
        GRID_SHAPE,
        snapshot_labels=global_selected,
        scale="log_abs",
        ncols=10,
    )
    basis_2d_path = os.path.join(OUT_DIR, "cpg_output_snapshots_2d.png")
    fig_basis_2d.savefig(basis_2d_path, dpi=160, bbox_inches="tight")
    print(f"2D output snapshots plot saved → {basis_2d_path}")

    # ── 6. Mesh-free input contact-force diagnostics ──────────────────────
    fig_input_diag, diag_axes = plt.subplots(2, 2, figsize=(18, 10))

    CPG.plot_nodal_heatmap(
        active_snaps,
        diag_axes[0, 0],
        row_labels=active_idx,
        title="Input contact-force matrix",
        ylabel="Snapshot index",
        scale="log_abs",
    )

    ax = diag_axes[0, 1]
    active_norms = np.linalg.norm(active_snaps, axis=1)
    ax.semilogy(active_idx, active_norms, "o-", ms=4, color="steelblue")
    for gi in global_selected:
        ax.axvline(gi, color="orange", lw=1.0, alpha=0.7)
    ax.set_title("Input snapshot norms")
    ax.set_xlabel("Snapshot index")
    ax.set_ylabel(r"$\|\lambda(\mu_i)\|$")
    ax.grid(True, which="both", alpha=0.35)

    ax = diag_axes[1, 0]
    node_indices = np.arange(d)
    tiny = np.finfo(float).tiny
    node_max = np.max(np.abs(active_snaps), axis=0)
    node_mean = np.mean(np.abs(active_snaps), axis=0)
    ax.semilogy(node_indices, node_max + tiny, lw=1.0, label="max over snapshots")
    ax.semilogy(node_indices, node_mean + tiny, lw=1.0, label="mean over snapshots")
    ax.set_title("Per-node force envelope")
    ax.set_xlabel("Node index")
    ax.set_ylabel("Absolute contact-force value")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.35)

    ax = diag_axes[1, 1]
    envelope_2d = node_max.reshape(GRID_SHAPE)
    im = ax.imshow(
        np.log10(envelope_2d + tiny),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    fig_input_diag.colorbar(im, ax=ax, label=r"$\log_{10}(|max force|)$")
    ax.set_title(f"2D max-force envelope ({GRID_SHAPE[0]} x {GRID_SHAPE[1]})")
    ax.set_xlabel("Grid x index")
    ax.set_ylabel("Grid y index")

    plt.tight_layout()
    input_diag_path = os.path.join(OUT_DIR, "cpg_input_contact_diagnostics.png")
    fig_input_diag.savefig(input_diag_path, dpi=150, bbox_inches="tight")
    print(f"Input diagnostics plot saved → {input_diag_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
