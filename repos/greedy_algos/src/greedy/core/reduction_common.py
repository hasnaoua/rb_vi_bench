from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize, nnls

try:
    import cvxopt

    cvxopt.solvers.options["show_progress"] = False
except ImportError:  # cvxopt is optional; mCPG falls back to scipy/NNLS.
    cvxopt = None

def row_map(function, rows):
    """
    Apply ``function`` to each row of ``rows``, in order.

    Deliberately serial. This used to dispatch through ``joblib.Parallel``, but
    each per-row NNLS solve costs microseconds while joblib pickles the basis
    matrix and pays inter-process latency per row, so the parallel version lost
    on every workload here -- measured serial vs joblib, identical results:
    hertz_a (31x51) 264x faster serial, hertz (81x47) 21x, membrane (125x885)
    5.5x, and physics-scale (96x7676) 6.6x. Re-parallelising is only worth
    revisiting if a single projection ever becomes expensive enough to dominate
    that overhead; measure before assuming it does.
    """
    return [function(row) for row in rows]


def _apply_publication_style() -> None:
    """
    Repo-wide figure defaults, applied once at import time since every
    plotting script already imports this module. Deliberately leaves the
    right spine alone (unlike a typical "remove top+right spines" preset)
    because several plots (e.g. convergence-vs-angle) rely on a visible
    right spine for a twin y-axis.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.2,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


_apply_publication_style()


# Fixed per-method colors so every comparison figure across the repo (lambda,
# physics, synthetic) uses the same visual identity for CPG / ADG / mCPG.
METHOD_COLORS: dict[str, str] = {
    "CPG": "#2563eb",
    "Angular Defect Greedy": "#f59e0b",
    "ADG": "#f59e0b",
    "mCPG": "#059669",
}
_FALLBACK_COLOR_CYCLE = ["#7c3aed", "#dc2626", "#0891b2", "#a16207"]


def method_color(method: str) -> str:
    """Fixed color for a known method name; deterministic fallback otherwise."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    index = abs(hash(method)) % len(_FALLBACK_COLOR_CYCLE)
    return _FALLBACK_COLOR_CYCLE[index]


def vector_projection_angle(
    vector: np.ndarray,
    projection: np.ndarray,
    *,
    zero_tol: float = 1e-14,
) -> float:
    """
    Angle in radians between ``vector`` and its cone projection.

    Shared by CPG, AngularDefectGreedy, and mCPG so all three report the
    same "how novel is this candidate direction" diagnostic: 0 means the
    candidate was already fully representable by the current cone, pi/2
    means it is orthogonal to it (maximally novel).
    """
    norm_vector = float(np.linalg.norm(vector))
    if norm_vector <= zero_tol:
        return 0.0

    norm_projection = float(np.linalg.norm(projection))
    if norm_projection <= zero_tol:
        return float(np.pi / 2.0)

    cos_theta = float(np.dot(vector, projection) / (norm_vector * norm_projection))
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


DEFAULT_DATASET = Path("results/lambda/dataset/lambda_dataset.npz")
DEFAULT_CPG_BASIS = Path("results/lambda/cpg/cpg_basis.npy")


def solve_nonnegative_least_squares(
    system: np.ndarray,
    vector: np.ndarray,
    *,
    maxiter: int | None = None,
) -> np.ndarray:
    system = np.asarray(system, dtype=float)
    vector = np.asarray(vector, dtype=float)
    if system.ndim != 2:
        raise ValueError(f"system must be 2-D, got shape {system.shape}")
    if vector.ndim != 1:
        raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
    if system.shape[0] != vector.shape[0]:
        raise ValueError(
            f"system rows ({system.shape[0]}) must match vector length ({vector.shape[0]})"
        )
    if system.shape[1] == 0:
        return np.zeros(0, dtype=float)

    if maxiter is None:
        maxiter = max(1000, 10 * system.shape[1])

    try:
        coeffs, _ = nnls(system, vector, maxiter=maxiter)
        return np.asarray(coeffs, dtype=float)
    except RuntimeError:
        pass

    try:
        result = lsq_linear(
            system,
            vector,
            bounds=(0.0, np.inf),
            max_iter=maxiter,
            lsmr_tol="auto",
        )
        coeffs = np.asarray(result.x, dtype=float)
        if coeffs.size == system.shape[1]:
            return np.maximum(coeffs, 0.0)
    except Exception:
        pass

    coeffs, _, _, _ = np.linalg.lstsq(system, vector, rcond=None)
    return np.maximum(np.asarray(coeffs, dtype=float), 0.0)


def solve_cone_shift_projection(
    system: np.ndarray,
    vector: np.ndarray,
    *,
    upper_tol: float = 1e-9,
    maxiter: int | None = None,
    cone_system: np.ndarray | None = None,
    cone_vector: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve  min_{c >= 0}  ||system @ c - vector||_2
           s.t.           cone_system @ c <= cone_vector + upper_tol   (elementwise)

    This is the constrained cone-shift projection used by the mCPG algorithm
    (Niakh, Drouet, Ehrlacher & Ern, 2022, Algorithm 2):
    Upsilon_r = argmin_{Upsilon in K_{r-1} \\cap (theta_qr - W^+)} ||theta_qr - Upsilon||,
    where ``system`` holds the current cone generators as columns and
    ``vector`` is theta_qr. The elementwise upper bound keeps
    theta_qr - Upsilon inside the positive cone W^+ (here: the nonnegative
    orthant), unlike plain NNLS which only constrains the coefficients.

    ``cone_system``/``cone_vector`` default to ``system``/``vector`` and only
    differ when the objective is measured in a non-Euclidean inner product.
    Minimising ||.||_W is done by passing the Cholesky-transformed pair
    (U @ A, U @ theta) as system/vector, but "theta - Upsilon stays in W^+" is
    an elementwise statement about the *physical* dual coefficients, and
    elementwise-nonnegative is not preserved by U. Passing the untransformed
    (A, theta) here keeps that membership test in the coordinates where it is
    actually meaningful.

    Solved as a QP with cvxopt when available; falls back to scipy SLSQP,
    and finally to plain NNLS (ignoring the upper bound) if both fail.
    """
    system = np.asarray(system, dtype=float)
    vector = np.asarray(vector, dtype=float)
    if system.ndim != 2:
        raise ValueError(f"system must be 2-D, got shape {system.shape}")
    if vector.ndim != 1:
        raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
    if system.shape[0] != vector.shape[0]:
        raise ValueError(
            f"system rows ({system.shape[0]}) must match vector length ({vector.shape[0]})"
        )
    if system.shape[1] == 0:
        return np.zeros(0, dtype=float)

    constraint_system = system if cone_system is None else np.asarray(cone_system, dtype=float)
    constraint_vector = vector if cone_vector is None else np.asarray(cone_vector, dtype=float)
    if constraint_system.ndim != 2:
        raise ValueError(f"cone_system must be 2-D, got shape {constraint_system.shape}")
    if constraint_vector.ndim != 1:
        raise ValueError(f"cone_vector must be 1-D, got shape {constraint_vector.shape}")
    if constraint_system.shape[1] != system.shape[1]:
        raise ValueError(
            f"cone_system columns ({constraint_system.shape[1]}) must match "
            f"system columns ({system.shape[1]})"
        )
    if constraint_system.shape[0] != constraint_vector.shape[0]:
        raise ValueError(
            f"cone_system rows ({constraint_system.shape[0]}) must match "
            f"cone_vector length ({constraint_vector.shape[0]})"
        )

    k = system.shape[1]

    if cvxopt is not None:
        try:
            gram = system.T @ system
            ridge = 1e-10 * (float(np.trace(gram)) / k if k > 0 else 1.0)
            gram = gram + ridge * np.eye(k)
            linear = -(system.T @ vector)
            G = np.vstack([-np.eye(k), constraint_system])
            h = np.concatenate([np.zeros(k), constraint_vector + upper_tol])

            solution = cvxopt.solvers.qp(
                cvxopt.matrix(gram),
                cvxopt.matrix(linear),
                cvxopt.matrix(G),
                cvxopt.matrix(h),
            )
            if solution["status"] == "optimal":
                coeffs = np.asarray(solution["x"], dtype=float).reshape(-1)
                return np.maximum(coeffs, 0.0)
        except Exception:
            pass

    try:
        def objective(c: np.ndarray) -> float:
            residual = system @ c - vector
            return 0.5 * float(residual @ residual)

        def gradient(c: np.ndarray) -> np.ndarray:
            return system.T @ (system @ c - vector)

        result = minimize(
            objective,
            np.zeros(k),
            jac=gradient,
            bounds=Bounds(np.zeros(k), np.full(k, np.inf)),
            constraints=[
                LinearConstraint(constraint_system, -np.inf, constraint_vector + upper_tol)
            ],
            method="SLSQP",
            options={"maxiter": 200 if maxiter is None else maxiter},
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            return np.maximum(np.asarray(result.x, dtype=float), 0.0)
    except Exception:
        pass

    return solve_nonnegative_least_squares(system, vector, maxiter=maxiter)


def gram_cholesky(gram: np.ndarray, *, jitter: float = 1e-12) -> np.ndarray:
    """
    Upper-triangular U with ``gram = U.T @ U``, so that ``||x||_W = ||U @ x||_2``.

    This is what lets the Euclidean CPG / AngularDefectGreedy / mCPG in this
    package run in a non-Euclidean dual inner product without touching their
    selection logic: a cone combination transforms as U @ (A c) = (U A) c with
    the same c >= 0, so running the plain algorithms on U-transformed snapshots
    is exactly the W-norm greedy. ``jitter`` is scaled by the Gram's own
    magnitude, since mass matrices here have entries ~1e-4 and a fixed absolute
    nudge would swamp them.
    """
    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram must be square 2-D, got shape {gram.shape}")
    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-10 * max(1.0, float(np.abs(gram).max()))):
        raise ValueError("gram must be symmetric")

    scale = float(np.trace(gram)) / gram.shape[0] if gram.shape[0] else 1.0
    return la.cholesky(gram + jitter * max(scale, 1.0) * np.eye(gram.shape[0]), lower=False)


def w_norm(vectors: np.ndarray, gram: np.ndarray) -> np.ndarray:
    """Row-wise W-norm sqrt(x.T @ gram @ x) for a (n, d) stack of vectors."""
    vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
    quadratic = np.einsum("ij,jk,ik->i", vectors, np.asarray(gram, dtype=float), vectors)
    return np.sqrt(np.maximum(quadratic, 0.0))


def load_lambda_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    data = np.load(path, allow_pickle=False)
    snapshots = np.asarray(data["snapshots"], dtype=float)
    radii = np.asarray(data["radii"], dtype=float)

    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got shape {snapshots.shape}")
    if radii.ndim != 1 or radii.size != snapshots.shape[0]:
        raise ValueError(
            f"radii must be 1-D with {snapshots.shape[0]} values, got {radii.shape}"
        )
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshots contain non-finite values")
    if not np.all(np.isfinite(radii)):
        raise ValueError("radii contain non-finite values")

    return snapshots, radii, str(path)


def infer_component_count(path: Path = DEFAULT_CPG_BASIS, fallback: int = 10) -> int:
    if not path.exists():
        return fallback
    basis = np.load(path, allow_pickle=False)
    if basis.ndim != 2 or basis.shape[0] == 0:
        return fallback
    return int(basis.shape[0])


def nice_indices(length: int, max_count: int) -> np.ndarray:
    if length <= 0 or max_count <= 0:
        return np.array([], dtype=int)
    if length <= max_count:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_count, dtype=int))


def parameter_train_test_split(
    parameters: np.ndarray,
    *,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows by sorted parameter values for offline/online validation.

    The first and last parameter values stay in the training set so that the
    test set measures interpolation at unseen parameters, not extrapolation.
    """
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    n_rows = parameters.size
    if n_rows == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if test_fraction <= 0.0 or n_rows < 3:
        return np.arange(n_rows, dtype=int), np.array([], dtype=int)
    if test_fraction >= 1.0:
        raise ValueError("test_fraction must be < 1.0")

    order = np.argsort(parameters, kind="mergesort")
    interior_positions = np.arange(1, n_rows - 1, dtype=int)
    if interior_positions.size == 0:
        return order.copy(), np.array([], dtype=int)

    requested = int(round(test_fraction * n_rows))
    test_count = min(max(1, requested), interior_positions.size)
    raw_positions = np.linspace(0, interior_positions.size - 1, test_count)
    chosen_positions = list(
        dict.fromkeys(interior_positions[np.round(raw_positions).astype(int)].tolist())
    )

    if len(chosen_positions) < test_count:
        chosen_set = set(chosen_positions)
        for position in interior_positions:
            if int(position) not in chosen_set:
                chosen_positions.append(int(position))
                chosen_set.add(int(position))
            if len(chosen_positions) == test_count:
                break

    test_sorted_positions = np.array(sorted(chosen_positions), dtype=int)
    test_indices = np.sort(order[test_sorted_positions])
    train_mask = np.ones(n_rows, dtype=bool)
    train_mask[test_indices] = False
    train_indices = np.flatnonzero(train_mask)
    return train_indices.astype(int), test_indices.astype(int)


def tensor_grid_interior_split(
    mu_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hold out the strict interior of a tensor-product parameter grid.

    For a multi-dimensional parameter (e.g. the membrane's 5x5x5 grid over
    (radius, cx, cy)), splitting on a single component is degenerate: that
    component takes only a handful of distinct values, so a sorted split neither
    interpolates in it nor controls the others. Instead this keeps the whole
    boundary shell of the grid (any point with a component at its min or max)
    for training and holds out every strictly interior point, which makes the
    test set a genuine interpolation check in *all* parameter directions at
    once.

    Only meaningful for >= 2 parameter dimensions; a 1-D parameter would leave
    all but two points held out, so use ``parameter_train_test_split`` there.
    """
    mu_samples = np.atleast_2d(np.asarray(mu_samples, dtype=float))
    if mu_samples.shape[1] < 2:
        raise ValueError(
            "tensor_grid_interior_split needs a multi-dimensional parameter; "
            "use parameter_train_test_split for a scalar parameter"
        )

    interior = np.ones(mu_samples.shape[0], dtype=bool)
    for column in range(mu_samples.shape[1]):
        values = np.unique(mu_samples[:, column])
        position = np.searchsorted(values, mu_samples[:, column])
        interior &= (position > 0) & (position < values.size - 1)

    test_indices = np.flatnonzero(interior)
    train_indices = np.flatnonzero(~interior)
    if train_indices.size == 0:
        raise ValueError("tensor grid split produced an empty training set")
    return train_indices.astype(int), test_indices.astype(int)


def split_by_displacement_intensity(
    parameters: np.ndarray,
    *,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows by increasing imposed u_z intensity.

    The lowest-intensity snapshots are used for training and the highest-intensity
    snapshots are held out for testing, so both methods are evaluated on a
    strictly separate regime.
    """
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    n_rows = parameters.size
    if n_rows == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if test_fraction <= 0.0:
        return np.arange(n_rows, dtype=int), np.array([], dtype=int)
    if test_fraction >= 1.0:
        raise ValueError("test_fraction must be < 1.0")

    order = np.argsort(parameters, kind="mergesort")
    test_count = max(1, min(n_rows - 1, int(round(test_fraction * n_rows))))
    train_count = n_rows - test_count
    if train_count <= 0:
        raise ValueError("not enough rows to form a non-empty training set")

    train_indices = np.sort(order[:train_count])
    test_indices = np.sort(order[train_count:])
    return train_indices.astype(int), test_indices.astype(int)


def validate_train_test_split(
    row_count: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    require_test: bool = False,
    allow_overlap: bool = False,
) -> None:
    """Raise if a train/test split can leak held-out snapshots into fitting."""
    if row_count < 0:
        raise ValueError("row_count must be non-negative")

    train = np.asarray(train_indices, dtype=int).reshape(-1)
    test = np.asarray(test_indices, dtype=int).reshape(-1)

    if train.size == 0:
        raise ValueError("training split must contain at least one row")
    if require_test and test.size == 0:
        raise ValueError("test split must contain at least one held-out row")

    for name, indices in (("train", train), ("test", test)):
        if indices.size == 0:
            continue
        if np.min(indices) < 0 or np.max(indices) >= row_count:
            raise ValueError(
                f"{name} split contains rows outside [0, {row_count - 1}]"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} split contains duplicate rows")

    overlap = np.intersect1d(train, test, assume_unique=False)
    if overlap.size and not allow_overlap:
        preview = ", ".join(str(int(row)) for row in overlap[:10])
        suffix = "..." if overlap.size > 10 else ""
        raise ValueError(f"train/test split overlaps at rows: {preview}{suffix}")


def split_by_displacement_gap(
    parameters: np.ndarray,
    *,
    train_ranges: list[tuple[float, float]] = [(0.18, 0.295), (0.3, 1.01)],
    gap_range: tuple[float, float] = (0.295, 0.3),
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Split displacements by explicit training ranges and a small test gap."""
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    n_rows = parameters.size
    if n_rows == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    train_mask = np.zeros(n_rows, dtype=bool)
    for lower, upper in train_ranges:
        train_mask |= (parameters >= lower - atol) & (parameters <= upper + atol)

    gap_lower, gap_upper = gap_range
    gap_mask = (parameters > gap_lower + atol) & (parameters < gap_upper - atol)
    train_indices = np.flatnonzero(train_mask)
    if np.any(gap_mask):
        test_indices = np.flatnonzero(gap_mask)
    else:
        test_indices = np.arange(n_rows, dtype=int)

    if train_indices.size == 0:
        raise ValueError("No training displacements matched the requested ranges.")

    return train_indices.astype(int), test_indices.astype(int)


def split_by_training_values(
    parameters: np.ndarray,
    training_values: np.ndarray,
    *,
    parameter_shift: float = 0.0,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows by an explicit discrete training set.

    ``parameter_shift`` is applied before matching. This is useful for datasets
    whose folder names store a shifted parameter, such as FEM_SOLS where the
    paper parameter is represented by ``rad - 0.65`` in the local archive.
    """
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    training_values = np.asarray(training_values, dtype=float).reshape(-1)
    if parameters.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if training_values.size == 0:
        raise ValueError("training_values must not be empty")

    effective_parameters = parameters + float(parameter_shift)
    train_mask = np.zeros(parameters.size, dtype=bool)
    for value in training_values:
        train_mask |= np.isclose(effective_parameters, value, atol=atol, rtol=0.0)

    train_indices = np.flatnonzero(train_mask)
    if train_indices.size == 0:
        raise ValueError(
            "No parameters matched the requested training set. "
            f"parameter range after shift is "
            f"[{float(np.min(effective_parameters)):.12g}, "
            f"{float(np.max(effective_parameters)):.12g}]"
        )
    test_indices = np.flatnonzero(~train_mask)
    return train_indices.astype(int), test_indices.astype(int)


def fem_sols_training_parameters() -> np.ndarray:
    """Return the FEM_SOLS paper training set: 0.15 + 0.01 i, 0 <= i <= 30."""
    return 0.15 + 0.01 * np.arange(31, dtype=float)


def fem_sols_train_test_split(
    radii: np.ndarray,
    *,
    radius_to_parameter_shift: float = -0.65,
    atol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split FEM_SOLS rows with the paper grid.

    Local FEM_SOLS folders are named ``rad=0.8`` ... while the paper statement
    gives the parameter range as 0.15 ... 0.45. The default shift maps
    ``rad=0.8`` to parameter 0.15.
    """
    return split_by_training_values(
        radii,
        fem_sols_training_parameters(),
        parameter_shift=radius_to_parameter_shift,
        atol=atol,
    )


def split_membership(
    row_count: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> np.ndarray:
    labels = np.full(row_count, "none", dtype=object)
    train = np.asarray(train_indices, dtype=int).reshape(-1)
    test = np.asarray(test_indices, dtype=int).reshape(-1)
    labels[train] = "train"
    overlap_mask = np.isin(test, train)
    if test.size:
        labels[test[~overlap_mask]] = "test"
        labels[test[overlap_mask]] = "train+test"
    return labels


def residual_metrics(
    snapshots: np.ndarray,
    reconstructions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.linalg.norm(snapshots - reconstructions, axis=1)
    norms = np.linalg.norm(snapshots, axis=1)
    relative = np.divide(
        residuals,
        norms,
        out=np.zeros_like(residuals),
        where=norms > 0.0,
    )
    return residuals, relative


def max_projection_error_history(
    snapshots: np.ndarray,
    basis: np.ndarray,
    *,
    relative: bool = False,
    norm_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return max projection error for each basis prefix K_n.

    By default this is e_n = max_x ||x - proj_Kn(x)||_2. With relative=True,
    this is max_x ||x - proj_Kn(x)||_2 / max(||x||_2, norm_floor).
    """
    snapshots = np.asarray(snapshots, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if norm_floor < 0.0:
        raise ValueError("norm_floor must be non-negative")
    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got shape {snapshots.shape}")
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D, got shape {basis.shape}")
    if basis.shape[0] == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if basis.shape[1] != snapshots.shape[1]:
        raise ValueError(
            f"basis width ({basis.shape[1]}) must match snapshots width ({snapshots.shape[1]})"
        )

    errors: list[float] = []
    denominators = np.maximum(np.linalg.norm(snapshots, axis=1), float(norm_floor))
    for size in range(1, basis.shape[0] + 1):
        prefix = basis[:size]
        if size == 1:
            vector = prefix[0]
            vv = float(vector @ vector)
            if vv == 0.0:
                residuals = np.linalg.norm(snapshots, axis=1)
            else:
                coeffs = np.maximum(0.0, snapshots @ vector / vv)
                projections = np.outer(coeffs, vector)
                residuals = np.linalg.norm(snapshots - projections, axis=1)
        else:
            system = prefix.T
            residuals = np.array(
                [
                    np.linalg.norm(
                        snapshot
                        - system @ solve_nonnegative_least_squares(system, snapshot)
                    )
                    for snapshot in snapshots
                ],
                dtype=float,
            )
        if relative:
            values = np.divide(
                residuals,
                denominators,
                out=np.zeros_like(residuals),
                where=denominators > 0.0,
            )
        else:
            values = residuals
        errors.append(float(np.max(values)) if values.size else 0.0)

    iterations = np.arange(1, basis.shape[0] + 1, dtype=float)
    return iterations, np.asarray(errors, dtype=float)


def _normalise_error_series(
    series: tuple[np.ndarray, np.ndarray] | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(series, tuple):
        iterations = np.asarray(series[0], dtype=float).reshape(-1)
        errors = np.asarray(series[1], dtype=float).reshape(-1)
    else:
        errors = np.asarray(series, dtype=float).reshape(-1)
        iterations = np.arange(1, errors.size + 1, dtype=float)

    if iterations.size != errors.size:
        raise ValueError(
            f"iteration count ({iterations.size}) must match error count ({errors.size})"
        )

    finite = np.isfinite(iterations) & np.isfinite(errors)
    iterations = iterations[finite]
    errors = errors[finite]
    if iterations.size:
        order = np.argsort(iterations, kind="mergesort")
        iterations = iterations[order]
        errors = errors[order]
    return iterations, errors


def _compute_gain_series(
    errors: np.ndarray | list[float],
    gain_mode: str,
    *,
    tiny: float | None = None,
) -> np.ndarray:
    errors = np.asarray(errors, dtype=float).reshape(-1)
    if errors.size <= 1:
        return np.array([], dtype=float)

    if tiny is None:
        tiny = np.finfo(float).tiny

    if gain_mode == "absolute":
        return errors[:-1] - errors[1:]
    if gain_mode == "relative":
        before = np.maximum(np.abs(errors[:-1]), tiny)
        return (errors[:-1] - errors[1:]) / before
    if gain_mode == "log10_ratio":
        before = np.maximum(np.abs(errors[:-1]), tiny)
        after = np.maximum(np.abs(errors[1:]), tiny)
        return np.log10(before) - np.log10(after)
    raise ValueError("gain_mode must be 'absolute', 'relative', or 'log10_ratio'")


def write_error_gain_csv(
    histories: dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray | list[float]],
    output_path: Path,
) -> None:
    """
    Write gains between consecutive errors e_n = max ||x - proj_Kn(x)||.

    The absolute gain from iteration a to b is e_a - e_b. The relative gain is
    normalised by e_a and left as zero when e_a is zero.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "from_iteration",
                "to_iteration",
                "error_before",
                "error_after",
                "log10_error_before",
                "log10_error_after",
                "absolute_gain",
                "relative_gain",
                "log10_gain",
            ]
        )
        for method, series in histories.items():
            iterations, errors = _normalise_error_series(series)
            for index in range(1, errors.size):
                before = float(errors[index - 1])
                after = float(errors[index])
                gain = before - after
                relative_gain = gain / before if before != 0.0 else 0.0
                log_before = float(np.log10(max(abs(before), np.finfo(float).tiny)))
                log_after = float(np.log10(max(abs(after), np.finfo(float).tiny)))
                log_gain = log_before - log_after
                writer.writerow(
                    [
                        method,
                        f"{float(iterations[index - 1]):.12g}",
                        f"{float(iterations[index]):.12g}",
                        f"{before:.18e}",
                        f"{after:.18e}",
                        f"{log_before:.18e}",
                        f"{log_after:.18e}",
                        f"{gain:.18e}",
                        f"{relative_gain:.18e}",
                        f"{log_gain:.18e}",
                    ]
                )


def plot_convergence_with_angle(
    residual_history: list[float] | np.ndarray,
    epsilon: float,
    output_path: Path,
    *,
    basis_sizes: list[int] | np.ndarray | None = None,
    angle_history_deg: list[float] | np.ndarray | None = None,
    angle_basis_sizes: list[int] | np.ndarray | None = None,
    title: str = "Convergence",
    method_name: str | None = None,
    residual_label: str = "max projection residual $r_n$",
    angle_label: str = r"candidate angle to cone $\theta_n$ (deg)",
    threshold_label: str | None = None,
) -> None:
    """
    Publication-style convergence plot: max residual r_n on a log left axis,
    and (optionally) the angle between each accepted candidate and the cone
    it was added to on a linear right axis, sharing the same "cone size n"
    x-axis. The angle series is the "how novel was this direction" signal
    already used to pick candidates in AngularDefectGreedy and now also
    reported as a diagnostic by CPG and mCPG (see
    ``reduction_common.vector_projection_angle``); angle=90 deg means the
    candidate was orthogonal to the existing cone, 0 deg means it added no
    new information.

    ``angle_basis_sizes`` lets the angle series use its own x-positions when
    it isn't 1:1 with ``residual_history`` (e.g. AngularDefectGreedy's
    initial pair has no associated "angle to cone").
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    residual_history = np.asarray(residual_history, dtype=float)
    color = method_color(method_name) if method_name else "#2563eb"

    fig, ax_residual = plt.subplots(figsize=(9.5, 5.2))
    if residual_history.size:
        iterations = (
            np.asarray(basis_sizes, dtype=float)
            if basis_sizes is not None and len(basis_sizes)
            else np.arange(1, residual_history.size + 1, dtype=float)
        )
        ax_residual.semilogy(
            iterations,
            np.maximum(residual_history, np.finfo(float).tiny),
            marker="o",
            color=color,
            label=residual_label,
        )
    ax_residual.axhline(
        epsilon,
        color="#c2410c",
        linestyle="--",
        linewidth=1.3,
        label=threshold_label or f"epsilon = {epsilon:g}",
    )
    ax_residual.set_xlabel("cone size n")
    ax_residual.set_ylabel(residual_label, color=color)
    ax_residual.tick_params(axis="y", labelcolor=color)
    ax_residual.grid(True, which="both", alpha=0.28)
    ax_residual.set_title(title)

    lines, labels = ax_residual.get_legend_handles_labels()

    angle_history_deg = (
        np.asarray(angle_history_deg, dtype=float) if angle_history_deg is not None else None
    )
    if angle_history_deg is not None and angle_history_deg.size:
        angle_color = "#7c3aed"
        ax_angle = ax_residual.twinx()
        angle_x = (
            np.asarray(angle_basis_sizes, dtype=float)
            if angle_basis_sizes is not None and len(angle_basis_sizes)
            else np.arange(1, angle_history_deg.size + 1, dtype=float)
        )
        ax_angle.plot(
            angle_x,
            angle_history_deg,
            marker="^",
            linestyle=":",
            color=angle_color,
            alpha=0.9,
            label=angle_label,
        )
        ax_angle.set_ylabel(angle_label, color=angle_color)
        ax_angle.tick_params(axis="y", labelcolor=angle_color)
        ax_angle.set_ylim(-2.0, 95.0)
        angle_lines, angle_labels = ax_angle.get_legend_handles_labels()
        lines += angle_lines
        labels += angle_labels

    ax_residual.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def basis_condition_number(basis: np.ndarray) -> float:
    """2-norm condition number (sigma_max/sigma_min) of a basis matrix.

    Returns nan for an empty basis. A high value flags near-parallel or
    redundant generators, which makes the NNLS projection subproblem
    ill-posed even though the greedy error/angle criteria are satisfied.
    """
    basis = np.asarray(basis, dtype=float)
    if basis.size == 0 or min(basis.shape) == 0:
        return float("nan")
    return float(np.linalg.cond(basis))


def write_angle_history_csv(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Write the (component_count, angle_deg) series behind plot_angle_vs_components."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "component_count", "angle_deg"])
        for method, (sizes, angles_deg) in histories.items():
            for n, angle_deg in zip(np.asarray(sizes), np.asarray(angles_deg)):
                writer.writerow([method, int(n), f"{float(angle_deg):.12g}"])


def plot_angle_vs_components(
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    title: str = "Selected-snapshot angle vs. component count",
    xlabel: str = "component count n",
) -> None:
    """
    Compare methods by the angle (degrees) each newly selected snapshot made
    with the cone it joined, at the component count reached right after it
    was added. 0 deg means the pick added no new direction, 90 deg means it
    was orthogonal to everything already in the cone.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    plotted = False

    for method, (sizes, angles_deg) in histories.items():
        sizes = np.asarray(sizes, dtype=float)
        angles_deg = np.asarray(angles_deg, dtype=float)
        if angles_deg.size == 0:
            continue
        plotted = True
        ax.plot(
            sizes,
            angles_deg,
            marker="o",
            linewidth=1.5,
            markersize=3.6,
            color=method_color(method),
            label=method,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"angle to prior cone $\theta_n$ (deg)")
    ax.set_ylim(-2.0, 95.0)
    ax.grid(True, alpha=0.28)
    if plotted:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No angle history", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_gain(
    histories: dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray | list[float]],
    output_path: Path,
    *,
    title: str,
    xlabel: str = "iteration n / cone size",
    gain_mode: str = "relative",
    error_ylabel: str = r"$e_n=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    tiny = np.finfo(float).tiny
    plotted = False

    for method, series in histories.items():
        iterations, errors = _normalise_error_series(series)
        if errors.size == 0:
            continue

        plotted = True
        color = method_color(method)
        axes[0].semilogy(
            iterations,
            np.maximum(errors, tiny),
            marker="o",
            linewidth=1.5,
            markersize=3.6,
            color=color,
            label=method,
        )

        if errors.size > 1:
            gains = _compute_gain_series(errors, gain_mode, tiny=tiny)
            axes[1].plot(
                iterations[1:],
                gains,
                marker="o",
                linewidth=1.5,
                markersize=3.6,
                color=color,
                label=method,
            )

    axes[0].set_title(title)
    axes[0].set_ylabel(error_ylabel)
    axes[0].grid(True, which="both", alpha=0.28)

    axes[1].axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel(xlabel)
    if gain_mode == "absolute":
        axes[1].set_ylabel(r"gain $e_{n-1}-e_n$")
    elif gain_mode == "relative":
        axes[1].set_ylabel(r"relative gain $(e_{n-1}-e_n)/e_{n-1}$")
    else:
        axes[1].set_ylabel(r"log gain $\log_{10}(e_{n-1}/e_n)$")
    axes[1].grid(True, alpha=0.28)

    if plotted:
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No error history", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_residuals_csv(
    path: Path,
    radii: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "radius", "residual", "relative_residual"])
        for row, (radius, residual, relative) in enumerate(
            zip(radii, residuals, relative_residuals)
        ):
            writer.writerow(
                [
                    row,
                    f"{radius:.12g}",
                    f"{residual:.18e}",
                    f"{relative:.18e}",
                ]
            )


def write_report(path: Path, title: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([title, "", *rows]) + "\n", encoding="utf-8")


def plot_basis_profiles(
    basis: np.ndarray,
    output_path: Path,
    *,
    title: str,
    ylabel: str = "basis value",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = basis.shape[0]
    height = max(4.0, min(14.0, 1.35 * max(1, count)))
    fig, axes = plt.subplots(count, 1, figsize=(11, height), sharex=True, squeeze=False)
    y_min = float(np.min(basis)) if basis.size else 0.0
    y_max = float(np.max(basis)) if basis.size else 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    for row, ax in enumerate(axes.ravel()):
        ax.plot(basis[row], linewidth=1.2, color="#2563eb")
        ax.set_ylabel(ylabel)
        ax.set_title(f"component {row + 1}", fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.28)

    axes[-1, 0].set_xlabel("stored lambda/contact index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basis_heatmap(
    basis: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(4.0, min(9.0, 2.0 + 0.34 * max(1, basis.shape[0])))
    fig, ax = plt.subplots(figsize=(11, height))
    image = ax.imshow(basis, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("stored lambda/contact index")
    ax.set_ylabel("component")
    ax.set_yticks(np.arange(basis.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(basis.shape[0])])
    fig.colorbar(image, ax=ax, pad=0.012, label="basis value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_coefficients(
    coefficients: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for component in range(coefficients.shape[1]):
        ax.plot(
            radii,
            coefficients[:, component],
            marker="o",
            linewidth=1.35,
            markersize=3.4,
            label=f"component {component + 1}",
        )
    ax.set_title(title)
    ax.set_xlabel("radius")
    ax.set_ylabel("coefficient")
    ax.grid(True, alpha=0.28)
    if coefficients.shape[1] <= 12:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residuals_by_radius(
    radii: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(radii, residuals, marker="o", linewidth=1.5, color="#2563eb")
    axes[0].set_title(title)
    axes[0].set_ylabel("absolute residual")
    axes[0].grid(True, alpha=0.28)

    axes[1].plot(radii, relative_residuals, marker="o", linewidth=1.5, color="#0f766e")
    axes[1].set_xlabel("radius")
    axes[1].set_ylabel("relative residual")
    axes[1].grid(True, alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_examples(
    snapshots: np.ndarray,
    reconstructions: np.ndarray,
    radii: np.ndarray,
    residuals: np.ndarray,
    output_path: Path,
    *,
    title: str,
    max_examples: int = 6,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = nice_indices(snapshots.shape[0], max_examples)
    if rows.size == 0:
        return

    rows = np.unique(np.append(rows, int(np.argmax(residuals))))
    if rows.size > max_examples:
        ranked = rows[np.argsort(residuals[rows])[::-1]]
        rows = np.array(sorted(ranked[:max_examples]), dtype=int)

    ncols = min(3, rows.size)
    nrows = int(np.ceil(rows.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.35 * nrows),
        squeeze=False,
    )
    x = np.arange(snapshots.shape[1])

    for axis_index, ax in enumerate(axes.ravel()):
        if axis_index >= rows.size:
            ax.axis("off")
            continue

        row = int(rows[axis_index])
        ax.plot(x, snapshots[row], color="#17202a", linewidth=1.55, label="input")
        ax.plot(
            x,
            reconstructions[row],
            color="#0f766e",
            linewidth=1.45,
            linestyle="--",
            label="reconstruction",
        )
        ax.fill_between(
            x,
            snapshots[row],
            reconstructions[row],
            color="#c2410c",
            alpha=0.16,
            linewidth=0,
        )
        ax.set_title(f"row {row}, r={radii[row]:.5g}, residual={residuals[row]:.2e}")
        ax.set_xlabel("stored lambda/contact index")
        ax.set_ylabel("lambda")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(
    snapshots: np.ndarray,
    radii: np.ndarray,
    basis: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    output_path: Path,
    *,
    method_name: str,
    dataset_label: str = "lambda snapshots",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

    norms = np.linalg.norm(snapshots, axis=1)
    axes[0, 0].plot(radii, norms, marker="o", linewidth=1.4)
    axes[0, 0].set_title("Input snapshot norms")
    axes[0, 0].set_xlabel("radius")
    axes[0, 0].set_ylabel("L2 norm")
    axes[0, 0].grid(True, alpha=0.28)

    axes[0, 1].plot(radii, residuals, marker="o", linewidth=1.4, color="#2563eb")
    axes[0, 1].set_title("Absolute residual")
    axes[0, 1].set_xlabel("radius")
    axes[0, 1].set_ylabel("L2 residual")
    axes[0, 1].grid(True, alpha=0.28)

    axes[1, 0].plot(radii, relative_residuals, marker="o", linewidth=1.4, color="#0f766e")
    axes[1, 0].set_title("Relative residual")
    axes[1, 0].set_xlabel("radius")
    axes[1, 0].set_ylabel("relative residual")
    axes[1, 0].grid(True, alpha=0.28)

    image = axes[1, 1].imshow(basis, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 1].set_title("Basis/components")
    axes[1, 1].set_xlabel("stored lambda/contact index")
    axes[1, 1].set_ylabel("component")
    fig.colorbar(image, ax=axes[1, 1], pad=0.012, label="basis value")

    fig.suptitle(f"{method_name} on {dataset_label}: R={basis.shape[0]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
