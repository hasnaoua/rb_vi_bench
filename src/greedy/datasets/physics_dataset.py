from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("data/physics_data.txt")
DEFAULT_OUTPUT_DIR = Path("results/physics/dataset")
DEFAULT_DATASET = DEFAULT_OUTPUT_DIR / "physics_dataset.npz"
DEFAULT_GRID_SHAPE = (76, 101)
CONTACT_NODE_COUNT = 7676

HEIGHT_MM = 5.0
R_IN_MM = 4.1
R_OUT_MM = 4.7
INITIAL_GAP_MM = 0.05
PELLET_RADIUS_MM = R_IN_MM - INITIAL_GAP_MM
SECTOR_ANGLE_RAD = 0.5 * np.pi


def displacement_grid() -> np.ndarray:
    """Return the imposed Z-displacement grid from the problem statement."""
    first_block = 0.18 + 0.005 * np.arange(24, dtype=float)
    second_block = 0.3 + 0.01 * np.arange(72, dtype=float)
    return np.concatenate([first_block, second_block])


def load_raw_physics_matrix(path: Path = DEFAULT_INPUT) -> np.ndarray:
    raw = np.loadtxt(path)
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D numeric table, got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{path} contains non-finite values")
    return raw


def orient_as_snapshots(
    raw: np.ndarray,
    *,
    node_count: int = CONTACT_NODE_COUNT,
) -> tuple[np.ndarray, str]:
    """
    Return snapshots with shape (N_snapshots, N_contact_nodes).

    The supplied physics file is expected as nodes x snapshots, but this helper
    also accepts an already-oriented snapshots x nodes matrix.
    """
    if raw.shape[0] == node_count:
        return raw.T.copy(), "nodes_by_snapshots_transposed"
    if raw.shape[1] == node_count:
        return raw.copy(), "snapshots_by_nodes"

    raise ValueError(
        f"Could not identify the contact-node dimension {node_count} in raw shape "
        f"{raw.shape}"
    )


def align_to_displacement_grid(
    snapshots: np.ndarray,
    *,
    extra_columns: str = "drop-leading",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Attach the 96 displacement values from the text to the snapshot matrix.

    The local file currently has 99 columns. With the published 96-point
    displacement grid, the most consistent alignment is to drop the three
    leading extra columns:
    this leaves two nearly-zero no-contact states at displacements 0.18 and
    0.185 mm, followed by active contact snapshots.
    """
    displacements = displacement_grid()
    n_snapshots = snapshots.shape[0]
    original_indices = np.arange(n_snapshots, dtype=int)

    if n_snapshots == displacements.size:
        return snapshots, displacements, original_indices, "exact_displacement_grid_size"

    if n_snapshots < displacements.size:
        used = displacements[:n_snapshots]
        note = (
            f"data_has_fewer_snapshots_than_displacement_grid:"
            f"{n_snapshots}_of_{displacements.size}"
        )
        return snapshots, used, original_indices, note

    extra = n_snapshots - displacements.size
    if extra_columns == "drop-leading":
        return (
            snapshots[extra:].copy(),
            displacements,
            original_indices[extra:].copy(),
            f"dropped_{extra}_leading_extra_columns",
        )
    if extra_columns == "drop-trailing":
        return (
            snapshots[:-extra].copy(),
            displacements,
            original_indices[:-extra].copy(),
            f"dropped_{extra}_trailing_extra_columns",
        )
    if extra_columns == "keep-index":
        parameters = np.arange(n_snapshots, dtype=float)
        return snapshots, parameters, original_indices, "kept_all_using_snapshot_index"

    raise ValueError(
        "--extra-columns must be one of: drop-leading, drop-trailing, keep-index"
    )


def apply_active_filter(
    snapshots: np.ndarray,
    parameters: np.ndarray,
    original_indices: np.ndarray,
    *,
    active_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if active_threshold is None:
        return snapshots, parameters, original_indices, "inactive_snapshots_kept"

    norms = np.linalg.norm(snapshots, axis=1)
    active = norms > active_threshold
    return (
        snapshots[active].copy(),
        parameters[active].copy(),
        original_indices[active].copy(),
        f"kept_{int(np.count_nonzero(active))}_with_norm_gt_{active_threshold:g}",
    )


def build_physics_dataset(
    input_path: Path = DEFAULT_INPUT,
    *,
    extra_columns: str = "drop-leading",
    active_threshold: float | None = None,
) -> dict[str, np.ndarray | str]:
    raw = load_raw_physics_matrix(input_path)
    snapshots, orientation = orient_as_snapshots(raw)
    aligned, parameters, original_indices, alignment_note = align_to_displacement_grid(
        snapshots,
        extra_columns=extra_columns,
    )
    filtered, filtered_parameters, filtered_indices, active_note = apply_active_filter(
        aligned,
        parameters,
        original_indices,
        active_threshold=active_threshold,
    )

    if filtered.shape[0] != filtered_parameters.size:
        raise ValueError("snapshot rows and parameter values are inconsistent")
    if filtered.shape[1] != CONTACT_NODE_COUNT:
        raise ValueError(
            f"expected {CONTACT_NODE_COUNT} contact nodes, got {filtered.shape[1]}"
        )

    return {
        "snapshots": filtered,
        "radii": filtered_parameters,
        "displacements": filtered_parameters,
        "original_indices": filtered_indices,
        "raw_shape": np.array(raw.shape, dtype=int),
        "grid_shape": np.array(DEFAULT_GRID_SHAPE, dtype=int),
        "input_path": str(input_path),
        "orientation": orientation,
        "alignment_note": alignment_note,
        "active_filter_note": active_note,
        "parameter_name": "imposed_displacement_mm"
        if extra_columns != "keep-index"
        else "snapshot_index",
    }


def _heatmap_values(matrix: np.ndarray) -> np.ndarray:
    tiny = np.finfo(float).tiny
    return np.log10(np.abs(matrix) + tiny)


def contact_surface_axes(
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return angular and axial grid axes for the quarter contact surface."""
    theta_count, z_count = grid_shape
    theta = np.linspace(0.0, SECTOR_ANGLE_RAD, theta_count)
    z = np.linspace(0.0, HEIGHT_MM, z_count)
    return theta, z


def reshape_contact_surface(
    values: np.ndarray,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> np.ndarray:
    """
    Reshape one flattened contact snapshot as theta x z values.

    The data file does not include mesh connectivity or node coordinates. The
    7,676 contact values are therefore displayed on the structured 76 x 101
    quarter-cylinder grid implied by the local dataset metadata.
    """
    values = np.asarray(values, dtype=float)
    expected = int(np.prod(grid_shape))
    if values.size != expected:
        raise ValueError(
            f"grid_shape={grid_shape} expects {expected} values, got {values.size}"
        )
    return values.reshape(grid_shape)


def contact_surface_coordinates(
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    *,
    radius_mm: float = R_IN_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, Z coordinates for the quarter-cylinder contact surface."""
    theta, z = contact_surface_axes(grid_shape)
    theta_grid, z_grid = np.meshgrid(theta, z, indexing="ij")
    x_grid = radius_mm * np.cos(theta_grid)
    y_grid = radius_mm * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def _surface_log_values(values: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.abs(values))


def _robust_limits(
    values: np.ndarray,
    *,
    percentile_clip: tuple[float, float] = (1.0, 99.5),
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, percentile_clip)
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def plot_snapshot_norms(
    snapshots: np.ndarray,
    displacements: np.ndarray,
    original_indices: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(snapshots, axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(displacements, norms, marker="o", linewidth=1.45, markersize=3.5)
    for x, idx, norm in zip(displacements, original_indices, norms):
        if norm <= 0.1:
            ax.annotate(
                str(int(idx)),
                (x, max(norm, np.finfo(float).tiny)),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )
    ax.set_title("Physics contact-force snapshot norms")
    ax.set_xlabel("imposed displacement [mm]")
    ax.set_ylabel("L2 norm of contact-force vector")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_snapshot_heatmap(
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    image = ax.imshow(
        _heatmap_values(snapshots),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_title("Physics contact-force snapshots")
    ax.set_xlabel("contact node index")
    ax.set_ylabel("imposed displacement [mm]")

    tick_count = min(12, snapshots.shape[0])
    ticks = np.unique(np.linspace(0, snapshots.shape[0] - 1, tick_count, dtype=int))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{displacements[i]:.3g}" for i in ticks])
    fig.colorbar(image, ax=ax, pad=0.012, label=r"$\log_{10}(|value|)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_surface_snapshot_grid(
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    max_snapshots: int = 8,
) -> None:
    """Plot representative snapshots as unwrapped theta-z contact maps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.unique(
        np.linspace(0, snapshots.shape[0] - 1, min(max_snapshots, snapshots.shape[0]), dtype=int)
    )
    if rows.size == 0:
        return

    transformed = np.array(
        [_surface_log_values(reshape_contact_surface(snapshots[row], grid_shape)) for row in rows]
    )
    vmin, vmax = _robust_limits(transformed)

    ncols = min(4, rows.size)
    nrows = int(np.ceil(rows.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.0 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for axis_index, ax in enumerate(axes.ravel()):
        if axis_index >= rows.size:
            ax.axis("off")
            continue
        row = int(rows[axis_index])
        image_values = transformed[axis_index]
        last_image = ax.imshow(
            image_values,
            origin="lower",
            aspect="auto",
            extent=[0.0, HEIGHT_MM, 0.0, np.degrees(SECTOR_ANGLE_RAD)],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"u_z={displacements[row]:.3g} mm")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("theta [deg]")

    fig.suptitle("Physics contact force on unwrapped quarter-cylinder interface")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label(r"$\log_{10}(1+|contact force|)$")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_surface_envelope(
    snapshots: np.ndarray,
    output_path: Path,
    *,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> None:
    """Plot the maximum absolute contact force reached at every interface node."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = np.max(np.abs(snapshots), axis=0)
    surface = reshape_contact_surface(envelope, grid_shape)
    values = _surface_log_values(surface)
    vmin, vmax = _robust_limits(values)

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[0.0, HEIGHT_MM, 0.0, np.degrees(SECTOR_ANGLE_RAD)],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Maximum contact-force envelope on quarter-cylinder interface")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("theta [deg]")
    fig.colorbar(image, ax=ax, pad=0.012, label=r"$\log_{10}(1+|max force|)$")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quarter_cylinder_snapshots(
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    max_snapshots: int = 6,
) -> None:
    """Render representative contact snapshots on the 3D quarter cylinder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.unique(
        np.linspace(0, snapshots.shape[0] - 1, min(max_snapshots, snapshots.shape[0]), dtype=int)
    )
    if rows.size == 0:
        return

    x_grid, y_grid, z_grid = contact_surface_coordinates(grid_shape)
    transformed = np.array(
        [_surface_log_values(reshape_contact_surface(snapshots[row], grid_shape)) for row in rows]
    )
    vmin, vmax = _robust_limits(transformed)
    normaliser = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    ncols = min(3, rows.size)
    nrows = int(np.ceil(rows.size / ncols))
    fig = plt.figure(figsize=(4.6 * ncols, 4.0 * nrows), constrained_layout=True)

    for plot_index, row in enumerate(rows, start=1):
        ax = fig.add_subplot(nrows, ncols, plot_index, projection="3d")
        surface_values = _surface_log_values(reshape_contact_surface(snapshots[int(row)], grid_shape))
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            facecolors=cmap(normaliser(surface_values)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        ax.set_title(f"u_z={displacements[int(row)]:.3g} mm", pad=8)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_xlim(0.0, R_OUT_MM)
        ax.set_ylim(0.0, R_OUT_MM)
        ax.set_zlim(0.0, HEIGHT_MM)
        ax.view_init(elev=22.0, azim=-55.0)

    scalar_mappable = plt.cm.ScalarMappable(norm=normaliser, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=fig.axes, shrink=0.72)
    cbar.set_label(r"$\log_{10}(1+|contact force|)$")
    fig.suptitle("Contact-force snapshots on the 3D quarter-cylinder interface")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(dataset: dict[str, np.ndarray | str], output_path: Path) -> None:
    snapshots = np.asarray(dataset["snapshots"], dtype=float)
    displacements = np.asarray(dataset["displacements"], dtype=float)
    original_indices = np.asarray(dataset["original_indices"], dtype=int)
    norms = np.linalg.norm(snapshots, axis=1)

    lines = [
        "Physics contact-force dataset report",
        "",
        f"input_path: {dataset['input_path']}",
        f"raw_shape: {tuple(np.asarray(dataset['raw_shape'], dtype=int))}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"grid_shape: {tuple(np.asarray(dataset['grid_shape'], dtype=int))}",
        "surface_grid_interpretation: theta_count x z_count",
        f"height_mm: {HEIGHT_MM:.12g}",
        f"sector_angle_rad: {SECTOR_ANGLE_RAD:.12g}",
        f"inner_cladding_radius_mm: {R_IN_MM:.12g}",
        f"outer_cladding_radius_mm: {R_OUT_MM:.12g}",
        f"pellet_radius_mm: {PELLET_RADIUS_MM:.12g}",
        f"parameter_name: {dataset['parameter_name']}",
        f"orientation: {dataset['orientation']}",
        f"alignment_note: {dataset['alignment_note']}",
        f"active_filter_note: {dataset['active_filter_note']}",
        f"original_column_indices: {[int(i) for i in original_indices]}",
        f"displacement_min_mm: {float(np.min(displacements)):.12g}",
        f"displacement_max_mm: {float(np.max(displacements)):.12g}",
        f"norm_min: {float(np.min(norms)):.18e}",
        f"norm_max: {float(np.max(norms)):.18e}",
        f"norm_gt_0.1_count: {int(np.count_nonzero(norms > 0.1))}",
        "",
        "Problem description encoded in this dataset:",
        "- quarter 3D pellet-cladding contact interface",
        "- 7,676 potential contact nodes",
        "- imposed top-surface axial displacement used as parameter",
        "- displacement values follow the two-block grid in the prompt",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_physics_dataset(
    dataset: dict[str, np.ndarray | str],
    dataset_path: Path = DEFAULT_DATASET,
) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare physics_data.txt as snapshots x contact-nodes data compatible "
            "with the existing CPG/ADG comparison scripts."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-name",
        default="physics_dataset.npz",
        help="Name of the compressed .npz dataset written inside --output-dir.",
    )
    parser.add_argument(
        "--extra-columns",
        choices=["drop-leading", "drop-trailing", "keep-index"],
        default="drop-leading",
        help=(
            "How to handle a file with more columns than the 96 displacement "
            "values from the prompt grid."
        ),
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=None,
        help="Optional norm threshold for dropping inactive/no-contact snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_physics_dataset(
        args.input,
        extra_columns=args.extra_columns,
        active_threshold=args.active_threshold,
    )
    dataset_path = args.output_dir / args.dataset_name
    report_path = args.output_dir / "physics_dataset_report.txt"
    norms_plot = args.output_dir / "physics_snapshot_norms.png"
    heatmap_plot = args.output_dir / "physics_snapshot_heatmap.png"
    surface_plot = args.output_dir / "physics_surface_snapshots.png"
    envelope_plot = args.output_dir / "physics_surface_envelope.png"
    cylinder_plot = args.output_dir / "physics_quarter_cylinder_snapshots.png"

    save_physics_dataset(dataset, dataset_path)
    write_report(dataset, report_path)
    plot_snapshot_norms(
        np.asarray(dataset["snapshots"], dtype=float),
        np.asarray(dataset["displacements"], dtype=float),
        np.asarray(dataset["original_indices"], dtype=int),
        norms_plot,
    )
    plot_snapshot_heatmap(
        np.asarray(dataset["snapshots"], dtype=float),
        np.asarray(dataset["displacements"], dtype=float),
        heatmap_plot,
    )
    plot_surface_snapshot_grid(
        np.asarray(dataset["snapshots"], dtype=float),
        np.asarray(dataset["displacements"], dtype=float),
        surface_plot,
    )
    plot_surface_envelope(
        np.asarray(dataset["snapshots"], dtype=float),
        envelope_plot,
    )
    plot_quarter_cylinder_snapshots(
        np.asarray(dataset["snapshots"], dtype=float),
        np.asarray(dataset["displacements"], dtype=float),
        cylinder_plot,
    )

    snapshots = np.asarray(dataset["snapshots"], dtype=float)
    print(f"Wrote dataset: {dataset_path}")
    print(f"Snapshot matrix: {snapshots.shape}")
    print(f"Alignment: {dataset['alignment_note']}")
    print(f"Report: {report_path}")
    print(f"Plots: {norms_plot}, {heatmap_plot}")
    print(f"Surface plots: {surface_plot}, {envelope_plot}, {cylinder_plot}")


if __name__ == "__main__":
    main()
