from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from greedy.core.reduction_common import (
    fem_sols_train_test_split,
    max_projection_error_history,
    parameter_train_test_split,
    plot_convergence_with_angle,
    plot_error_gain,
    solve_nonnegative_least_squares,
    write_error_gain_csv,
)

from greedy.core.cpg import CPG
from greedy.datasets.lambda_snapshots import (
    default_input_path,
    load_records,
    make_visualizations,
    records_to_arrays,
    save_merged_outputs,
)
from greedy.viz.contact_force_profiles import (
    centered_contact_view,
    load_x_coordinates,
    proxy_abscissa,
)


DEFAULT_DATASET = Path("results/lambda/dataset/lambda_dataset.npz")
DEFAULT_OUTPUT_DIR = Path("results/lambda/cpg")


def load_or_build_dataset(
    dataset_path: Path,
    input_path: Path,
    merge_output_dir: Path,
    merge_prefix: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if dataset_path.exists():
        data = np.load(dataset_path, allow_pickle=False)
        snapshots = np.asarray(data["snapshots"], dtype=float)
        radii = np.asarray(data["radii"], dtype=float)
        source = str(dataset_path)
    else:
        records = load_records(input_path)
        save_merged_outputs(records, merge_output_dir, merge_prefix)
        snapshots, radii = records_to_arrays(records)
        source = str(input_path)

    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got shape {snapshots.shape}")
    if radii.ndim != 1:
        raise ValueError(f"radii must be 1-D, got shape {radii.shape}")
    if snapshots.shape[0] != radii.size:
        raise ValueError(
            f"snapshot rows ({snapshots.shape[0]}) must match radii ({radii.size})"
        )
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshots contain non-finite values")
    if not np.all(np.isfinite(radii)):
        raise ValueError("radii contain non-finite values")

    return snapshots, radii, source


def project_snapshots_onto_basis(
    snapshots: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if basis.size == 0:
        approximations = np.zeros_like(snapshots)
        coefficients = np.zeros((snapshots.shape[0], 0))
        residuals = np.linalg.norm(snapshots, axis=1)
        return approximations, coefficients, residuals

    system = basis.T
    approximations = np.empty_like(snapshots)
    coefficients = np.empty((snapshots.shape[0], basis.shape[0]), dtype=float)
    residuals = np.empty(snapshots.shape[0], dtype=float)

    for row, snapshot in enumerate(snapshots):
        coeff = solve_nonnegative_least_squares(system, snapshot)
        approximation = system @ coeff
        coefficients[row] = coeff
        approximations[row] = approximation
        residuals[row] = np.linalg.norm(snapshot - approximation)

    return approximations, coefficients, residuals


def nice_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_count, dtype=int))


def safe_relative_residuals(snapshots: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(snapshots, axis=1)
    return np.divide(
        residuals,
        norms,
        out=np.zeros_like(residuals),
        where=norms > 0.0,
    )


def contact_axis_label(
    x_coordinates: np.ndarray | None,
    x_mode: str,
    fixed_extent: float,
) -> str:
    if x_coordinates is not None:
        return "contact coordinate from supplied file"
    if x_mode == "index":
        return "stored lambda/contact index"
    if x_mode == "radius":
        return "proxy coordinate, linearly spaced in [-parameter, parameter]"
    if x_mode == "unit":
        return "proxy coordinate, linearly spaced in [-1, 1]"
    return (
        "proxy coordinate, "
        f"linearly spaced in [-{fixed_extent:g}, {fixed_extent:g}]"
    )


def contact_curve(
    values: np.ndarray,
    radius: float,
    x_coordinates: np.ndarray | None,
    x_mode: str,
    fixed_extent: float,
    center_contact: bool,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if x_coordinates is not None:
        return x_coordinates, values

    x = proxy_abscissa(values.size, radius, x_mode, fixed_extent)
    y = centered_contact_view(values, threshold) if center_contact else values
    return x, y


def centered_pair(
    reference: np.ndarray,
    other: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(np.abs(reference) > threshold)
    if active.size == 0:
        return reference.copy(), other.copy()

    active_center = int(round(0.5 * (int(active[0]) + int(active[-1]))))
    target_center = (reference.size - 1) // 2
    shift = target_center - active_center
    return np.roll(reference, shift), np.roll(other, shift)


def plot_contact_basis_profiles(
    basis: np.ndarray,
    selected_radii: np.ndarray,
    selected_indices: list[int],
    output_path: Path,
    x_coordinates: np.ndarray | None,
    x_mode: str,
    fixed_extent: float,
    center_contact: bool,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    if basis.shape[0] == 0:
        ax.text(0.5, 0.5, "No CPG basis vectors selected", ha="center", va="center")
    else:
        for basis_row, values in enumerate(basis):
            radius = float(selected_radii[basis_row])
            x, y = contact_curve(
                values,
                radius,
                x_coordinates,
                x_mode,
                fixed_extent,
                center_contact,
                threshold,
            )
            ax.plot(
                x,
                y,
                marker=markers[basis_row % len(markers)],
                markersize=3.3,
                linewidth=1.35,
                label=(
                    f"basis {basis_row + 1}: "
                    f"row {selected_indices[basis_row]}, r={radius:.5g}"
                ),
            )

    title = "CPG selected basis contact-force profiles"
    if center_contact and x_coordinates is None and x_mode != "index":
        title += "  (contact patch centered for display)"
    ax.axhline(0.0, color="#8a95a1", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(contact_axis_label(x_coordinates, x_mode, fixed_extent))
    ax.set_ylabel("lambda value")
    ax.grid(True, alpha=0.28)
    if basis.shape[0] > 0:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_contact_projection_profiles(
    approximations: np.ndarray,
    radii: np.ndarray,
    rows: np.ndarray,
    output_path: Path,
    x_coordinates: np.ndarray | None,
    x_mode: str,
    fixed_extent: float,
    center_contact: bool,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for curve_index, row in enumerate(rows):
        x, y = contact_curve(
            approximations[row],
            float(radii[row]),
            x_coordinates,
            x_mode,
            fixed_extent,
            center_contact,
            threshold,
        )
        ax.plot(
            x,
            y,
            marker=markers[curve_index % len(markers)],
            markersize=3.3,
            linewidth=1.35,
            label=f"CPG projection r={radii[row]:.5g}  (row {row})",
        )

    title = "CPG-projected contact-force profiles"
    if center_contact and x_coordinates is None and x_mode != "index":
        title += "  (contact patch centered for display)"
    ax.axhline(0.0, color="#8a95a1", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(contact_axis_label(x_coordinates, x_mode, fixed_extent))
    ax.set_ylabel("lambda value")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_contact_projection_comparison(
    snapshots: np.ndarray,
    approximations: np.ndarray,
    radii: np.ndarray,
    residuals: np.ndarray,
    rows: np.ndarray,
    output_path: Path,
    x_coordinates: np.ndarray | None,
    x_mode: str,
    fixed_extent: float,
    center_contact: bool,
    threshold: float,
) -> None:
    if rows.size == 0:
        return

    ncols = min(3, rows.size)
    nrows = int(np.ceil(rows.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 3.4 * nrows),
        squeeze=False,
    )
    short_xlabel = (
        "contact coordinate"
        if x_coordinates is not None
        else "stored lambda/contact index"
    )

    for axis_index, ax in enumerate(axes.ravel()):
        if axis_index >= rows.size:
            ax.axis("off")
            continue

        row = int(rows[axis_index])
        if x_coordinates is not None:
            x = x_coordinates
            input_values = snapshots[row]
            projected_values = approximations[row]
        else:
            x = proxy_abscissa(snapshots.shape[1], float(radii[row]), x_mode, fixed_extent)
            if center_contact:
                input_values, projected_values = centered_pair(
                    snapshots[row],
                    approximations[row],
                    threshold,
                )
            else:
                input_values = snapshots[row]
                projected_values = approximations[row]

        ax.plot(x, input_values, color="#17202a", linewidth=1.55, label="input")
        ax.plot(
            x,
            projected_values,
            color="#0f766e",
            linewidth=1.5,
            linestyle="--",
            label="CPG projection",
        )
        ax.fill_between(
            x,
            input_values,
            projected_values,
            color="#c2410c",
            alpha=0.16,
            linewidth=0,
        )
        ax.set_title(f"row {row}, r={radii[row]:.5g}, residual={residuals[row]:.2e}")
        if axis_index // ncols == nrows - 1:
            ax.set_xlabel(short_xlabel)
        ax.set_ylabel("lambda")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    title = "Input vs CPG projection on the contact boundary"
    if center_contact and x_coordinates is None and x_mode != "index":
        title += "  (contact patch centered for display)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_profiles(
    snapshots: np.ndarray,
    radii: np.ndarray,
    selected_indices: list[int],
    output_path: Path,
) -> None:
    x = np.arange(snapshots.shape[1])
    fig, ax = plt.subplots(figsize=(12, 6))

    for row in range(snapshots.shape[0]):
        ax.plot(x, snapshots[row], color="#c9d2dc", linewidth=0.8, alpha=0.65)

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(selected_indices))))
    for color, row in zip(colors, selected_indices):
        ax.plot(
            x,
            snapshots[row],
            linewidth=2.2,
            color=color,
            label=f"row {row}, r={radii[row]:.3g}",
        )

    ax.set_title("Input snapshots with CPG-selected basis profiles")
    ax.set_xlabel("lambda component index")
    ax.set_ylabel("lambda value")
    ax.grid(True, alpha=0.28)
    if selected_indices:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basis_heatmap(
    basis: np.ndarray,
    selected_radii: np.ndarray,
    output_path: Path,
) -> None:
    height = max(4.0, min(9.0, 2.4 + 0.32 * max(1, basis.shape[0])))
    fig, ax = plt.subplots(figsize=(12, height))

    if basis.shape[0] == 0:
        ax.text(0.5, 0.5, "No basis vectors selected", ha="center", va="center")
        ax.set_axis_off()
    else:
        image = ax.imshow(basis, aspect="auto", origin="lower", cmap="viridis")
        ax.set_yticks(np.arange(basis.shape[0]))
        ax.set_yticklabels([f"{radius:.3g}" for radius in selected_radii])
        cbar = fig.colorbar(image, ax=ax, pad=0.012)
        cbar.set_label("lambda value")

    ax.set_title(f"CPG basis heatmap ({basis.shape[0]} selected snapshots)")
    ax.set_xlabel("lambda component index")
    ax.set_ylabel("selected radius")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_convergence(
    residual_history: list[float],
    residual_threshold: float,
    output_path: Path,
    basis_sizes: list[int] | None = None,
    candidate_angle_history: list[float] | None = None,
    threshold_label: str | None = None,
) -> None:
    """
    Residual convergence with an optional twin axis showing the angle (deg)
    between each accepted candidate and the cone it joined — 0 deg means it
    added no new information, 90 deg means it was fully novel/orthogonal.
    """
    angle_deg = (
        np.degrees(np.asarray(candidate_angle_history, dtype=float))
        if candidate_angle_history
        else None
    )
    plot_convergence_with_angle(
        residual_history,
        residual_threshold,
        output_path,
        basis_sizes=basis_sizes,
        angle_history_deg=angle_deg,
        angle_basis_sizes=basis_sizes,
        title="CPG convergence",
        method_name="CPG",
        residual_label="max residual after basis update",
        threshold_label=threshold_label,
    )


def plot_residuals_by_radius(
    radii: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    selected_indices: list[int],
    epsilon: float,
    output_path: Path,
) -> None:
    selected = np.array(selected_indices, dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(radii, residuals, marker="o", linewidth=1.5)
    axes[0].axhline(
        epsilon,
        color="#c2410c",
        linestyle="--",
        label="relative epsilon threshold",
    )

    if selected.size:
        axes[0].scatter(
            radii[selected],
            residuals[selected],
            color="#c2410c",
            zorder=3,
            label="selected",
        )
        axes[1].scatter(
            radii[selected],
            relative_residuals[selected],
            color="#c2410c",
            zorder=3,
        )

    axes[0].set_ylabel("absolute residual")
    axes[0].set_title("CPG projection residuals by radius")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    axes[1].plot(radii, relative_residuals, marker="o", linewidth=1.5, color="#0f766e")
    axes[1].set_ylabel("relative residual")
    axes[1].set_xlabel("radius")
    axes[1].grid(True, alpha=0.28)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_examples(
    snapshots: np.ndarray,
    approximations: np.ndarray,
    radii: np.ndarray,
    residuals: np.ndarray,
    output_path: Path,
    max_examples: int,
) -> None:
    if snapshots.shape[0] == 0:
        return

    selected = set(nice_indices(snapshots.shape[0], max_examples).tolist())
    selected.add(int(np.argmax(residuals)))
    example_rows = np.array(sorted(selected), dtype=int)
    if example_rows.size > max_examples:
        ranked = example_rows[np.argsort(residuals[example_rows])[::-1]]
        keep = set(ranked[:max_examples].tolist())
        example_rows = np.array([row for row in example_rows if row in keep], dtype=int)

    ncols = min(3, example_rows.size)
    nrows = int(np.ceil(example_rows.size / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.4 * nrows),
        squeeze=False,
    )
    x = np.arange(snapshots.shape[1])

    for axis_index, ax in enumerate(axes.ravel()):
        if axis_index >= example_rows.size:
            ax.axis("off")
            continue

        row = int(example_rows[axis_index])
        ax.plot(x, snapshots[row], color="#17202a", linewidth=1.7, label="input")
        ax.plot(x, approximations[row], color="#0f766e", linewidth=1.5, linestyle="--", label="CPG projection")
        ax.fill_between(
            x,
            snapshots[row],
            approximations[row],
            color="#c2410c",
            alpha=0.16,
            linewidth=0,
            label="error" if axis_index == 0 else None,
        )
        ax.set_title(f"row {row}, r={radii[row]:.3g}, residual={residuals[row]:.2e}")
        ax.set_xlabel("lambda component")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("CPG projection examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cpg_summary(
    snapshots: np.ndarray,
    radii: np.ndarray,
    basis: np.ndarray,
    selected_indices: list[int],
    residual_history: list[float],
    residuals: np.ndarray,
    residual_threshold: float,
    relative_epsilon: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    selected = np.array(selected_indices, dtype=int)

    norms = np.linalg.norm(snapshots, axis=1)
    axes[0, 0].plot(radii, norms, marker="o", linewidth=1.5)
    if selected.size:
        axes[0, 0].scatter(radii[selected], norms[selected], color="#c2410c", zorder=3)
    axes[0, 0].set_title("Input snapshot norms")
    axes[0, 0].set_xlabel("radius")
    axes[0, 0].set_ylabel("L2 norm")
    axes[0, 0].grid(True, alpha=0.28)

    if residual_history:
        axes[0, 1].semilogy(
            np.arange(1, len(residual_history) + 1),
            residual_history,
            marker="o",
            linewidth=1.5,
        )
    axes[0, 1].axhline(residual_threshold, color="#c2410c", linestyle="--")
    axes[0, 1].set_title("Convergence")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("max residual")
    axes[0, 1].grid(True, which="both", alpha=0.28)

    axes[1, 0].plot(radii, residuals, marker="o", linewidth=1.5, color="#0f766e")
    axes[1, 0].axhline(residual_threshold, color="#c2410c", linestyle="--")
    if selected.size:
        axes[1, 0].scatter(radii[selected], residuals[selected], color="#c2410c", zorder=3)
    axes[1, 0].set_title("Final projection residuals")
    axes[1, 0].set_xlabel("radius")
    axes[1, 0].set_ylabel("absolute residual")
    axes[1, 0].grid(True, alpha=0.28)

    if basis.shape[0] == 0:
        axes[1, 1].text(
            0.5,
            0.5,
            "No basis vectors selected",
            ha="center",
            va="center",
        )
        axes[1, 1].set_axis_off()
    else:
        image = axes[1, 1].imshow(basis, aspect="auto", origin="lower", cmap="viridis")
        fig.colorbar(image, ax=axes[1, 1], pad=0.012, label="lambda value")
    axes[1, 1].set_title("CPG basis snapshots")
    axes[1, 1].set_xlabel("lambda component")
    axes[1, 1].set_ylabel("basis row")
    if basis.shape[0] > 0:
        axes[1, 1].set_yticks(np.arange(basis.shape[0]))
        axes[1, 1].set_yticklabels([str(row) for row in selected_indices])

    fig.suptitle(
        f"CPG on lambda snapshots: relative epsilon={relative_epsilon:g}, R={basis.shape[0]}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_cpg_outputs(
    output_dir: Path,
    snapshots: np.ndarray,
    radii: np.ndarray,
    basis: np.ndarray,
    selected_indices: list[int],
    coefficients: np.ndarray,
    approximations: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    residual_history: list[float],
    epsilon: float,
    elapsed: float,
    dataset_source: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split_strategy: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = np.array(selected_indices, dtype=int)
    selected_radii = radii[selected] if selected.size else np.array([], dtype=float)

    paths = {
        "basis": output_dir / "cpg_basis.npy",
        "selected_indices": output_dir / "cpg_selected_indices.txt",
        "selected_radii": output_dir / "cpg_selected_radii.txt",
        "coefficients": output_dir / "cpg_coefficients.npy",
        "approximations": output_dir / "cpg_projections.npy",
        "residuals_csv": output_dir / "cpg_residuals.csv",
        "convergence": output_dir / "cpg_convergence.txt",
        "report": output_dir / "cpg_report.txt",
    }

    np.save(paths["basis"], basis)
    np.save(paths["coefficients"], coefficients)
    np.save(paths["approximations"], approximations)
    np.savetxt(paths["selected_indices"], selected, fmt="%d")
    np.savetxt(paths["selected_radii"], selected_radii, fmt="%.12g")
    np.savetxt(paths["convergence"], np.asarray(residual_history), fmt="%.18e")

    selected_set = set(selected_indices)
    split_labels = np.full(snapshots.shape[0], "train", dtype=object)
    split_labels[np.asarray(test_indices, dtype=int)] = "test"
    with paths["residuals_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "split", "radius", "residual", "relative_residual", "selected"])
        for row, (radius, residual, relative) in enumerate(
            zip(radii, residuals, relative_residuals)
        ):
            writer.writerow(
                [
                    row,
                    split_labels[row],
                    f"{radius:.12g}",
                    f"{residual:.18e}",
                    f"{relative:.18e}",
                    int(row in selected_set),
                ]
            )

    final_max = float(np.max(residuals)) if residuals.size else 0.0
    final_relative_max = float(np.max(relative_residuals)) if relative_residuals.size else 0.0
    train_relative = relative_residuals[train_indices] if train_indices.size else np.array([])
    test_relative = relative_residuals[test_indices] if test_indices.size else np.array([])
    train_norms = np.linalg.norm(snapshots[train_indices], axis=1) if train_indices.size else np.array([])
    stopping_scale = float(np.max(train_norms)) if train_norms.size else 1.0
    if stopping_scale <= 0.0:
        stopping_scale = 1.0
    stopping_tolerance = epsilon * stopping_scale
    report = [
        "CPG lambda snapshot report",
        "",
        f"dataset_source: {dataset_source}",
        f"relative_epsilon: {epsilon:.12g}",
        f"stopping_scale_max_train_norm: {stopping_scale:.18e}",
        f"absolute_stopping_tolerance: {stopping_tolerance:.18e}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"split_strategy: {split_strategy}",
        "validation: offline basis built on training radii only; "
        "test residuals use unseen held-out radii.",
        f"train_count: {train_indices.size}",
        f"test_count: {test_indices.size}",
        f"train_radii: {[float(radii[i]) for i in train_indices]}",
        f"test_radii: {[float(radii[i]) for i in test_indices]}",
        f"basis_shape: {basis.shape}",
        f"selected_count_R: {basis.shape[0]}",
        f"selected_indices: {selected_indices}",
        f"selected_radii: {[float(v) for v in selected_radii]}",
        f"final_max_residual: {final_max:.18e}",
        f"final_max_relative_residual: {final_relative_max:.18e}",
        f"train_max_relative_residual: {float(np.max(train_relative)) if train_relative.size else float('nan'):.18e}",
        f"train_mean_relative_residual: {float(np.mean(train_relative)) if train_relative.size else float('nan'):.18e}",
        f"test_max_relative_residual: {float(np.max(test_relative)) if test_relative.size else float('nan'):.18e}",
        f"test_mean_relative_residual: {float(np.mean(test_relative)) if test_relative.size else float('nan'):.18e}",
        f"elapsed_seconds: {elapsed:.6f}",
        "",
        "Files:",
        "- cpg_basis.npy: selected CPG basis snapshots",
        "- cpg_selected_indices.txt: selected row numbers in the input snapshot matrix",
        "- cpg_selected_radii.txt: selected radius values",
        "- cpg_coefficients.npy: non-negative coefficients for projecting each input snapshot",
        "- cpg_projections.npy: CPG projection/reconstruction of every input snapshot",
        "- cpg_residuals.csv: residual per radius after the final CPG basis",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize lambda input snapshots, run CPG on them, and visualize "
            "the selected basis plus projection residuals."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Merged dataset .npz with snapshots and radii.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Fallback FEM_SOLS zip/archive directory if --dataset is missing.",
    )
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        default=Path("results/lambda/dataset"),
        help="Where to write merged files if --dataset must be rebuilt.",
    )
    parser.add_argument(
        "--merge-prefix",
        default="lambda",
        help="Filename prefix if merged files are rebuilt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CPG outputs and plots.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        help="CPG stopping tolerance in Euclidean norm.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["fem-sols-paper", "fraction"],
        default="fem-sols-paper",
        help=(
            "Training/test split. fem-sols-paper uses rad - 0.65 in "
            "{0.15 + 0.01 i | 0 <= i <= 30}; fraction uses --test-fraction."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of sorted interior radii held out when --split-strategy fraction."
        ),
    )
    parser.add_argument(
        "--max-input-curves",
        type=int,
        default=12,
        help="Maximum raw snapshots to overlay in the input profile plot.",
    )
    parser.add_argument(
        "--max-reconstruction-examples",
        type=int,
        default=6,
        help="Maximum original-vs-projection examples to plot.",
    )
    parser.add_argument(
        "--max-contact-curves",
        type=int,
        default=6,
        help="Maximum contact-force profiles to plot for projection examples.",
    )
    parser.add_argument(
        "--contact-x-coordinates",
        type=Path,
        default=None,
        help="Optional .npy/.txt array with the physical x-coordinate of each contact node.",
    )
    parser.add_argument(
        "--contact-x-mode",
        choices=["index", "unit", "radius", "fixed"],
        default="index",
        help="X-axis for contact plots when physical coordinates are not supplied.",
    )
    parser.add_argument(
        "--contact-fixed-extent",
        type=float,
        default=1.0,
        help="Half-width used by --contact-x-mode fixed.",
    )
    parser.add_argument(
        "--center-contact",
        action="store_true",
        help="Center the active contact patch in contact-force plots for display only.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=1e-10,
        help="Threshold used to identify active contact nodes for --center-contact.",
    )
    parser.add_argument(
        "--no-input-html",
        action="store_true",
        help="Skip the interactive input snapshot HTML viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots, radii, dataset_source = load_or_build_dataset(
        args.dataset,
        args.input,
        args.merge_output_dir,
        args.merge_prefix,
    )
    if args.split_strategy == "fem-sols-paper":
        train_indices, test_indices = fem_sols_train_test_split(radii)
        split_strategy = "fem-sols-paper: train where rad - 0.65 in {0.15 + 0.01 i | 0 <= i <= 30}"
    else:
        if args.test_fraction < 0.0 or args.test_fraction >= 1.0:
            raise ValueError("--test-fraction must be in [0, 1)")
        train_indices, test_indices = parameter_train_test_split(
            radii,
            test_fraction=args.test_fraction,
        )
        split_strategy = f"fraction: test_fraction={args.test_fraction:.12g}"
    train_snapshots = snapshots[train_indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_plot_paths = make_visualizations(
        snapshots,
        radii,
        args.output_dir,
        "input_snapshots",
        max_curves=args.max_input_curves,
        write_html=not args.no_input_html,
    )

    t0 = time.perf_counter()
    cpg = CPG(snapshots=train_snapshots, epsilon=args.epsilon)
    cpg.compute_phases()
    elapsed = time.perf_counter() - t0
    if cpg.basis_matrix is None:
        raise RuntimeError("CPG did not produce a basis matrix")

    basis = np.asarray(cpg.basis_matrix, dtype=float)
    if basis.size == 0:
        basis = np.empty((0, snapshots.shape[1]), dtype=float)
    selected_train_indices = [int(index) for index in cpg.selected_indices]
    selected_indices = [int(train_indices[index]) for index in selected_train_indices]
    approximations, coefficients, residuals = project_snapshots_onto_basis(snapshots, basis)
    relative_residuals = safe_relative_residuals(snapshots, residuals)

    data_paths = save_cpg_outputs(
        args.output_dir,
        snapshots,
        radii,
        basis,
        selected_indices,
        coefficients,
        approximations,
        residuals,
        relative_residuals,
        cpg.residual_history,
        args.epsilon,
        elapsed,
        dataset_source,
        train_indices,
        test_indices,
        split_strategy,
    )

    selected = np.array(selected_indices, dtype=int)
    selected_radii = radii[selected] if selected.size else np.array([], dtype=float)
    contact_x_coordinates = (
        load_x_coordinates(args.contact_x_coordinates, snapshots.shape[1])
        if args.contact_x_coordinates is not None
        else None
    )
    contact_rows = nice_indices(snapshots.shape[0], args.max_contact_curves)
    plot_paths = {
        "selected_profiles": args.output_dir / "cpg_selected_profiles.png",
        "basis_heatmap": args.output_dir / "cpg_basis_heatmap.png",
        "convergence_png": args.output_dir / "cpg_convergence.png",
        "error_gain_csv": args.output_dir / "cpg_error_gain.csv",
        "error_gain_png": args.output_dir / "cpg_error_gain.png",
        "absolute_error_gain_csv": args.output_dir / "cpg_absolute_error_gain.csv",
        "absolute_error_gain_png": args.output_dir / "cpg_absolute_error_gain.png",
        "residuals_png": args.output_dir / "cpg_residuals_by_radius.png",
        "projection_examples": args.output_dir / "cpg_projection_examples.png",
        "summary": args.output_dir / "cpg_summary.png",
        "basis_contact_profiles": args.output_dir / "cpg_basis_contact_force_profiles.png",
        "projection_contact_profiles": args.output_dir / "cpg_projection_contact_force_profiles.png",
        "projection_contact_comparison": args.output_dir / "cpg_projection_contact_force_comparison.png",
    }
    plot_selected_profiles(snapshots, radii, selected_indices, plot_paths["selected_profiles"])
    plot_basis_heatmap(basis, selected_radii, plot_paths["basis_heatmap"])
    plot_convergence(
        cpg.residual_history,
        cpg.stopping_tolerance,
        plot_paths["convergence_png"],
        cpg.residual_basis_sizes,
        cpg.candidate_angle_history,
        threshold_label=f"relative epsilon = {args.epsilon:g}",
    )
    error_iterations, error_values = max_projection_error_history(
        snapshots,
        basis,
        relative=True,
    )
    cpg_error_history = {"CPG": (error_iterations, error_values)}
    absolute_error_iterations, absolute_error_values = max_projection_error_history(
        snapshots,
        basis,
        relative=False,
    )
    cpg_absolute_error_history = {
        "CPG": (absolute_error_iterations, absolute_error_values)
    }
    write_error_gain_csv(cpg_error_history, plot_paths["error_gain_csv"])
    plot_error_gain(
        cpg_error_history,
        plot_paths["error_gain_png"],
        title="CPG relative error gain between greedy iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
    )
    write_error_gain_csv(
        cpg_absolute_error_history,
        plot_paths["absolute_error_gain_csv"],
    )
    plot_error_gain(
        cpg_absolute_error_history,
        plot_paths["absolute_error_gain_png"],
        title="CPG absolute error gain between greedy iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
    )
    plot_residuals_by_radius(
        radii,
        residuals,
        relative_residuals,
        selected_indices,
        cpg.stopping_tolerance,
        plot_paths["residuals_png"],
    )
    plot_reconstruction_examples(
        snapshots,
        approximations,
        radii,
        residuals,
        plot_paths["projection_examples"],
        max_examples=args.max_reconstruction_examples,
    )
    plot_cpg_summary(
        snapshots,
        radii,
        basis,
        selected_indices,
        cpg.residual_history,
        residuals,
        cpg.stopping_tolerance,
        args.epsilon,
        plot_paths["summary"],
    )
    plot_contact_basis_profiles(
        basis,
        selected_radii,
        selected_indices,
        plot_paths["basis_contact_profiles"],
        contact_x_coordinates,
        args.contact_x_mode,
        args.contact_fixed_extent,
        args.center_contact,
        args.contact_threshold,
    )
    plot_contact_projection_profiles(
        approximations,
        radii,
        contact_rows,
        plot_paths["projection_contact_profiles"],
        contact_x_coordinates,
        args.contact_x_mode,
        args.contact_fixed_extent,
        args.center_contact,
        args.contact_threshold,
    )
    plot_contact_projection_comparison(
        snapshots,
        approximations,
        radii,
        residuals,
        contact_rows,
        plot_paths["projection_contact_comparison"],
        contact_x_coordinates,
        args.contact_x_mode,
        args.contact_fixed_extent,
        args.center_contact,
        args.contact_threshold,
    )

    print(f"Loaded snapshots from: {dataset_source}")
    print(f"Input snapshot matrix: {snapshots.shape}")
    print(f"Train/test split: {train_indices.size} train, {test_indices.size} unseen test")
    print(f"CPG relative epsilon: {args.epsilon:g}")
    print(f"Selected basis size R: {basis.shape[0]}")
    print(f"Selected rows: {selected_indices}")
    print(f"Selected radii: {[float(v) for v in selected_radii]}")
    print(f"Final max residual: {float(np.max(residuals)):.6e}")
    print(f"Final max relative residual: {float(np.max(relative_residuals)):.6e}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Outputs written to: {args.output_dir}")
    if contact_x_coordinates is None:
        print(f"Contact profile x-axis: '{args.contact_x_mode}' mode")
        if args.center_contact:
            print("Contact patches centered for display only.")
    else:
        print(f"Contact profile x-axis: {args.contact_x_coordinates}")
    for label, path in {**input_plot_paths, **data_paths, **plot_paths}.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
