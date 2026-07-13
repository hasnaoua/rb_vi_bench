from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from greedy.pipelines.component_sweep import (
    fit_angle_fixed_components,
    fit_cpg_fixed_components,
    fit_mcpg_fixed_components,
    record_new_angles,
    stopping_scale,
)
from greedy.pipelines.lambda_cpg import project_snapshots_onto_basis
from greedy.datasets.physics_dataset import (
    DEFAULT_DATASET,
    DEFAULT_GRID_SHAPE,
    DEFAULT_INPUT,
    HEIGHT_MM,
    SECTOR_ANGLE_RAD,
    build_physics_dataset,
    reshape_contact_surface,
    save_physics_dataset,
    write_report as write_dataset_report,
)
from greedy.core.reduction_common import (
    basis_condition_number,
    max_projection_error_history,
    nice_indices,
    plot_angle_vs_components,
    plot_error_gain,
    write_angle_history_csv,
    write_error_gain_csv,
)


DEFAULT_OUTPUT_DIR = Path("results/physics/reduction")
ADG_NAME = "Angular Defect Greedy"
MCPG_NAME = "mCPG"


def load_or_build_dataset(
    dataset_path: Path,
    input_path: Path,
    *,
    rebuild: bool,
    extra_columns: str,
    active_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if rebuild or not dataset_path.exists():
        dataset = build_physics_dataset(
            input_path,
            extra_columns=extra_columns,
            active_threshold=active_threshold,
        )
        save_physics_dataset(dataset, dataset_path)
        write_dataset_report(
            dataset,
            dataset_path.parent / f"{dataset_path.stem}_report.txt",
        )
        source = str(input_path)
    else:
        dataset = dict(np.load(dataset_path, allow_pickle=False))
        source = str(dataset_path)

    snapshots = np.asarray(dataset["snapshots"], dtype=float)
    displacements = np.asarray(dataset["displacements"], dtype=float)
    original_indices = np.asarray(dataset["original_indices"], dtype=int)

    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got {snapshots.shape}")
    if displacements.ndim != 1 or displacements.size != snapshots.shape[0]:
        raise ValueError("displacements must match snapshot rows")
    if original_indices.ndim != 1 or original_indices.size != snapshots.shape[0]:
        raise ValueError("original_indices must match snapshot rows")
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshots contain non-finite values")

    return snapshots, displacements, original_indices, source


def parse_methods(raw: str | list[str]) -> list[str]:
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
    else:
        parts = raw

    aliases = {
        "cpg": "CPG",
        "adg": ADG_NAME,
        "angular_defect_greedy": ADG_NAME,
        "mcpg": MCPG_NAME,
        "modified_cpg": MCPG_NAME,
    }
    methods: list[str] = []
    for part in parts:
        key = str(part).lower()
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown method: {part}")
        method = aliases[key]
        if method not in methods:
            methods.append(method)
    return methods


def physics_residual_metrics(
    snapshots: np.ndarray,
    reconstructions: np.ndarray,
    *,
    norm_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.linalg.norm(snapshots - reconstructions, axis=1)
    norms = np.linalg.norm(snapshots, axis=1)
    denominator = np.maximum(norms, norm_floor)
    relative = residuals / denominator
    return residuals, relative


def run_method(
    method: str,
    snapshots: np.ndarray,
    components: int,
    *,
    zero_tol: float,
    epsilon: float | None = None,
    basis: np.ndarray | None = None,
    selected_indices: list[int] | None = None,
    adg_normalize_snapshots: bool = False,
) -> dict[str, np.ndarray | list[int] | float | str]:
    start = time.perf_counter()

    if method == "CPG":
        basis, selected, new_angles = fit_cpg_fixed_components(
            snapshots,
            components,
            zero_tol=zero_tol,
            epsilon=epsilon,
            basis=basis,
            selected_indices=selected_indices,
        )
        fit_seconds = time.perf_counter() - start
        reconstructions, coefficients, residuals = project_snapshots_onto_basis(
            snapshots,
            basis,
        )
        return {
            "method": method,
            "basis": basis,
            "coefficients": coefficients,
            "reconstructions": reconstructions,
            "residuals": residuals,
            "selected_indices": selected,
            "new_angles": new_angles,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - start,
        }

    if method == ADG_NAME:
        basis, selected, new_angles = fit_angle_fixed_components(
            snapshots,
            components,
            zero_tol=zero_tol,
            epsilon=epsilon,
            basis=basis,
            selected_indices=selected_indices,
            normalize_snapshots=adg_normalize_snapshots,
        )
        fit_seconds = time.perf_counter() - start
        reconstructions, coefficients, residuals = project_snapshots_onto_basis(
            snapshots,
            basis,
        )
        return {
            "method": method,
            "basis": basis,
            "coefficients": coefficients,
            "reconstructions": reconstructions,
            "residuals": residuals,
            "selected_indices": selected,
            "new_angles": new_angles,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - start,
        }

    if method == MCPG_NAME:
        basis, selected, new_angles = fit_mcpg_fixed_components(
            snapshots,
            components,
            zero_tol=zero_tol,
            epsilon=epsilon,
            basis=basis,
            selected_indices=selected_indices,
        )
        fit_seconds = time.perf_counter() - start
        reconstructions, coefficients, residuals = project_snapshots_onto_basis(
            snapshots,
            basis,
        )
        return {
            "method": method,
            "basis": basis,
            "coefficients": coefficients,
            "reconstructions": reconstructions,
            "residuals": residuals,
            "selected_indices": selected,
            "new_angles": new_angles,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - start,
        }

    raise ValueError(f"unsupported method: {method}")


def method_slug(method: str) -> str:
    if method == ADG_NAME:
        return "adg"
    return method.lower().replace(" ", "_")


def write_sweep_metrics_csv(
    rows: list[dict[str, np.ndarray | list[int] | float | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "requested_components",
                "components",
                "stopping_residual",
                "stopping_delta",
                "fit_seconds",
                "total_seconds",
                "condition_number",
                "max_residual",
                "mean_residual",
                "median_residual",
                "max_relative_residual",
                "mean_relative_residual",
                "median_relative_residual",
                "selected_rows",
                "selected_original_columns",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row.get("requested_components", row["components"]),
                    row["components"],
                    f"{float(row['stopping_residual']):.18e}",
                    f"{float(row['stopping_delta']):.18e}",
                    f"{float(row['fit_seconds']):.9f}",
                    f"{float(row['total_seconds']):.9f}",
                    f"{float(row.get('condition_number', float('nan'))):.18e}",
                    f"{float(row['max_residual']):.18e}",
                    f"{float(row['mean_residual']):.18e}",
                    f"{float(row['median_residual']):.18e}",
                    f"{float(row['max_relative_residual']):.18e}",
                    f"{float(row['mean_relative_residual']):.18e}",
                    f"{float(row['median_relative_residual']):.18e}",
                    row["selected_indices"],
                    row.get("selected_original_columns", ""),
                ]
            )


def plot_sweep_metrics(
    rows: list[dict[str, np.ndarray | list[int] | float | str]],
    output_path: Path,
    *,
    error_key: str = "max_relative_residual",
    error_label: str = "max relative residual",
    title: str = "Physics epsilon sweep: max relative residual vs components",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for method in methods:
        method_rows = [row for row in rows if str(row["method"]) == method]
        method_rows.sort(key=lambda row: int(row["components"]))
        x = np.array([int(row["components"]) for row in method_rows], dtype=float)
        y_error = np.array(
            [float(row[error_key]) for row in method_rows],
            dtype=float,
        )
        y_time = np.array([float(row["total_seconds"]) for row in method_rows], dtype=float)
        axes[0].plot(x, y_error, marker="o", linewidth=1.4, markersize=3.2, label=method)
        axes[1].plot(x, y_time, marker="o", linewidth=1.4, markersize=3.2, label=method)

    axes[0].set_title(title)
    axes[0].set_ylabel(error_label)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.28)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Physics epsilon sweep: cumulative runtime vs components")
    axes[1].set_xlabel("components reached by relative-epsilon greedy growth")
    axes[1].set_ylabel("cumulative seconds")
    axes[1].grid(True, which="both", alpha=0.28)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def error_histories_from_final_bases(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    *,
    relative: bool = False,
    norm_floor: float = 0.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for result in results:
        basis = np.asarray(result["basis"], dtype=float)
        if basis.size == 0:
            continue
        histories[str(result["method"])] = max_projection_error_history(
            snapshots,
            basis,
            relative=relative,
            norm_floor=norm_floor,
        )
    return histories


def save_method_arrays(
    result: dict[str, np.ndarray | list[int] | float | str],
    output_dir: Path,
) -> dict[str, Path]:
    slug = method_slug(str(result["method"]))
    paths = {
        "basis": output_dir / f"{slug}_basis.npy",
        "coefficients": output_dir / f"{slug}_coefficients.npy",
        "reconstructions": output_dir / f"{slug}_reconstructions.npy",
        "selected": output_dir / f"{slug}_selected_indices.txt",
    }
    np.save(paths["basis"], np.asarray(result["basis"], dtype=float))
    np.save(paths["coefficients"], np.asarray(result["coefficients"], dtype=float))
    np.save(paths["reconstructions"], np.asarray(result["reconstructions"], dtype=float))
    np.savetxt(
        paths["selected"],
        np.asarray(result["selected_original_indices"], dtype=int),
        fmt="%d",
    )
    return paths


def write_metrics_csv(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    original_indices: np.ndarray,
    output_path: Path,
    *,
    relative_norm_floor: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "components",
                "stopping_residual",
                "stopping_delta",
                "fit_seconds",
                "total_seconds",
                "condition_number",
                "max_residual",
                "mean_residual",
                "median_residual",
                "max_relative_residual",
                "mean_relative_residual",
                "median_relative_residual",
                "selected_rows",
                "selected_original_columns",
                "selected_displacements_mm",
            ]
        )

        for result in results:
            reconstructions = np.asarray(result["reconstructions"], dtype=float)
            residuals, relative = physics_residual_metrics(
                snapshots,
                reconstructions,
                norm_floor=relative_norm_floor,
            )
            selected = [int(v) for v in result["selected_original_indices"]]
            writer.writerow(
                [
                    result["method"],
                    np.asarray(result["basis"]).shape[0],
                    f"{float(result['stopping_residual']):.18e}",
                    f"{float(result['stopping_delta']):.18e}",
                    f"{float(result['fit_seconds']):.9f}",
                    f"{float(result['total_seconds']):.9f}",
                    f"{float(result.get('condition_number', float('nan'))):.18e}",
                    f"{float(np.max(residuals)):.18e}",
                    f"{float(np.mean(residuals)):.18e}",
                    f"{float(np.median(residuals)):.18e}",
                    f"{float(np.max(relative)):.18e}",
                    f"{float(np.mean(relative)):.18e}",
                    f"{float(np.median(relative)):.18e}",
                    selected,
                    [int(original_indices[i]) for i in selected],
                    [float(displacements[i]) for i in selected],
                ]
            )


def write_residuals_csv(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    relative_norm_floor: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "displacement_mm", "method", "residual", "relative"])
        for result in results:
            residuals, relative = physics_residual_metrics(
                snapshots,
                np.asarray(result["reconstructions"], dtype=float),
                norm_floor=relative_norm_floor,
            )
            for row, displacement in enumerate(displacements):
                writer.writerow(
                    [
                        row,
                        f"{float(displacement):.12g}",
                        result["method"],
                        f"{float(residuals[row]):.18e}",
                        f"{float(relative[row]):.18e}",
                    ]
                )


def plot_residual_comparison(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    relative_norm_floor: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for result in results:
        residuals, relative = physics_residual_metrics(
            snapshots,
            np.asarray(result["reconstructions"], dtype=float),
            norm_floor=relative_norm_floor,
        )
        axes[0].plot(
            displacements,
            residuals,
            marker="o",
            linewidth=1.4,
            markersize=3.2,
            label=str(result["method"]),
        )
        relative_for_plot = np.maximum(relative, np.finfo(float).tiny)
        axes[1].plot(
            displacements,
            relative_for_plot,
            marker="o",
            linewidth=1.4,
            markersize=3.2,
            label=str(result["method"]),
        )

    axes[0].set_title("Physics reconstruction residuals on the full dataset")
    axes[0].set_ylabel("absolute residual")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.28)
    axes[0].legend()
    axes[1].set_xlabel("imposed displacement [mm]")
    axes[1].set_ylabel("relative residual")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.28)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_examples(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    max_examples: int,
) -> None:
    rows = nice_indices(snapshots.shape[0], max_examples)
    if rows.size == 0:
        return

    method_count = len(results)
    fig, axes = plt.subplots(
        rows.size,
        method_count,
        figsize=(4.6 * method_count, 2.8 * rows.size),
        squeeze=False,
        sharex=True,
    )
    x = np.arange(snapshots.shape[1])

    for row_index, row in enumerate(rows):
        for col_index, result in enumerate(results):
            ax = axes[row_index, col_index]
            reconstruction = np.asarray(result["reconstructions"], dtype=float)[row]
            residual = float(np.linalg.norm(snapshots[row] - reconstruction))
            ax.plot(x, snapshots[row], color="#17202a", linewidth=1.2, label="Actual HF")
            ax.plot(
                x,
                reconstruction,
                color="#0f766e",
                linewidth=1.1,
                linestyle="--",
                label=str(result["method"]),
            )
            ax.set_title(
                f"{result['method']}: u={displacements[row]:.3g} mm, "
                f"err={residual:.2e}"
            )
            ax.grid(True, alpha=0.23)
            if row_index == rows.size - 1:
                ax.set_xlabel("contact node index")
            if col_index == 0:
                ax.set_ylabel("contact force")
            if row_index == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Physics HF vs reduced reconstruction examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basis_heatmaps(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    output_path: Path,
) -> None:
    if not results:
        return

    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(12, max(3.2, 2.7 * len(results))),
        squeeze=False,
    )
    tiny = np.finfo(float).tiny
    for ax, result in zip(axes.ravel(), results):
        basis = np.asarray(result["basis"], dtype=float)
        if basis.size == 0:
            ax.text(0.5, 0.5, "No basis", ha="center", va="center")
            ax.set_axis_off()
            continue
        image = ax.imshow(
            np.log10(np.abs(basis) + tiny),
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        ax.set_title(f"{result['method']} basis/components")
        ax.set_xlabel("contact node index")
        ax.set_ylabel("component")
        fig.colorbar(image, ax=ax, pad=0.01, label=r"$\log_{10}(|value|)$")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _surface_log(values: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.abs(values))


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.5])
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _example_rows(
    snapshots: np.ndarray,
    max_examples: int,
) -> np.ndarray:
    rows = nice_indices(snapshots.shape[0], max_examples)
    norms = np.linalg.norm(snapshots, axis=1)
    active = np.flatnonzero(norms > 0.1)
    if active.size:
        rows = np.unique(np.append(rows, active[0]))
    return np.array(sorted(set(int(row) for row in rows)), dtype=int)


def plot_reconstruction_surface_maps(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    max_examples: int,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
) -> None:
    rows = _example_rows(snapshots, max_examples)
    if rows.size == 0:
        return

    panels: list[np.ndarray] = []
    for row in rows:
        panels.append(_surface_log(reshape_contact_surface(snapshots[row], grid_shape)))
        for result in results:
            reconstruction = np.asarray(result["reconstructions"], dtype=float)[row]
            panels.append(_surface_log(reshape_contact_surface(reconstruction, grid_shape)))
    vmin, vmax = _robust_limits(np.asarray(panels))

    ncols = len(results) + 1
    fig, axes = plt.subplots(
        rows.size,
        ncols,
        figsize=(3.2 * ncols, 2.75 * rows.size),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for row_index, row in enumerate(rows):
        for col_index in range(ncols):
            ax = axes[row_index, col_index]
            if col_index == 0:
                values = snapshots[row]
                title = "Actual HF"
            else:
                result = results[col_index - 1]
                values = np.asarray(result["reconstructions"], dtype=float)[row]
                title = str(result["method"])
            surface = _surface_log(reshape_contact_surface(values, grid_shape))
            last_image = ax.imshow(
                surface,
                origin="lower",
                aspect="auto",
                extent=[0.0, HEIGHT_MM, 0.0, np.degrees(SECTOR_ANGLE_RAD)],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.set_ylabel(f"u_z={displacements[row]:.3g} mm\ntheta [deg]")
            if row_index == rows.size - 1:
                ax.set_xlabel("z [mm]")

    fig.suptitle("Input and reconstruction maps on the unwrapped quarter-cylinder")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label(r"$\log_{10}(1+|contact force|)$")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_error_surface_maps(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    output_path: Path,
    *,
    max_examples: int,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    relative: bool = False,
    norm_floor: float = 0.0,
) -> None:
    rows = _example_rows(snapshots, max_examples)
    if rows.size == 0 or not results:
        return

    panels: list[np.ndarray] = []
    for row in rows:
        denominator = max(float(np.linalg.norm(snapshots[row])), float(norm_floor))
        for result in results:
            reconstruction = np.asarray(result["reconstructions"], dtype=float)[row]
            error = np.abs(snapshots[row] - reconstruction)
            if relative and denominator > 0.0:
                error = error / denominator
            panels.append(_surface_log(reshape_contact_surface(error, grid_shape)))
    vmin, vmax = _robust_limits(np.asarray(panels))

    fig, axes = plt.subplots(
        rows.size,
        len(results),
        figsize=(3.2 * len(results), 2.75 * rows.size),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for row_index, row in enumerate(rows):
        denominator = max(float(np.linalg.norm(snapshots[row])), float(norm_floor))
        for col_index, result in enumerate(results):
            ax = axes[row_index, col_index]
            reconstruction = np.asarray(result["reconstructions"], dtype=float)[row]
            error = np.abs(snapshots[row] - reconstruction)
            if relative and denominator > 0.0:
                error = error / denominator
            surface = _surface_log(reshape_contact_surface(error, grid_shape))
            last_image = ax.imshow(
                surface,
                origin="lower",
                aspect="auto",
                extent=[0.0, HEIGHT_MM, 0.0, np.degrees(SECTOR_ANGLE_RAD)],
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(str(result["method"]))
            if col_index == 0:
                ax.set_ylabel(f"u_z={displacements[row]:.3g} mm\ntheta [deg]")
            if row_index == rows.size - 1:
                ax.set_xlabel("z [mm]")

    if relative:
        fig.suptitle("Relative reconstruction error maps on the quarter-cylinder")
        colorbar_label = r"$\log_{10}(1+|error|/\max(\|x\|_2,\tau))$"
    else:
        fig.suptitle("Absolute reconstruction error maps on the quarter-cylinder")
        colorbar_label = r"$\log_{10}(1+|error|)$"
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_basis_surface_maps(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    output_path: Path,
    *,
    grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
    max_components: int = 6,
) -> None:
    if not results:
        return

    panel_count = max(
        1,
        max(
            min(max_components, np.asarray(result["basis"], dtype=float).shape[0])
            for result in results
        ),
    )
    panels: list[np.ndarray] = []
    for result in results:
        basis = np.asarray(result["basis"], dtype=float)
        for component in range(min(panel_count, basis.shape[0])):
            panels.append(_surface_log(reshape_contact_surface(basis[component], grid_shape)))
    vmin, vmax = _robust_limits(np.asarray(panels)) if panels else (0.0, 1.0)

    fig, axes = plt.subplots(
        len(results),
        panel_count,
        figsize=(3.05 * panel_count, 2.75 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for row_index, result in enumerate(results):
        basis = np.asarray(result["basis"], dtype=float)
        for component in range(panel_count):
            ax = axes[row_index, component]
            if component >= basis.shape[0]:
                ax.axis("off")
                continue
            surface = _surface_log(reshape_contact_surface(basis[component], grid_shape))
            last_image = ax.imshow(
                surface,
                origin="lower",
                aspect="auto",
                extent=[0.0, HEIGHT_MM, 0.0, np.degrees(SECTOR_ANGLE_RAD)],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(f"component {component + 1}")
            if component == 0:
                ax.set_ylabel(f"{result['method']}\ntheta [deg]")
            if row_index == len(results) - 1:
                ax.set_xlabel("z [mm]")

    fig.suptitle("Basis/component maps on the unwrapped quarter-cylinder")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.8)
        cbar.set_label(r"$\log_{10}(1+|basis value|)$")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(
    results: list[dict[str, np.ndarray | list[int] | float | str]],
    snapshots: np.ndarray,
    displacements: np.ndarray,
    original_indices: np.ndarray,
    output_path: Path,
    *,
    dataset_source: str,
    components: int,
    relative_norm_floor: float,
    adg_normalize_snapshots: bool = False,
) -> None:
    lines = [
        "Physics reduction experiment report",
        "",
        f"dataset_source: {dataset_source}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"surface_grid_shape_theta_by_z: {DEFAULT_GRID_SHAPE}",
        f"surface_domain: theta=0..{float(np.degrees(SECTOR_ANGLE_RAD)):.6g} deg, "
        f"z=0..{HEIGHT_MM:.6g} mm",
        f"components: {components}",
        f"snapshot_count: {snapshots.shape[0]}",
        f"adg_normalize_snapshots: {adg_normalize_snapshots} "
        "(ADG's stopping/selection criterion fit on per-snapshot unit-normalized "
        "snapshots instead of one absolute tolerance shared across all snapshots; "
        "only affects ADG, not CPG/mCPG; basis vectors are physical-scale either way)",
        "basis_construction: CPG, ADG, and mCPG are fitted on the whole physics dataset; "
        "no train/test split is used.",
        "sweep_rule: grow each greedy basis once, reusing previous components; "
        "stopping_delta is the max residual after the final prefix divided by "
        "max snapshot norm; this relative tolerance would stop at that "
        "component count.",
        f"displacement_min_mm: {float(np.min(displacements)):.12g}",
        f"displacement_max_mm: {float(np.max(displacements)):.12g}",
        f"relative_norm_floor: {relative_norm_floor:.12g}",
        "",
        "Method summary:",
    ]

    for result in results:
        reconstructions = np.asarray(result["reconstructions"], dtype=float)
        residuals, relative = physics_residual_metrics(
            snapshots,
            reconstructions,
            norm_floor=relative_norm_floor,
        )
        selected = [int(v) for v in result["selected_original_indices"]]
        condition_number = result.get("condition_number", float("nan"))
        lines.append(
            "- "
            f"{result['method']}: "
            f"R={np.asarray(result['basis']).shape[0]}, "
            f"delta={float(result['stopping_delta']):.6e}, "
            f"max_relative={float(np.max(relative)):.6e}, "
            f"mean_relative={float(np.mean(relative)):.6e}, "
            f"condition_number={float(condition_number):.6e}, "
            f"fit_seconds={float(result['fit_seconds']):.6f}"
        )
        if selected:
            lines.append(
                "  selected_original_columns: "
                f"{[int(original_indices[i]) for i in selected]}"
            )
            lines.append(
                "  selected_displacements_mm: "
                f"{[float(displacements[i]) for i in selected]}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPG and angular defect greedy (ADG) on the physics contact dataset "
            "by growing relative-epsilon greedy prefixes."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--extra-columns",
        choices=["drop-leading", "drop-trailing", "keep-index"],
        default="drop-leading",
    )
    parser.add_argument("--active-threshold", type=float, default=None)
    parser.add_argument("--components", type=int, default=None)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Optional relative stopping tolerance. Omit it to grow to "
            "--max-components and report the stopping delta reached at each prefix."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["cpg", "adg", "mcpg"],
        help="Methods to run: cpg, adg, mcpg. Default: cpg adg mcpg.",
    )
    parser.add_argument("--zero-tol", type=float, default=1e-12)
    parser.add_argument(
        "--adg-normalize-snapshots",
        action="store_true",
        help=(
            "Fit ADG's stopping/selection criterion on per-snapshot unit-normalized "
            "snapshots instead of one absolute tolerance shared across all snapshots "
            "(epsilon * max_x ||x||). Returned basis vectors stay physical-scale "
            "either way. Only affects ADG; CPG and mCPG are unaffected."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument(
        "--relative-norm-floor",
        type=float,
        default=0.1,
        help=(
            "Denominator floor for relative residuals, useful for nearly-zero "
            "no-contact snapshots."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = parse_methods(args.methods)

    if args.components is not None and args.components <= 0:
        raise ValueError("--components must be positive")
    if args.min_components is not None and args.min_components <= 0:
        raise ValueError("--min-components must be positive")
    if args.max_components is not None and args.min_components is not None and args.max_components < args.min_components:
        raise ValueError("--max-components must be >= --min-components")
    if args.step <= 0:
        raise ValueError("--step must be positive")
    snapshots, displacements, original_indices, dataset_source = load_or_build_dataset(
        args.dataset,
        args.input,
        rebuild=args.rebuild_dataset,
        extra_columns=args.extra_columns,
        active_threshold=args.active_threshold,
    )
    fit_snapshots = snapshots
    fit_stopping_scale = stopping_scale(fit_snapshots)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.components is not None:
        components_values = [min(args.components, fit_snapshots.shape[0], fit_snapshots.shape[1])]
    else:
        components_values = list(
            range(
                args.min_components,
                min(args.max_components, fit_snapshots.shape[0], fit_snapshots.shape[1]) + 1,
                args.step,
            )
        )

    method_states: dict[str, dict[str, np.ndarray | list[int] | None]] = {}
    cumulative_times: dict[str, dict[str, float]] = {}
    sweep_rows: list[dict[str, np.ndarray | list[int] | float | str]] = []
    final_results: list[dict[str, np.ndarray | list[int] | float | str]] = []
    angle_rows: dict[str, list[tuple[int, float]]] = {}

    for component_count in components_values:
        print(f"Running physics sweep at R={component_count} ...")
        for method in methods:
            state = method_states.setdefault(method, {"basis": None, "selected_indices": None})
            basis_state = None if state["basis"] is None else np.asarray(state["basis"], dtype=float)
            selected_state = None if state["selected_indices"] is None else [int(v) for v in state["selected_indices"]]
            result = run_method(
                method,
                fit_snapshots,
                component_count,
                zero_tol=args.zero_tol,
                epsilon=args.epsilon,
                basis=basis_state,
                selected_indices=selected_state,
                adg_normalize_snapshots=args.adg_normalize_snapshots,
            )
            incremental_fit_seconds = float(result["fit_seconds"])
            project_start = time.perf_counter()
            full_reconstructions, full_coefficients, full_residuals = project_snapshots_onto_basis(
                snapshots,
                np.asarray(result["basis"], dtype=float),
            )
            full_project_seconds = time.perf_counter() - project_start
            incremental_total_seconds = incremental_fit_seconds + full_project_seconds
            cumulative_entry = cumulative_times.setdefault(
                method,
                {"fit": 0.0, "total": 0.0},
            )
            cumulative_entry["fit"] += incremental_fit_seconds
            cumulative_entry["total"] += incremental_total_seconds
            selected_rows = [int(v) for v in result["selected_indices"]]
            record_new_angles(angle_rows, method, selected_rows, result.get("new_angles", []))
            result["condition_number"] = basis_condition_number(np.asarray(result["basis"], dtype=float))
            result["coefficients"] = full_coefficients
            result["reconstructions"] = full_reconstructions
            result["residuals"] = full_residuals
            result["selected_indices"] = selected_rows
            result["selected_original_indices"] = selected_rows
            result["selected_original_columns"] = [
                int(original_indices[row]) for row in selected_rows
            ]
            result["requested_components"] = component_count
            result["components"] = len(selected_rows)
            result["max_residual"] = float(np.max(full_residuals))
            result["mean_residual"] = float(np.mean(full_residuals))
            result["median_residual"] = float(np.median(full_residuals))
            stopping_residual = float(np.max(full_residuals))
            result["stopping_residual"] = stopping_residual
            result["stopping_delta"] = (
                stopping_residual / fit_stopping_scale if fit_stopping_scale > 0.0 else 0.0
            )
            result["fit_seconds"] = cumulative_entry["fit"]
            result["total_seconds"] = cumulative_entry["total"]
            residuals, relative = physics_residual_metrics(
                snapshots,
                np.asarray(result["reconstructions"], dtype=float),
                norm_floor=args.relative_norm_floor,
            )
            result["max_relative_residual"] = float(np.max(relative))
            result["mean_relative_residual"] = float(np.mean(relative))
            result["median_relative_residual"] = float(np.median(relative))
            sweep_rows.append(result)
            final_results = [row for row in final_results if str(row["method"]) != method]
            final_results.append(result)
            state["basis"] = np.asarray(result["basis"], dtype=float)
            state["selected_indices"] = selected_rows

    results = [
        result for result in final_results if str(result["method"]) in {method for method in methods}
    ]
    results.sort(key=lambda row: str(row["method"]))

    metrics_path = args.output_dir / "physics_method_metrics.csv"
    residuals_csv = args.output_dir / "physics_residuals_by_snapshot.csv"
    report_path = args.output_dir / "physics_reduction_report.txt"
    residual_plot = args.output_dir / "physics_residual_comparison.png"
    examples_plot = args.output_dir / "physics_reconstruction_examples.png"
    basis_plot = args.output_dir / "physics_basis_heatmaps.png"
    surface_examples_plot = args.output_dir / "physics_reconstruction_surface_maps.png"
    surface_errors_plot = args.output_dir / "physics_reconstruction_error_maps.png"
    surface_relative_errors_plot = (
        args.output_dir / "physics_reconstruction_relative_error_maps.png"
    )
    surface_basis_plot = args.output_dir / "physics_basis_surface_maps.png"
    sweep_metrics_path = args.output_dir / "physics_component_sweep_metrics.csv"
    sweep_plot_path = args.output_dir / "physics_component_sweep_summary.png"
    sweep_absolute_plot_path = args.output_dir / "physics_component_sweep_absolute_summary.png"
    error_gain_csv = args.output_dir / "physics_error_gain.csv"
    error_gain_plot = args.output_dir / "physics_error_gain.png"
    absolute_error_gain_csv = args.output_dir / "physics_absolute_error_gain.csv"
    absolute_error_gain_plot = args.output_dir / "physics_absolute_error_gain.png"
    angle_history_csv = args.output_dir / "physics_angle_history.csv"
    angle_history_plot = args.output_dir / "physics_angle_history.png"

    for result in results:
        save_method_arrays(result, args.output_dir)

    write_metrics_csv(
        results,
        snapshots,
        displacements,
        original_indices,
        metrics_path,
        relative_norm_floor=args.relative_norm_floor,
    )
    write_residuals_csv(
        results,
        snapshots,
        displacements,
        residuals_csv,
        relative_norm_floor=args.relative_norm_floor,
    )
    write_report(
        results,
        snapshots,
        displacements,
        original_indices,
        report_path,
        dataset_source=dataset_source,
        components=int(results[0]["components"] if results else (args.components or args.min_components)),
        relative_norm_floor=args.relative_norm_floor,
        adg_normalize_snapshots=args.adg_normalize_snapshots,
    )
    plot_residual_comparison(
        results,
        snapshots,
        displacements,
        residual_plot,
        relative_norm_floor=args.relative_norm_floor,
    )
    plot_reconstruction_examples(
        results,
        snapshots,
        displacements,
        examples_plot,
        max_examples=args.max_examples,
    )

    if sweep_rows:
        write_sweep_metrics_csv(sweep_rows, sweep_metrics_path)
        plot_sweep_metrics(sweep_rows, sweep_plot_path)
        plot_sweep_metrics(
            sweep_rows,
            sweep_absolute_plot_path,
            error_key="max_residual",
            error_label="max absolute residual",
            title="Physics epsilon sweep: max absolute residual vs components",
        )

    angle_histories = {
        method: (
            np.array([n for n, _ in angle_rows.get(method, [])], dtype=float),
            np.array([angle for _, angle in angle_rows.get(method, [])], dtype=float),
        )
        for method in methods
    }
    write_angle_history_csv(angle_histories, angle_history_csv)
    plot_angle_vs_components(
        angle_histories,
        angle_history_plot,
        title="Selected-snapshot angle vs. component count (CPG vs ADG vs mCPG, physics dataset)",
    )

    error_histories = error_histories_from_final_bases(
        results,
        snapshots,
        relative=True,
        norm_floor=args.relative_norm_floor,
    )
    absolute_error_histories = error_histories_from_final_bases(
        results,
        snapshots,
        relative=False,
    )
    if error_histories:
        write_error_gain_csv(error_histories, error_gain_csv)
        plot_error_gain(
            error_histories,
            error_gain_plot,
            title="Physics dataset relative error gain from final basis prefixes",
            xlabel="cone size n",
            gain_mode="relative",
            error_ylabel=(
                r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/"
                r"\max(\|x\|_2,\tau)$"
            ),
        )
    if absolute_error_histories:
        write_error_gain_csv(absolute_error_histories, absolute_error_gain_csv)
        plot_error_gain(
            absolute_error_histories,
            absolute_error_gain_plot,
            title="Physics dataset absolute error gain from final basis prefixes",
            xlabel="cone size n",
            gain_mode="relative",
            error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
        )
    plot_basis_heatmaps(results, basis_plot)
    plot_reconstruction_surface_maps(
        results,
        snapshots,
        displacements,
        surface_examples_plot,
        max_examples=args.max_examples,
    )
    plot_reconstruction_error_surface_maps(
        results,
        snapshots,
        displacements,
        surface_errors_plot,
        max_examples=args.max_examples,
    )
    plot_reconstruction_error_surface_maps(
        results,
        snapshots,
        displacements,
        surface_relative_errors_plot,
        max_examples=args.max_examples,
        relative=True,
        norm_floor=args.relative_norm_floor,
    )
    plot_basis_surface_maps(results, surface_basis_plot)

    print(f"Loaded snapshots from: {dataset_source}")
    print(f"Snapshot matrix: {snapshots.shape}")
    print("Physics reduction uses the whole dataset for basis construction and residuals.")
    print(f"Methods: {', '.join(str(result['method']) for result in results)}")
    print(f"Outputs written to: {args.output_dir}")
    print(f"  metrics: {metrics_path}")
    print(f"  residuals: {residuals_csv}")
    print(f"  report: {report_path}")
    print(f"  residual_plot: {residual_plot}")
    print(f"  examples_plot: {examples_plot}")
    print(f"  basis_plot: {basis_plot}")
    print(f"  surface_examples_plot: {surface_examples_plot}")
    print(f"  surface_errors_plot: {surface_errors_plot}")
    print(f"  surface_relative_errors_plot: {surface_relative_errors_plot}")
    print(f"  surface_basis_plot: {surface_basis_plot}")
    if sweep_rows:
        print(f"  sweep_metrics: {sweep_metrics_path}")
        print(f"  sweep_plot: {sweep_plot_path}")
        print(f"  sweep_absolute_plot: {sweep_absolute_plot_path}")
        print(f"  error_gain_csv: {error_gain_csv}")
        print(f"  error_gain_plot: {error_gain_plot}")
        print(f"  absolute_error_gain_csv: {absolute_error_gain_csv}")
        print(f"  absolute_error_gain_plot: {absolute_error_gain_plot}")
    print(f"  angle_history_csv: {angle_history_csv}")
    print(f"  angle_history_plot: {angle_history_plot}")
    for result in results:
        print(f"  final_condition_number[{result['method']}]: {float(result.get('condition_number', float('nan'))):.6e}")


if __name__ == "__main__":
    main()
