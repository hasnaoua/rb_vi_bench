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

from CPG import CPG
from lambda_cpg_pipeline import project_snapshots_onto_basis
from modified_angle_greedy import AngularDefectGreedy
from reduction_common import (
    DEFAULT_DATASET,
    fem_sols_train_test_split,
    load_lambda_dataset,
    max_projection_error_history,
    parameter_train_test_split,
    plot_error_gain,
    residual_metrics,
    write_error_gain_csv,
)


DEFAULT_OUTPUT_DIR = Path("results/lambda/component_sweep")
ADG_NAME = "Angular Defect Greedy"


def fit_cpg_fixed_components(
    snapshots: np.ndarray,
    components: int,
    *,
    zero_tol: float,
    epsilon: float | None = None,
    basis: np.ndarray | None = None,
    selected_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    model = CPG(snapshots=snapshots, epsilon=0.0 if epsilon is None else float(epsilon))
    if basis is not None and basis.size > 0:
        model._basis = [np.asarray(vec, dtype=float).copy() for vec in basis]
    else:
        model._basis = []

    if selected_indices is not None:
        model.selected_indices = [int(index) for index in selected_indices]
    else:
        model.selected_indices = []

    target_components = min(max(components, 0), snapshots.shape[0])
    if target_components <= len(model.selected_indices):
        basis_array = np.asarray(model._basis, dtype=float)
        if basis_array.size == 0:
            basis_array = np.empty((0, snapshots.shape[1]), dtype=float)
        return basis_array, [int(index) for index in model.selected_indices]

    for _ in range(target_components - len(model.selected_indices)):
        residuals = model._compute_residuals()
        if epsilon is not None and len(model.selected_indices) > 0:
            if float(np.max(residuals)) <= epsilon:
                break

        scores = residuals.copy()
        if model.selected_indices:
            scores[np.array(model.selected_indices, dtype=int)] = -np.inf

        candidate = int(np.argmax(scores))
        if not np.isfinite(scores[candidate]) or scores[candidate] <= zero_tol:
            break

        model._basis.append(snapshots[candidate].copy())
        model.selected_indices.append(candidate)

    basis_array = np.asarray(model._basis, dtype=float)
    if basis_array.size == 0:
        basis_array = np.empty((0, snapshots.shape[1]), dtype=float)
    return basis_array, [int(index) for index in model.selected_indices]


def fit_angle_fixed_components(
    snapshots: np.ndarray,
    components: int,
    *,
    zero_tol: float,
    epsilon: float | None = None,
    basis: np.ndarray | None = None,
    selected_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    model = AngularDefectGreedy(
        snapshots=snapshots,
        epsilon=zero_tol if epsilon is None else float(epsilon),
        zero_tol=zero_tol,
    )

    if components <= 0:
        return np.empty((0, snapshots.shape[1]), dtype=float), []

    if basis is not None and basis.size > 0:
        model._basis = [np.asarray(vec, dtype=float).copy() for vec in basis]
        model.selected_indices = [int(index) for index in (selected_indices or [])]
    elif components == 1:
        norms = np.linalg.norm(snapshots, axis=1)
        index = int(np.argmax(norms))
        if norms[index] <= zero_tol:
            return np.empty((0, snapshots.shape[1]), dtype=float), []
        return snapshots[index : index + 1].copy(), [index]
    else:
        model._init_basis()

    target_components = min(components, snapshots.shape[0])
    if target_components <= len(model.selected_indices):
        basis_array = np.asarray(model._basis, dtype=float)
        if basis_array.size == 0:
            basis_array = np.empty((0, snapshots.shape[1]), dtype=float)
        return basis_array, [int(index) for index in model.selected_indices]

    while len(model.selected_indices) < target_components:
        residuals, angles = model._compute_projection_metrics()
        if epsilon is not None and len(model.selected_indices) > 0:
            if float(np.max(residuals)) <= epsilon:
                break

        candidates = model._next_angle_candidates(residuals, angles)
        if not candidates:
            break

        candidate = candidates[0]
        model._basis.append(snapshots[candidate].copy())
        model.selected_indices.append(candidate)

    basis_array = np.asarray(model._basis, dtype=float)
    if basis_array.size == 0:
        basis_array = np.empty((0, snapshots.shape[1]), dtype=float)
    return basis_array, [int(index) for index in model.selected_indices]


def evaluate_basis(
    snapshots: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reconstructions, coefficients, residuals = project_snapshots_onto_basis(
        snapshots,
        basis,
    )
    _, relative = residual_metrics(snapshots, reconstructions)
    return reconstructions, coefficients, residuals, relative


def append_result(
    rows: list[dict[str, str]],
    *,
    method: str,
    requested_components: int,
    components: int,
    stopping_epsilon: float,
    fit_seconds: float,
    total_seconds: float,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    selected_indices: list[int] | None = None,
    selected_original_indices: list[int] | None = None,
    selected_radii: np.ndarray | None = None,
    train_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
) -> None:
    train_indices = np.array([], dtype=int) if train_indices is None else np.asarray(train_indices, dtype=int)
    test_indices = np.array([], dtype=int) if test_indices is None else np.asarray(test_indices, dtype=int)

    def _subset_stats(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
        if indices.size == 0:
            return float("nan"), float("nan"), float("nan")
        subset = values[indices]
        return float(np.max(subset)), float(np.mean(subset)), float(np.median(subset))

    train_relative = _subset_stats(relative_residuals, train_indices)
    test_relative = _subset_stats(relative_residuals, test_indices)
    rows.append(
        {
            "method": method,
            "requested_components": str(requested_components),
            "components": str(components),
            "stopping_epsilon": f"{stopping_epsilon:.18e}",
            "fit_seconds": f"{fit_seconds:.9f}",
            "total_seconds": f"{total_seconds:.9f}",
            "max_residual": f"{float(np.max(residuals)):.18e}",
            "mean_residual": f"{float(np.mean(residuals)):.18e}",
            "median_residual": f"{float(np.median(residuals)):.18e}",
            "max_relative_residual": f"{float(np.max(relative_residuals)):.18e}",
            "mean_relative_residual": f"{float(np.mean(relative_residuals)):.18e}",
            "median_relative_residual": f"{float(np.median(relative_residuals)):.18e}",
            "max_train_relative_residual": f"{train_relative[0]:.18e}",
            "mean_train_relative_residual": f"{train_relative[1]:.18e}",
            "median_train_relative_residual": f"{train_relative[2]:.18e}",
            "max_test_relative_residual": f"{test_relative[0]:.18e}",
            "mean_test_relative_residual": f"{test_relative[1]:.18e}",
            "median_test_relative_residual": f"{test_relative[2]:.18e}",
            "selected_indices": "" if selected_indices is None else str(selected_indices),
            "selected_original_indices": ""
            if selected_original_indices is None
            else str(selected_original_indices),
            "selected_radii": ""
            if selected_radii is None
            else str([float(value) for value in selected_radii]),
        }
    )


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "requested_components",
        "components",
        "stopping_epsilon",
        "fit_seconds",
        "total_seconds",
        "max_residual",
        "mean_residual",
        "median_residual",
        "max_relative_residual",
        "mean_relative_residual",
        "median_relative_residual",
        "max_train_relative_residual",
        "mean_train_relative_residual",
        "median_train_relative_residual",
        "max_test_relative_residual",
        "mean_test_relative_residual",
        "median_test_relative_residual",
        "selected_indices",
        "selected_original_indices",
        "selected_radii",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _series(
    rows: list[dict[str, str]],
    method: str,
    key: str,
) -> tuple[np.ndarray, np.ndarray]:
    method_rows = [row for row in rows if row["method"] == method]
    method_rows.sort(key=lambda row: int(row["components"]))
    x = np.array([int(row["components"]) for row in method_rows], dtype=float)
    y = np.array([float(row[key]) for row in method_rows], dtype=float)
    return x, y


def plot_metric(
    rows: list[dict[str, str]],
    methods: list[str],
    output_path: Path,
    *,
    keys: tuple[str, str],
    labels: tuple[str, str],
    title: str,
    log_y: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for method in methods:
        for ax, key, label in zip(axes, keys, labels):
            x, y = _series(rows, method, key)
            if x.size == 0:
                continue
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, label=method)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.28, which="both")
            if log_y:
                ax.set_yscale("log")

    axes[0].set_title(title)
    axes[1].set_xlabel("components reached by epsilon-driven greedy growth")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def error_histories_from_final_bases(
    snapshots: np.ndarray,
    bases: dict[str, np.ndarray | None],
    *,
    relative: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, basis in bases.items():
        if basis is None:
            continue
        basis = np.asarray(basis, dtype=float)
        if basis.size == 0:
            continue
        histories[method] = max_projection_error_history(
            snapshots,
            basis,
            relative=relative,
        )
    return histories


def write_report(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    dataset_source: str,
    snapshot_shape: tuple[int, int],
    components: list[int],
    methods: list[str],
    skipped: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    parameters: np.ndarray,
    split_strategy: str,
) -> None:
    lines = [
        "Component sweep comparison report",
        "",
        f"dataset_source: {dataset_source}",
        f"snapshot_matrix_shape: {snapshot_shape}",
        f"split_strategy: {split_strategy}",
        "validation: offline bases are built on training parameters only; "
        "test residuals use unseen held-out parameters.",
        f"train_count: {train_indices.size}",
        f"test_count: {test_indices.size}",
        f"train_parameters: {[float(parameters[i]) for i in train_indices]}",
        f"test_parameters: {[float(parameters[i]) for i in test_indices]}",
        f"components: {components}",
        f"methods: {methods}",
        "sweep_rule: grow each greedy basis once, reusing previous components; "
        "stopping_epsilon is the max residual after the prefix and is the "
        "tolerance that would stop at that component count.",
    ]
    if skipped:
        lines.append(f"skipped: {skipped}")

    lines.extend(["", "Best unseen-test max-relative residual by method:"])
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        best = min(method_rows, key=lambda row: float(row["max_test_relative_residual"]))
        lines.append(
            "- "
            f"{method}: R={best['components']}, "
            f"epsilon={float(best['stopping_epsilon']):.6e}, "
            f"test_max_relative={float(best['max_test_relative_residual']):.6e}, "
            f"test_mean_relative={float(best['mean_test_relative_residual']):.6e}, "
            f"total_seconds={float(best['total_seconds']):.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_methods(raw: str) -> list[str]:
    aliases = {
        "cpg": "CPG",
        "adg": ADG_NAME,
        "angular_defect_greedy": ADG_NAME,
    }
    methods: list[str] = []
    for part in raw.replace(",", " ").split():
        key = part.lower()
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown method: {part}")
        value = aliases[key]
        if value not in methods:
            methods.append(value)
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grow CPG and angular-defect-greedy bases once, then compare "
            "epsilon-driven prefixes by reconstruction error and elapsed time."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-components", type=int, default=5)
    parser.add_argument("--max-components", type=int, default=30)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=parse_methods("cpg adg"),
        help=(
            "Methods to run. Use any of: cpg, adg, angular_defect_greedy. "
            "Default: cpg adg."
        ),
    )
    parser.add_argument("--zero-tol", type=float, default=1e-14)
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
            "Fraction of sorted interior parameters held out when --split-strategy fraction."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=(
            "Optional absolute stopping tolerance. Omit it to grow to "
            "--max-components and report the stopping epsilon reached at each prefix."
        ),
    )
    return parser.parse_args()


def component_range(args: argparse.Namespace, snapshots: np.ndarray) -> list[int]:
    if args.min_components <= 0:
        raise ValueError("--min-components must be positive")
    if args.max_components < args.min_components:
        raise ValueError("--max-components must be >= --min-components")
    if args.step <= 0:
        raise ValueError("--step must be positive")

    upper = min(args.max_components, snapshots.shape[0], snapshots.shape[1])
    return list(range(args.min_components, upper + 1, args.step))


def main() -> None:
    args = parse_args()
    snapshots, radii, dataset_source = load_lambda_dataset(args.dataset)
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
    components = component_range(args, train_snapshots)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped: list[str] = []

    cpg_basis: np.ndarray | None = None
    cpg_selected: list[int] | None = None
    angle_basis: np.ndarray | None = None
    angle_selected: list[int] | None = None
    cumulative_times: dict[str, dict[str, float]] = {}

    for component_count in components:
        print(f"R = {component_count}")

        if "CPG" in args.methods:
            start = time.perf_counter()
            basis, selected = fit_cpg_fixed_components(
                train_snapshots,
                component_count,
                zero_tol=args.zero_tol,
                epsilon=args.epsilon,
                basis=cpg_basis,
                selected_indices=cpg_selected,
            )
            fit_seconds = time.perf_counter() - start
            reconstructions, coefficients, residuals, relative = evaluate_basis(
                snapshots,
                basis,
            )
            total_seconds = time.perf_counter() - start
            cumulative_entry = cumulative_times.setdefault("CPG", {"fit": 0.0, "total": 0.0})
            cumulative_entry["fit"] += fit_seconds
            cumulative_entry["total"] += total_seconds
            append_result(
                rows,
                method="CPG",
                requested_components=component_count,
                components=len(selected),
                stopping_epsilon=float(np.max(residuals)),
                fit_seconds=cumulative_entry["fit"],
                total_seconds=cumulative_entry["total"],
                residuals=residuals,
                relative_residuals=relative,
                selected_indices=selected,
                selected_original_indices=[int(train_indices[i]) for i in selected],
                selected_radii=radii[np.array([int(train_indices[i]) for i in selected], dtype=int)]
                if selected
                else None,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            cpg_basis = basis
            cpg_selected = selected

        if ADG_NAME in args.methods:
            start = time.perf_counter()
            basis, selected = fit_angle_fixed_components(
                train_snapshots,
                component_count,
                zero_tol=args.zero_tol,
                epsilon=args.epsilon,
                basis=angle_basis,
                selected_indices=angle_selected,
            )
            fit_seconds = time.perf_counter() - start
            reconstructions, coefficients, residuals, relative = evaluate_basis(
                snapshots,
                basis,
            )
            total_seconds = time.perf_counter() - start
            cumulative_entry = cumulative_times.setdefault(ADG_NAME, {"fit": 0.0, "total": 0.0})
            cumulative_entry["fit"] += fit_seconds
            cumulative_entry["total"] += total_seconds
            append_result(
                rows,
                method=ADG_NAME,
                requested_components=component_count,
                components=len(selected),
                stopping_epsilon=float(np.max(residuals)),
                fit_seconds=cumulative_entry["fit"],
                total_seconds=cumulative_entry["total"],
                residuals=residuals,
                relative_residuals=relative,
                selected_indices=selected,
                selected_original_indices=[int(train_indices[i]) for i in selected],
                selected_radii=radii[np.array([int(train_indices[i]) for i in selected], dtype=int)]
                if selected
                else None,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            angle_basis = basis
            angle_selected = selected

    csv_path = args.output_dir / "component_sweep_metrics.csv"
    report_path = args.output_dir / "component_sweep_report.txt"
    absolute_plot = args.output_dir / "component_sweep_absolute_error.png"
    relative_plot = args.output_dir / "component_sweep_relative_error.png"
    runtime_plot = args.output_dir / "component_sweep_runtime.png"
    error_gain_csv = args.output_dir / "component_sweep_error_gain.csv"
    error_gain_plot = args.output_dir / "component_sweep_error_gain.png"
    absolute_error_gain_csv = args.output_dir / "component_sweep_absolute_error_gain.csv"
    absolute_error_gain_plot = args.output_dir / "component_sweep_absolute_error_gain.png"

    write_csv(rows, csv_path)
    plotted_methods = [method for method in args.methods if any(row["method"] == method for row in rows)]
    plot_metric(
        rows,
        plotted_methods,
        absolute_plot,
        keys=("max_residual", "mean_residual"),
        labels=("max absolute residual", "mean absolute residual"),
        title="Absolute reconstruction error vs components",
        log_y=True,
    )
    plot_metric(
        rows,
        plotted_methods,
        relative_plot,
        keys=("max_test_relative_residual", "mean_test_relative_residual"),
        labels=("test max relative residual", "test mean relative residual"),
        title="Unseen-test relative reconstruction error vs components",
        log_y=True,
    )
    plot_metric(
        rows,
        plotted_methods,
        runtime_plot,
        keys=("fit_seconds", "total_seconds"),
        labels=("fit seconds", "fit + reconstruction seconds"),
        title="Runtime vs components",
        log_y=False,
    )
    bases = {
        "CPG": cpg_basis,
        ADG_NAME: angle_basis,
    }
    error_histories = error_histories_from_final_bases(
        snapshots,
        bases,
        relative=True,
    )
    absolute_error_histories = error_histories_from_final_bases(
        snapshots,
        bases,
        relative=False,
    )
    write_error_gain_csv(error_histories, error_gain_csv)
    plot_error_gain(
        error_histories,
        error_gain_plot,
        title="Lambda dataset relative error gain between greedy iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
    )
    write_error_gain_csv(absolute_error_histories, absolute_error_gain_csv)
    plot_error_gain(
        absolute_error_histories,
        absolute_error_gain_plot,
        title="Lambda dataset absolute error gain between greedy iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
    )
    write_report(
        rows,
        report_path,
        dataset_source=dataset_source,
        snapshot_shape=snapshots.shape,
        components=components,
        methods=plotted_methods,
        skipped=skipped,
        train_indices=train_indices,
        test_indices=test_indices,
        parameters=radii,
        split_strategy=split_strategy,
    )

    print(f"Loaded snapshots from: {dataset_source}")
    print(f"Train/test split: {train_indices.size} train, {test_indices.size} unseen test")
    print(f"Component sweep: {components[0]}..{components[-1]} step {args.step}")
    print(f"Methods: {', '.join(plotted_methods)}")
    if skipped:
        print("Skipped:")
        for message in skipped:
            print(f"  {message}")
    print(f"Outputs written to: {args.output_dir}")
    print(f"  metrics: {csv_path}")
    print(f"  report: {report_path}")
    print(f"  absolute_plot: {absolute_plot}")
    print(f"  relative_plot: {relative_plot}")
    print(f"  runtime_plot: {runtime_plot}")
    print(f"  error_gain_csv: {error_gain_csv}")
    print(f"  error_gain_plot: {error_gain_plot}")
    print(f"  absolute_error_gain_csv: {absolute_error_gain_csv}")
    print(f"  absolute_error_gain_plot: {absolute_error_gain_plot}")


if __name__ == "__main__":
    main()
