from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from greedy.core.reduction_common import (
    DEFAULT_DATASET,
    fem_sols_train_test_split,
    load_lambda_dataset,
    method_color,
    nice_indices,
    parameter_train_test_split,
    residual_metrics,
)


DEFAULT_OUTPUT_DIR = Path("results/lambda/comparison")


def load_reconstruction(path: Path, snapshots: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    reconstructions = np.load(path, allow_pickle=False)
    if reconstructions.shape != snapshots.shape:
        raise ValueError(
            f"{path} has shape {reconstructions.shape}, expected {snapshots.shape}"
        )
    return np.asarray(reconstructions, dtype=float)


def parse_report(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _metadata_value(metadata: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return ""


def _metadata_float(metadata: dict[str, str], *keys: str) -> float:
    value = _metadata_value(metadata, *keys)
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def method_specs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    return [
        ("CPG", args.cpg_reconstructions, args.cpg_report),
        ("Angular Defect Greedy", args.angle_reconstructions, args.angle_report),
        ("mCPG", args.mcpg_reconstructions, args.mcpg_report),
    ]


def gather_results(
    snapshots: np.ndarray,
    specs: list[tuple[str, Path, Path]],
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, path, report_path in specs:
        reconstruction = load_reconstruction(path, snapshots)
        if reconstruction is None:
            continue
        residuals, relative = residual_metrics(snapshots, reconstruction)
        results[name] = {
            "reconstruction": reconstruction,
            "residuals": residuals,
            "relative": relative,
            "metadata": parse_report(report_path),
        }
    return results


def write_metrics(
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "component_or_selected_count",
                "elapsed_seconds",
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
            ]
        )
        for method, values in results.items():
            residuals = values["residuals"]
            relative = values["relative"]
            metadata = values.get("metadata", {})
            train_relative = relative[train_indices] if train_indices.size else np.array([])
            test_relative = relative[test_indices] if test_indices.size else np.array([])
            writer.writerow(
                [
                    method,
                    _metadata_value(metadata, "selected_count_R", "component_count_R"),
                    _metadata_value(metadata, "elapsed_seconds"),
                    f"{float(np.max(residuals)):.18e}",
                    f"{float(np.mean(residuals)):.18e}",
                    f"{float(np.median(residuals)):.18e}",
                    f"{float(np.max(relative)):.18e}",
                    f"{float(np.mean(relative)):.18e}",
                    f"{float(np.median(relative)):.18e}",
                    f"{float(np.max(train_relative)) if train_relative.size else float('nan'):.18e}",
                    f"{float(np.mean(train_relative)) if train_relative.size else float('nan'):.18e}",
                    f"{float(np.median(train_relative)) if train_relative.size else float('nan'):.18e}",
                    f"{float(np.max(test_relative)) if test_relative.size else float('nan'):.18e}",
                    f"{float(np.mean(test_relative)) if test_relative.size else float('nan'):.18e}",
                    f"{float(np.median(test_relative)) if test_relative.size else float('nan'):.18e}",
                    _metadata_value(metadata, "selected_indices"),
                ]
            )


def plot_residual_comparison(
    radii: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
    for method, values in results.items():
        color = method_color(method)
        axes[0].plot(
            radii,
            values["residuals"],
            marker="o",
            linewidth=1.45,
            markersize=3.4,
            color=color,
            label=method,
        )
        axes[1].plot(
            radii,
            values["relative"],
            marker="o",
            linewidth=1.45,
            markersize=3.4,
            color=color,
            label=method,
        )

    axes[0].set_title("Residual comparison by radius")
    axes[0].set_ylabel("absolute residual")
    axes[0].grid(True, alpha=0.28)
    axes[0].legend()

    axes[1].set_xlabel("radius")
    axes[1].set_ylabel("relative residual")
    axes[1].grid(True, alpha=0.28)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bar_summary(
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    test_indices: np.ndarray,
) -> None:
    methods = list(results)
    def _test_values(name: str) -> np.ndarray:
        relative = results[name]["relative"]
        return relative[test_indices] if test_indices.size else relative

    means = [float(np.mean(_test_values(name))) for name in methods]
    maxes = [float(np.max(_test_values(name))) for name in methods]
    colors = [method_color(name) for name in methods]

    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, means, width, label="mean relative residual", color=colors, alpha=0.55)
    ax.bar(x + width / 2, maxes, width, label="max relative residual", color=colors, hatch="///")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("relative residual")
    ax.set_title("Unseen-test method comparison summary")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, max(means + maxes, default=1.0) * 1.22)
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime_size_summary(
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    methods = list(results)
    elapsed = [
        _metadata_float(results[name].get("metadata", {}), "elapsed_seconds")
        for name in methods
    ]
    counts = [
        _metadata_float(
            results[name].get("metadata", {}),
            "selected_count_R",
            "component_count_R",
        )
        for name in methods
    ]

    colors = [method_color(name) for name in methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    axes[0].bar(x, elapsed, color=colors)
    axes[0].set_ylabel("elapsed seconds")
    axes[0].set_title("Runtime and reduced size")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, counts, color=colors)
    axes[1].set_ylabel("selected/components")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_comparison(
    snapshots: np.ndarray,
    radii: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
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
        figsize=(4.4 * method_count, 2.75 * rows.size),
        squeeze=False,
        sharex=True,
    )
    x = np.arange(snapshots.shape[1])

    for row_index, row in enumerate(rows):
        for col_index, (method, values) in enumerate(results.items()):
            ax = axes[row_index, col_index]
            reconstruction = values["reconstruction"][row]
            residual = values["residuals"][row]
            ax.plot(x, snapshots[row], color="#17202a", linewidth=1.3, label="input")
            ax.plot(
                x,
                reconstruction,
                color=method_color(method),
                linewidth=1.25,
                linestyle="--",
                label=method,
            )
            ax.fill_between(
                x,
                snapshots[row],
                reconstruction,
                color="#c2410c",
                alpha=0.13,
                linewidth=0,
            )
            ax.set_title(f"{method}: row {row}, r={radii[row]:.5g}, err={residual:.2e}")
            ax.grid(True, alpha=0.24)
            if row_index == rows.size - 1:
                ax.set_xlabel("stored lambda/contact index")
            if col_index == 0:
                ax.set_ylabel("lambda")
            if row_index == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Input vs reconstructions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPG, angular defect greedy (ADG), and mCPG reconstructions "
            "on lambda snapshots."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Merged lambda .npz file containing snapshots and radii.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for comparison tables and plots.",
    )
    parser.add_argument(
        "--cpg-reconstructions",
        type=Path,
        default=Path("results/lambda/cpg/cpg_projections.npy"),
        help="CPG reconstruction array.",
    )
    parser.add_argument(
        "--angle-reconstructions",
        type=Path,
        default=Path("results/lambda/adg/adg_projections.npy"),
        help="ADG reconstruction array.",
    )
    parser.add_argument(
        "--cpg-report",
        type=Path,
        default=Path("results/lambda/cpg/cpg_report.txt"),
        help="CPG report file with elapsed time and selected snapshots.",
    )
    parser.add_argument(
        "--angle-report",
        type=Path,
        default=Path("results/lambda/adg/adg_report.txt"),
        help="ADG report file with elapsed time and selected snapshots.",
    )
    parser.add_argument(
        "--mcpg-reconstructions",
        type=Path,
        default=Path("results/lambda/mcpg/mcpg_projections.npy"),
        help="mCPG reconstruction array.",
    )
    parser.add_argument(
        "--mcpg-report",
        type=Path,
        default=Path("results/lambda/mcpg/mcpg_report.txt"),
        help="mCPG report file with elapsed time and selected snapshots.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum snapshot rows in the reconstruction comparison plot.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["fem-sols-paper", "fraction"],
        default="fem-sols-paper",
        help=(
            "Training/test split for metrics. fem-sols-paper uses rad - 0.65 in "
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots, radii, dataset_source = load_lambda_dataset(args.dataset)
    if args.split_strategy == "fem-sols-paper":
        train_indices, test_indices = fem_sols_train_test_split(radii)
    else:
        if args.test_fraction < 0.0 or args.test_fraction >= 1.0:
            raise ValueError("--test-fraction must be in [0, 1)")
        train_indices, test_indices = parameter_train_test_split(
            radii,
            test_fraction=args.test_fraction,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = gather_results(snapshots, method_specs(args))
    if not results:
        raise FileNotFoundError(
            "No reconstruction files were found. Run "
            "python -m greedy.pipelines.lambda_cpg, greedy.pipelines.lambda_adg, "
            "and greedy.pipelines.lambda_mcpg first."
        )

    metrics_path = args.output_dir / "comparison_metrics.csv"
    residual_plot = args.output_dir / "comparison_residuals_by_radius.png"
    bar_plot = args.output_dir / "comparison_relative_residual_summary.png"
    runtime_size_plot = args.output_dir / "comparison_runtime_and_size.png"
    reconstruction_plot = args.output_dir / "comparison_reconstruction_examples.png"

    write_metrics(
        results,
        metrics_path,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    plot_residual_comparison(radii, results, residual_plot)
    plot_bar_summary(results, bar_plot, test_indices=test_indices)
    plot_runtime_size_summary(results, runtime_size_plot)
    plot_reconstruction_comparison(
        snapshots,
        radii,
        results,
        reconstruction_plot,
        max_examples=args.max_examples,
    )

    print(f"Loaded snapshots from: {dataset_source}")
    print(f"Train/test split: {train_indices.size} train, {test_indices.size} unseen test")
    print(f"Compared methods: {', '.join(results)}")
    print(f"Outputs written to: {args.output_dir}")
    print(f"  metrics: {metrics_path}")
    print(f"  residual_plot: {residual_plot}")
    print(f"  bar_plot: {bar_plot}")
    print(f"  runtime_size_plot: {runtime_size_plot}")
    print(f"  reconstruction_plot: {reconstruction_plot}")

    for method, values in results.items():
        residuals = values["residuals"]
        relative = values["relative"]
        test_relative = relative[test_indices] if test_indices.size else relative
        print(
            f"{method}: max residual={float(np.max(residuals)):.6e}, "
            f"test max relative={float(np.max(test_relative)):.6e}, "
            f"test mean relative={float(np.mean(test_relative)):.6e}"
        )


if __name__ == "__main__":
    main()
