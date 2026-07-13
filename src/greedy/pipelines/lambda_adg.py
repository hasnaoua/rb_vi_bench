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

from greedy.core.angle_defect_greedy import AngularDefectGreedy
from greedy.pipelines.lambda_cpg import (
    DEFAULT_DATASET,
    load_or_build_dataset,
    project_snapshots_onto_basis,
    safe_relative_residuals,
)
from greedy.datasets.lambda_snapshots import default_input_path
from greedy.core.reduction_common import (
    fem_sols_train_test_split,
    max_projection_error_history,
    parameter_train_test_split,
    plot_basis_heatmap,
    plot_basis_profiles,
    plot_coefficients,
    plot_convergence_with_angle,
    plot_error_gain,
    plot_reconstruction_examples,
    plot_residuals_by_radius,
    plot_summary,
    write_error_gain_csv,
)


DEFAULT_OUTPUT_DIR = Path("results/lambda/adg")


def plot_selected_profiles(
    snapshots: np.ndarray,
    radii: np.ndarray,
    selected_indices: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(snapshots.shape[1])
    fig, ax = plt.subplots(figsize=(12, 6))

    palette = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]
    styles = ["-", "--", "-.", ":"]
    for index, row in enumerate(selected_indices):
        ax.plot(
            x,
            snapshots[row],
            linewidth=2.6,
            color=palette[index % len(palette)],
            linestyle=styles[index % len(styles)],
            alpha=0.98,
            label=f"row {row}, r={radii[row]:.3g}",
            zorder=3,
        )

    ax.set_title("Input snapshots with ADG selected basis profiles")
    ax.set_xlabel("stored lambda/contact index")
    ax.set_ylabel("lambda value")
    ax.grid(True, alpha=0.28)
    if selected_indices:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_convergence(
    residual_history: list[float],
    residual_threshold: float,
    output_path: Path,
    basis_sizes: list[int] | None = None,
    angle_history: list[float] | None = None,
    initial_pair_size: int = 2,
    threshold_label: str | None = None,
) -> None:
    """
    Residual convergence with an optional twin axis showing the angle (deg)
    between each accepted candidate and the cone it joined. ADG's
    ``angle_history`` has no entry for the initial 2-vector pair (they are
    chosen by mutual angle, not angle-to-cone), so its x-axis starts one
    step after ``basis_sizes`` at ``initial_pair_size + 1``.
    """
    angle_deg = np.degrees(np.asarray(angle_history, dtype=float)) if angle_history else None
    angle_basis_sizes = (
        np.arange(initial_pair_size + 1, initial_pair_size + 1 + len(angle_history))
        if angle_history
        else None
    )
    plot_convergence_with_angle(
        residual_history,
        residual_threshold,
        output_path,
        basis_sizes=basis_sizes,
        angle_history_deg=angle_deg,
        angle_basis_sizes=angle_basis_sizes,
        title="Angular defect greedy convergence",
        method_name="Angular Defect Greedy",
        residual_label="max projection residual",
        threshold_label=threshold_label,
    )


def write_candidate_history(
    path: Path,
    selected_indices: list[int],
    selected_radii: np.ndarray,
    angle_history: list[float],
    candidate_residual_history: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accepted_step", "row", "radius", "angle", "candidate_residual"])

        # The first two rows are initialization, so angle/candidate histories start at row 3.
        for accepted_step, row in enumerate(selected_indices, start=1):
            if accepted_step <= 2:
                angle = ""
                residual = ""
            else:
                history_index = accepted_step - 3
                angle = f"{angle_history[history_index]:.18e}"
                residual = f"{candidate_residual_history[history_index]:.18e}"
            writer.writerow(
                [
                    accepted_step,
                    row,
                    f"{selected_radii[accepted_step - 1]:.12g}",
                    angle,
                    residual,
                ]
            )


def save_outputs(
    output_dir: Path,
    snapshots: np.ndarray,
    radii: np.ndarray,
    basis: np.ndarray,
    selected_indices: list[int],
    coefficients: np.ndarray,
    reconstructions: np.ndarray,
    residuals: np.ndarray,
    relative_residuals: np.ndarray,
    greedy: AngularDefectGreedy,
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
        "basis": output_dir / "adg_basis.npy",
        "selected_indices": output_dir / "adg_selected_indices.txt",
        "selected_radii": output_dir / "adg_selected_radii.txt",
        "coefficients": output_dir / "adg_coefficients.npy",
        "reconstructions": output_dir / "adg_projections.npy",
        "residuals_csv": output_dir / "adg_residuals.csv",
        "convergence": output_dir / "adg_convergence.txt",
        "candidate_history": output_dir / "adg_candidate_history.csv",
        "report": output_dir / "adg_report.txt",
    }

    np.save(paths["basis"], basis)
    np.save(paths["coefficients"], coefficients)
    np.save(paths["reconstructions"], reconstructions)
    np.savetxt(paths["selected_indices"], selected, fmt="%d")
    np.savetxt(paths["selected_radii"], selected_radii, fmt="%.12g")
    np.savetxt(paths["convergence"], np.asarray(greedy.residual_history), fmt="%.18e")
    write_candidate_history(
        paths["candidate_history"],
        selected_indices,
        selected_radii,
        greedy.angle_history,
        greedy.candidate_residual_history,
    )

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

    train_relative = relative_residuals[train_indices] if train_indices.size else np.array([])
    test_relative = relative_residuals[test_indices] if test_indices.size else np.array([])
    train_norms = np.linalg.norm(snapshots[train_indices], axis=1) if train_indices.size else np.array([])
    stopping_scale = float(np.max(train_norms)) if train_norms.size else 1.0
    if stopping_scale <= 0.0:
        stopping_scale = 1.0
    stopping_tolerance = epsilon * stopping_scale
    report = [
        "Angular defect greedy lambda snapshot report",
        "",
        f"dataset_source: {dataset_source}",
        f"relative_epsilon: {epsilon:.12g}",
        f"stopping_scale_max_train_norm: {stopping_scale:.18e}",
        f"absolute_stopping_tolerance: {stopping_tolerance:.18e}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"normalize_snapshots: {greedy.normalize_snapshots}",
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
        f"initial_pair: {greedy.initial_pair}",
        "initial_pair_original_rows: "
        f"{[int(train_indices[i]) for i in greedy.initial_pair] if greedy.initial_pair is not None else None}",
        f"initial_pair_angle: {greedy.initial_pair_angle}",
        f"final_max_residual: {float(np.max(residuals)):.18e}",
        f"final_mean_residual: {float(np.mean(residuals)):.18e}",
        f"final_max_relative_residual: {float(np.max(relative_residuals)):.18e}",
        f"final_mean_relative_residual: {float(np.mean(relative_residuals)):.18e}",
        f"train_max_relative_residual: {float(np.max(train_relative)) if train_relative.size else float('nan'):.18e}",
        f"train_mean_relative_residual: {float(np.mean(train_relative)) if train_relative.size else float('nan'):.18e}",
        f"test_max_relative_residual: {float(np.max(test_relative)) if test_relative.size else float('nan'):.18e}",
        f"test_mean_relative_residual: {float(np.mean(test_relative)) if test_relative.size else float('nan'):.18e}",
        f"elapsed_seconds: {elapsed:.6f}",
        "",
        "Files:",
        "- adg_basis.npy: selected angular-defect-greedy basis snapshots",
        "- adg_selected_indices.txt: selected row numbers in the input snapshot matrix",
        "- adg_selected_radii.txt: selected radius values",
        "- adg_coefficients.npy: non-negative coefficients for projecting each input snapshot",
        "- adg_projections.npy: ADG cone projection/reconstruction of every input snapshot",
        "- adg_residuals.csv: residual per radius after the final ADG basis",
        "- adg_candidate_history.csv: accepted candidate angle/residual diagnostics",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run angular defect greedy (ADG) on lambda snapshots with the same default "
            "hyperparameters as the CPG pipeline."
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
        help="Directory for ADG outputs and plots.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        help="Projection residual stopping tolerance, same default as CPG.",
    )
    parser.add_argument(
        "--normalize-snapshots",
        action="store_true",
        help=(
            "Normalize each training snapshot to unit norm before driving the "
            "stopping/unresolved-set check, turning epsilon into a genuine "
            "per-snapshot relative-error tolerance. Selection order and the "
            "stored (physical-scale) basis are provably unchanged; only which "
            "small-magnitude snapshots still count as unresolved changes."
        ),
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
        "--max-reconstruction-examples",
        type=int,
        default=6,
        help="Maximum original-vs-projection examples to plot.",
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

    t0 = time.perf_counter()
    greedy = AngularDefectGreedy(
        snapshots=train_snapshots,
        epsilon=args.epsilon,
        normalize_snapshots=args.normalize_snapshots,
    )
    greedy.compute_phases()
    elapsed = time.perf_counter() - t0
    if greedy.basis_matrix is None:
        raise RuntimeError("ADG did not produce a basis matrix")

    basis = np.asarray(greedy.basis_matrix, dtype=float)
    if basis.size == 0:
        basis = np.empty((0, snapshots.shape[1]), dtype=float)
    selected_train_indices = [int(index) for index in greedy.selected_indices]
    selected_indices = [int(train_indices[index]) for index in selected_train_indices]
    reconstructions, coefficients, residuals = project_snapshots_onto_basis(snapshots, basis)
    relative_residuals = safe_relative_residuals(snapshots, residuals)

    data_paths = save_outputs(
        args.output_dir,
        snapshots,
        radii,
        basis,
        selected_indices,
        coefficients,
        reconstructions,
        residuals,
        relative_residuals,
        greedy,
        args.epsilon,
        elapsed,
        dataset_source,
        train_indices,
        test_indices,
        split_strategy,
    )

    selected = np.array(selected_indices, dtype=int)
    selected_radii = radii[selected] if selected.size else np.array([], dtype=float)
    plot_paths = {
        "selected_profiles": args.output_dir / "adg_selected_profiles.png",
        "basis_profiles": args.output_dir / "adg_basis_profiles.png",
        "basis_heatmap": args.output_dir / "adg_basis_heatmap.png",
        "coefficients_png": args.output_dir / "adg_coefficients.png",
        "convergence_png": args.output_dir / "adg_convergence.png",
        "error_gain_csv": args.output_dir / "adg_error_gain.csv",
        "error_gain_png": args.output_dir / "adg_error_gain.png",
        "absolute_error_gain_csv": args.output_dir / "adg_absolute_error_gain.csv",
        "absolute_error_gain_png": args.output_dir / "adg_absolute_error_gain.png",
        "residuals_png": args.output_dir / "adg_residuals_by_radius.png",
        "projection_examples": args.output_dir / "adg_projection_examples.png",
        "summary": args.output_dir / "adg_summary.png",
    }
    plot_selected_profiles(snapshots, radii, selected_indices, plot_paths["selected_profiles"])
    plot_basis_profiles(
        basis,
        plot_paths["basis_profiles"],
        title="Angular defect greedy basis vectors",
        ylabel="basis value",
    )
    plot_basis_heatmap(
        basis,
        plot_paths["basis_heatmap"],
        title="Angular defect greedy basis heatmap",
    )
    plot_coefficients(
        coefficients,
        radii,
        plot_paths["coefficients_png"],
        title="Angular defect greedy projection coefficients by radius",
    )
    plot_convergence(
        greedy.residual_history,
        greedy.stopping_tolerance,
        plot_paths["convergence_png"],
        greedy.residual_basis_sizes,
        greedy.angle_history,
        initial_pair_size=len(greedy.initial_pair) if greedy.initial_pair else 2,
        threshold_label=f"relative epsilon = {args.epsilon:g}",
    )
    error_iterations, error_values = max_projection_error_history(
        snapshots,
        basis,
        relative=True,
    )
    adg_error_history = {"Angular Defect Greedy": (error_iterations, error_values)}
    absolute_error_iterations, absolute_error_values = max_projection_error_history(
        snapshots,
        basis,
        relative=False,
    )
    adg_absolute_error_history = {
        "Angular Defect Greedy": (absolute_error_iterations, absolute_error_values)
    }
    write_error_gain_csv(adg_error_history, plot_paths["error_gain_csv"])
    plot_error_gain(
        adg_error_history,
        plot_paths["error_gain_png"],
        title="Angular defect greedy relative error gain between iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
    )
    write_error_gain_csv(
        adg_absolute_error_history,
        plot_paths["absolute_error_gain_csv"],
    )
    plot_error_gain(
        adg_absolute_error_history,
        plot_paths["absolute_error_gain_png"],
        title="Angular defect greedy absolute error gain between iterations",
        xlabel="cone size n",
        error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
    )
    plot_residuals_by_radius(
        radii,
        residuals,
        relative_residuals,
        plot_paths["residuals_png"],
        title="Angular defect greedy residuals by radius",
    )
    plot_reconstruction_examples(
        snapshots,
        reconstructions,
        radii,
        residuals,
        plot_paths["projection_examples"],
        title="Input vs ADG projection",
        max_examples=args.max_reconstruction_examples,
    )
    plot_summary(
        snapshots,
        radii,
        basis,
        residuals,
        relative_residuals,
        plot_paths["summary"],
        method_name="Angular Defect Greedy",
    )

    print(f"Loaded snapshots from: {dataset_source}")
    print(f"Input snapshot matrix: {snapshots.shape}")
    print(f"Train/test split: {train_indices.size} train, {test_indices.size} unseen test")
    print(f"ADG relative epsilon: {args.epsilon:g}")
    print(f"Initial pair: {greedy.initial_pair}")
    print(f"Initial pair angle: {greedy.initial_pair_angle}")
    print(f"Selected basis size R: {basis.shape[0]}")
    print(f"Selected rows: {selected_indices}")
    print(f"Selected radii: {[float(v) for v in selected_radii]}")
    print(f"Final max residual: {float(np.max(residuals)):.6e}")
    print(f"Final max relative residual: {float(np.max(relative_residuals)):.6e}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Outputs written to: {args.output_dir}")
    for label, path in {**data_paths, **plot_paths}.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
