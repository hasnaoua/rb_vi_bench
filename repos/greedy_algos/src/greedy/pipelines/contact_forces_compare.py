"""
Run CPG, AngularDefectGreedy (ADG) and mCPG on the membrane / Hertz
contact-force datasets, in the dual W-inner product, and compare them.

    python -m greedy.pipelines.contact_forces_compare --case hertz
    python -m greedy.pipelines.contact_forces_compare --case membrane

Everything is measured in ||.||_W, not the Euclidean norm (see
docs/contact_force_datasets.md). The three greedies in greedy.core are written
against the Euclidean norm, so this module factors W = U.T @ U and runs them on
U-transformed snapshots: ||x||_W = ||U x||_2, and a cone combination transforms
as U(A c) = (U A) c with the same c >= 0, so the Euclidean algorithms operating
on U-transformed data *are* the W-norm greedies, with identical selections. The
one place this breaks down is mCPG's elementwise "theta - Upsilon in W^+"
constraint, which U does not preserve; mCPG is therefore handed
``constraint_transform=inv(U)`` so it imposes that bound back in physical
coordinates (see greedy.core.mcpg.mCPG).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from greedy.core.angle_defect_greedy import AngularDefectGreedy
from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.core.reduction_common import (
    gram_cholesky,
    method_color,
    nice_indices,
    parameter_train_test_split,
    solve_nonnegative_least_squares,
    tensor_grid_interior_split,
    validate_train_test_split,
)
from greedy.datasets.contact_forces import (
    CASES,
    ContactForceDataset,
    load_contact_force_dataset,
    pod_max_relative_error,
)


DEFAULT_OUTPUT_ROOT = Path("results/contact_forces")
METHODS = ("CPG", "ADG", "mCPG")


def project_onto_cone(snapshots: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cone-project each row of ``snapshots`` onto span_+ of ``basis`` rows."""
    if basis.shape[0] == 0:
        return np.zeros_like(snapshots), np.linalg.norm(snapshots, axis=1)

    system = basis.T
    projections = np.empty_like(snapshots)
    for row, snapshot in enumerate(snapshots):
        coeffs = solve_nonnegative_least_squares(system, snapshot)
        projections[row] = system @ coeffs
    return projections, np.linalg.norm(snapshots - projections, axis=1)


def relative_residuals(snapshots: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(snapshots, axis=1)
    return np.divide(residuals, norms, out=np.zeros_like(residuals), where=norms > 0.0)


def score_model(
    model,
    transformed: np.ndarray,
    train_indices: np.ndarray,
    elapsed: float,
) -> dict:
    """Cone-project every snapshot onto a fitted model's basis and collect metrics."""
    basis = np.asarray(model.basis_matrix, dtype=float)
    if basis.size == 0:
        basis = np.empty((0, transformed.shape[1]), dtype=float)

    projections, residuals = project_onto_cone(transformed, basis)
    return {
        "basis": basis,
        "selected_indices": [int(train_indices[i]) for i in model.selected_indices],
        "residuals": residuals,
        "relative": relative_residuals(transformed, residuals),
        "projections": projections,
        "elapsed": elapsed,
        "kappa": float(np.linalg.cond(basis)) if basis.shape[0] > 1 else 1.0,
        "residual_history": list(model.residual_history),
        "relative_history": list(model.relative_residual_history),
        "residual_basis_sizes": list(model.residual_basis_sizes),
    }


def fit(model, transformed: np.ndarray, train_indices: np.ndarray) -> dict:
    start = time.perf_counter()
    model.compute_phases()
    return score_model(model, transformed, train_indices, time.perf_counter() - start)


def run_methods(
    transformed: np.ndarray,
    train_indices: np.ndarray,
    epsilon: float,
    inverse_transform: np.ndarray,
    adg_options: dict | None = None,
) -> dict[str, dict]:
    """Run all three greedies on the W-transformed training snapshots."""
    train = transformed[train_indices]
    adg_options = adg_options or {}

    builders = {
        "CPG": lambda: CPG(snapshots=train, epsilon=epsilon),
        "ADG": lambda: AngularDefectGreedy(snapshots=train, epsilon=epsilon, **adg_options),
        "mCPG": lambda: mCPG(
            snapshots=train,
            epsilon=epsilon,
            constraint_transform=inverse_transform,
        ),
    }
    return {name: fit(build(), transformed, train_indices) for name, build in builders.items()}


# ADG's only knob. Normalization cannot change which snapshot wins a round (the
# angle criterion is invariant to positive per-vector scaling), so these two
# arms select the same generators in the same order and differ only in where
# epsilon stops them -- i.e. in R.
ADG_VARIANTS: list[tuple[str, dict]] = [
    ("default", {}),
    ("normalized", {"normalize_snapshots": True}),
]


def run_adg_study(
    transformed: np.ndarray,
    train_indices: np.ndarray,
    epsilon: float,
) -> dict[str, dict]:
    """Fit every ADG variant so the knobs can be compared at fixed epsilon."""
    train = transformed[train_indices]
    results: dict[str, dict] = {}
    for label, options in ADG_VARIANTS:
        model = AngularDefectGreedy(snapshots=train, epsilon=epsilon, **options)
        values = fit(model, transformed, train_indices)
        values["rounds"] = len(model.batch_size_history)
        values["certificate_violations"] = (
            model.verify_angular_defect_certificate()["violations"]
        )
        results[label] = values
    return results


def write_adg_study(
    results: dict[str, dict],
    output_path: Path,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "basis_R",
                "condition_number",
                "greedy_rounds",
                "certificate_violations",
                "elapsed_seconds",
                "train_max_relative_residual",
                "test_max_relative_residual",
                "test_mean_relative_residual",
            ]
        )
        for label, values in results.items():
            relative = values["relative"]
            test = relative[test_indices] if test_indices.size else np.array([np.nan])
            writer.writerow(
                [
                    label,
                    int(values["basis"].shape[0]),
                    f"{values['kappa']:.6e}",
                    values["rounds"],
                    values["certificate_violations"],
                    f"{values['elapsed']:.6f}",
                    f"{float(np.max(relative[train_indices])):.18e}",
                    f"{float(np.max(test)):.18e}",
                    f"{float(np.mean(test)):.18e}",
                ]
            )


def to_physical(basis: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map generators from the U-transformed space back to physical dual coords."""
    if basis.shape[0] == 0:
        return basis
    return la.solve_triangular(upper, basis.T, lower=False).T


def write_metrics(
    results: dict[str, dict],
    output_path: Path,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    pod_curve: np.ndarray,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "basis_R",
                "condition_number",
                "elapsed_seconds",
                "pod_error_at_same_R",
                "max_relative_residual",
                "mean_relative_residual",
                "train_max_relative_residual",
                "train_mean_relative_residual",
                "test_max_relative_residual",
                "test_mean_relative_residual",
                "selected_indices",
            ]
        )
        for method in METHODS:
            values = results[method]
            relative = values["relative"]
            train = relative[train_indices]
            test = relative[test_indices] if test_indices.size else np.array([np.nan])
            R = int(values["basis"].shape[0])
            pod = float(pod_curve[R]) if R < pod_curve.size else float("nan")
            writer.writerow(
                [
                    method,
                    R,
                    f"{values['kappa']:.6e}",
                    f"{values['elapsed']:.6f}",
                    f"{pod:.18e}",
                    f"{float(np.max(relative)):.18e}",
                    f"{float(np.mean(relative)):.18e}",
                    f"{float(np.max(train)):.18e}",
                    f"{float(np.mean(train)):.18e}",
                    f"{float(np.max(test)):.18e}",
                    f"{float(np.mean(test)):.18e}",
                    " ".join(str(i) for i in values["selected_indices"]),
                ]
            )


def plot_convergence(
    results: dict[str, dict],
    pod_curve: np.ndarray,
    epsilon: float,
    output_path: Path,
) -> None:
    """
    Greedy convergence against the POD lower bound, both as
    max_x ||x - proj(x)||_W / max_x ||x||_W on the training set.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    floor = np.finfo(float).tiny
    max_size = 0

    for method in METHODS:
        history = np.asarray(results[method]["relative_history"], dtype=float)
        sizes = np.asarray(results[method]["residual_basis_sizes"], dtype=float)
        if history.size == 0:
            continue
        max_size = max(max_size, int(sizes.max()))
        ax.semilogy(
            sizes,
            np.maximum(history, floor),
            marker="o",
            markersize=3.4,
            color=method_color(method),
            label=method,
        )

    limit = max(8, max_size + 2)
    ranks = np.arange(min(pod_curve.size, limit + 1))
    ax.semilogy(
        ranks,
        np.maximum(pod_curve[: ranks.size], floor),
        linestyle="--",
        color="#64748b",
        label="POD lower bound (unconstrained subspace)",
    )
    ax.axhline(epsilon, color="#c2410c", linestyle=":", linewidth=1.3,
               label=f"epsilon = {epsilon:g}")

    # POD hits machine zero once R reaches the training rank; without a floor
    # that single point would stretch the log axis over 300 decades.
    visible = np.concatenate(
        [np.asarray(results[m]["relative_history"], dtype=float) for m in METHODS]
        + [pod_curve[: ranks.size]]
    )
    visible = visible[visible > 0.0]
    if visible.size:
        ax.set_ylim(max(float(visible.min()) * 0.5, 1e-8), float(visible.max()) * 2.0)
    ax.set_xlim(0, limit)
    ax.set_xlabel("cone size / rank R")
    ax.set_ylabel(r"max relative residual (W-norm)")
    ax.set_title("Greedy convergence against the POD lower bound")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residuals_by_parameter(
    dataset: ContactForceDataset,
    results: dict[str, dict],
    test_indices: np.ndarray,
    output_path: Path,
) -> None:
    order = np.argsort(dataset.parameters, kind="mergesort")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method in METHODS:
        ax.semilogy(
            dataset.parameters[order],
            np.maximum(results[method]["relative"][order], np.finfo(float).tiny),
            marker="o",
            markersize=3.2,
            color=method_color(method),
            label=method,
        )
    if test_indices.size:
        ax.scatter(
            dataset.parameters[test_indices],
            np.full(test_indices.size, ax.get_ylim()[0]),
            marker="|",
            s=60,
            color="#c2410c",
            label="held-out parameters",
        )
    ax.set_xlabel(dataset.parameter_name)
    ax.set_ylabel("relative residual (W-norm)")
    ax.set_title(f"{dataset.name}: per-parameter accuracy")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_generators(
    dataset: ContactForceDataset,
    results: dict[str, dict],
    upper: np.ndarray,
    output_path: Path,
    *,
    max_generators: int = 6,
) -> None:
    """
    Physical-space generators for each method.

    Plotted against the arc abscissa when the dual space is 1-D (every Hertz
    case, lambda along the contact arc); the membrane's lambda lives on a 2-D
    disk, which has no meaningful single x-axis, so it is plotted by node index.
    """
    on_arc = dataset.coordinates is not None and dataset.coordinates.ndim == 1
    fig, axes = plt.subplots(1, len(METHODS), figsize=(5.2 * len(METHODS), 4.2), squeeze=False)
    for column, method in enumerate(METHODS):
        ax = axes[0, column]
        basis = to_physical(results[method]["basis"], upper)
        rows = nice_indices(basis.shape[0], max_generators)
        for row in rows:
            if on_arc:
                ax.plot(dataset.coordinates, basis[row], linewidth=1.3)
            else:
                ax.plot(basis[row], linewidth=1.0)
        ax.axhline(0.0, color="#8a95a1", linewidth=0.9)
        ax.set_title(f"{method}: R={basis.shape[0]}")
        ax.set_xlabel("contact abscissa x" if on_arc else "dual node index")
        if column == 0:
            ax.set_ylabel("generator value")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{dataset.name}: cone generators in physical coordinates "
        f"(first {max_generators} shown)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prefix_curve(
    basis: np.ndarray,
    transformed: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[dict]:
    """
    Accuracy and conditioning of every basis prefix K_1 .. K_R.

    All three greedies are nested — the first n generators are exactly the cone
    they would have produced at whatever epsilon stops them at n — so a prefix
    curve is a fair accuracy-vs-budget comparison between methods that otherwise
    stop at different R.
    """
    rows: list[dict] = []
    for size in range(1, basis.shape[0] + 1):
        prefix = basis[:size]
        _, residuals = project_onto_cone(transformed, prefix)
        relative = relative_residuals(transformed, residuals)
        rows.append(
            {
                "R": size,
                "kappa": float(np.linalg.cond(prefix)) if size > 1 else 1.0,
                "train_max": float(np.max(relative[train_indices])),
                "test_max": float(np.max(relative[test_indices])),
                "test_mean": float(np.mean(relative[test_indices])),
            }
        )
    return rows


def run_matched_r_study(
    transformed: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    epsilon: float,
    inverse_transform: np.ndarray,
    adg_options: dict,
) -> dict[str, list[dict]]:
    # No separate "ADG-best" arm: normalization leaves the selection order
    # untouched, so a normalized arm would trace the same prefix curve as ADG.
    train = transformed[train_indices]
    models = {
        "CPG": CPG(snapshots=train, epsilon=epsilon),
        "mCPG": mCPG(
            snapshots=train, epsilon=epsilon, constraint_transform=inverse_transform
        ),
        "ADG": AngularDefectGreedy(snapshots=train, epsilon=epsilon, **adg_options),
    }

    curves: dict[str, list[dict]] = {}
    for name, model in models.items():
        model.compute_phases()
        basis = np.asarray(model.basis_matrix, dtype=float)
        if basis.size:
            curves[name] = prefix_curve(basis, transformed, train_indices, test_indices)
    return curves


def write_matched_r(curves: dict[str, list[dict]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "R", "condition_number", "train_max_relative",
             "test_max_relative", "test_mean_relative"]
        )
        for method, rows in curves.items():
            for row in rows:
                writer.writerow(
                    [
                        method,
                        row["R"],
                        f"{row['kappa']:.6e}",
                        f"{row['train_max']:.18e}",
                        f"{row['test_max']:.18e}",
                        f"{row['test_mean']:.18e}",
                    ]
                )


def plot_matched_r(
    curves: dict[str, list[dict]],
    pod_curve: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for method, rows in curves.items():
        ranks = [row["R"] for row in rows]
        color = method_color(method)
        axes[0].semilogy(
            ranks, [row["train_max"] for row in rows],
            marker="o", markersize=3.0, color=color, label=method,
        )
        axes[1].semilogy(
            ranks, [row["kappa"] for row in rows],
            marker="o", markersize=3.0, color=color, label=method,
        )

    ranks = np.arange(min(pod_curve.size, max(len(r) for r in curves.values()) + 1))
    axes[0].semilogy(
        ranks, np.maximum(pod_curve[: ranks.size], np.finfo(float).tiny),
        linestyle="--", color="#64748b", label="POD lower bound",
    )
    axes[0].set_ylabel("max train relative residual (W-norm)")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.28)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("cone size R (matched budget)")
    axes[1].set_ylabel(r"basis condition number $\kappa$")
    axes[1].grid(True, which="both", alpha=0.28)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/contact_forces"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-2,
        help="Relative stopping tolerance, shared by all three methods.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of interior parameters held out (endpoints stay in train).",
    )
    parser.add_argument(
        "--adg-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "ADG: make the stopping test per-snapshot relative instead of globally "
            "scaled (standard ADG; on by default). Pass --no-adg-normalize for the "
            "global-scale criterion."
        ),
    )
    parser.add_argument(
        "--adg-study",
        action="store_true",
        help="Also compare ADG with and without normalization into adg_variants.csv.",
    )
    parser.add_argument(
        "--matched-r",
        action="store_true",
        help=(
            "Compare CPG / mCPG / ADG at equal cone size R by walking "
            "each basis prefix, into matched_r.csv and matched_r.png."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_contact_force_dataset(args.case, data_dir=args.data_dir)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.case)
    output_dir.mkdir(parents=True, exist_ok=True)

    upper = gram_cholesky(dataset.gram)
    inverse_transform = la.solve_triangular(upper, np.eye(upper.shape[0]), lower=False)
    transformed = dataset.snapshots @ upper.T

    # The membrane's mu is a 5x5x5 tensor grid, so a sorted split on any single
    # component would hold out points chosen essentially at random in the other
    # two; hold out the grid interior instead. Hertz's mu is the scalar R2, where
    # an ordered interior split is exactly right.
    if dataset.mu_samples.ndim > 1 and dataset.mu_samples.shape[1] > 1:
        train_indices, test_indices = tensor_grid_interior_split(dataset.mu_samples)
        split_description = "tensor-grid interior held out (boundary shell trains)"
    else:
        train_indices, test_indices = parameter_train_test_split(
            dataset.parameters,
            test_fraction=args.test_fraction,
        )
        split_description = f"ordered interior split, test_fraction={args.test_fraction:g}"
    validate_train_test_split(
        dataset.snapshot_count,
        train_indices,
        test_indices,
        require_test=True,
    )

    # Fitted on the training rows only: a POD that has seen the held-out
    # snapshots would not bound what the greedy could have reached. Measured the
    # same way the greedies measure themselves, so the curves are comparable.
    pod_curve = pod_max_relative_error(dataset, train_indices)
    adg_options = {"normalize_snapshots": args.adg_normalize}
    results = run_methods(
        transformed,
        train_indices,
        args.epsilon,
        inverse_transform,
        adg_options=adg_options,
    )

    metrics_path = output_dir / "comparison_metrics.csv"
    write_metrics(
        results,
        metrics_path,
        train_indices=train_indices,
        test_indices=test_indices,
        pod_curve=pod_curve,
    )
    plot_convergence(results, pod_curve, args.epsilon, output_dir / "convergence_vs_pod.png")
    plot_residuals_by_parameter(
        dataset,
        results,
        test_indices,
        output_dir / "residuals_by_parameter.png",
    )
    plot_generators(dataset, results, upper, output_dir / "generators.png")

    for method in METHODS:
        np.save(output_dir / f"{method.lower()}_basis_physical.npy",
                to_physical(results[method]["basis"], upper))

    print(f"Dataset: {dataset.source}")
    print(f"  {dataset.description}")
    print(f"  snapshots {dataset.snapshots.shape} (P x n_c), parameter: {dataset.parameter_name}")
    print(f"  train {train_indices.size} / held-out test {test_indices.size} — {split_description}")
    print(f"  epsilon {args.epsilon:g}, errors measured in the W-norm")
    print()
    header = f"{'method':>6}  {'R':>3}  {'kappa':>10}  {'POD@R':>9}  {'test max':>9}  {'time':>7}"
    print(header)
    print("-" * len(header))
    for method in METHODS:
        values = results[method]
        R = int(values["basis"].shape[0])
        test = values["relative"][test_indices]
        pod = float(pod_curve[R]) if R < pod_curve.size else float("nan")
        print(
            f"{method:>6}  {R:>3}  {values['kappa']:>10.2e}  {pod:>9.2e}  "
            f"{float(np.max(test)):>9.2e}  {values['elapsed']:>6.2f}s"
        )
    if args.adg_study:
        study = run_adg_study(transformed, train_indices, args.epsilon)
        write_adg_study(
            study,
            output_dir / "adg_variants.csv",
            train_indices=train_indices,
            test_indices=test_indices,
        )
        print()
        study_header = (
            f"{'ADG variant':>26}  {'R':>3}  {'kappa':>10}  {'rounds':>6}  "
            f"{'test max':>9}  {'time':>7}"
        )
        print(study_header)
        print("-" * len(study_header))
        for label, values in study.items():
            test = values["relative"][test_indices]
            print(
                f"{label:>26}  {int(values['basis'].shape[0]):>3}  {values['kappa']:>10.2e}  "
                f"{values['rounds']:>6}  {float(np.max(test)):>9.2e}  {values['elapsed']:>6.2f}s"
            )

    if args.matched_r:
        curves = run_matched_r_study(
            transformed,
            train_indices,
            test_indices,
            args.epsilon,
            inverse_transform,
            adg_options,
        )
        write_matched_r(curves, output_dir / "matched_r.csv")
        plot_matched_r(
            curves,
            pod_curve,
            output_dir / "matched_r.png",
            title=f"{dataset.name}: accuracy and conditioning at matched cone size",
        )

        # Compare only where every method actually reaches, so no arm is judged
        # at a budget it never produced.
        budget = min(len(rows) for rows in curves.values())
        print()
        print(f"Matched-R comparison at R={budget} (the largest budget all arms reach):")
        matched_header = f"{'method':>9}  {'kappa':>10}  {'train max':>9}  {'test max':>9}"
        print(matched_header)
        print("-" * len(matched_header))
        for method, rows in curves.items():
            row = rows[budget - 1]
            print(
                f"{method:>9}  {row['kappa']:>10.2e}  {row['train_max']:>9.3e}  "
                f"{row['test_max']:>9.3e}"
            )

    print()
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
