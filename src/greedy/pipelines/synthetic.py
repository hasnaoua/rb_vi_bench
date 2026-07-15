from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from greedy.synthetic_data.gaussian_data import gaussian_basis, create_data
from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.core.angle_defect_greedy import AngularDefectGreedy
from greedy.pipelines.lambda_cpg import (
    project_snapshots_onto_basis,
    safe_relative_residuals,
)
from greedy.core.reduction_common import (
    max_projection_error_history,
    write_error_gain_csv,
    plot_error_gain,
    plot_basis_heatmap,
    plot_basis_profiles,
    plot_reconstruction_examples,
    plot_residuals_by_radius,
    plot_summary,
    plot_coefficients,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_row(path: Path, row: str) -> None:
    header = (
        "configuration,method,relative_epsilon,R,final_max_residual,final_max_relative_residual,"
        "train_max_rel,test_max_rel,elapsed_seconds\n"
    )
    if not path.exists():
        path.write_text(header + row)
    else:
        path.write_text(path.read_text() + row)


def run_experiments(
    num_samples: int = 1000,
    width: float = 0.08,
    configurations: list[list[float]] | None = None,
    discretization_count: int = 500,
    epsilons: np.ndarray | None = None,
    test_fraction: float = 0.1,
    rng_seed: int = 123,
):
    if configurations is None:
        configurations = [
            [0.4, 0.8],
            [0.2, 0.8],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.2, 0.5, 0.8],
        ]
    if epsilons is None:
        epsilons = np.logspace(-2, -6, 5)

    out_root = Path("results/synthetic")
    ensure_dir(out_root)

    summary_path = out_root / "experiment_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    for centers in configurations:
        config_name = "means_" + "_".join(str(int(center * 100)) for center in centers)
        config_dir = out_root / config_name
        ensure_dir(config_dir)

        x, basis = gaussian_basis(
            dim_basis=len(centers),
            width=width,
            discretization_count=discretization_count,
            centers=list(centers),
        )
        data = create_data(basis, num_fields=num_samples, noise_level=0.0, min_value=0.0)
        snapshots = np.asarray(data, dtype=float)
        radii = np.linspace(0.0, 1.0, snapshots.shape[0])

        rng = np.random.default_rng(rng_seed)
        indices = np.arange(snapshots.shape[0])
        rng.shuffle(indices)
        test_count = int(round(test_fraction * snapshots.shape[0]))
        test_indices = np.sort(indices[:test_count])
        train_indices = np.sort(indices[test_count:])
        train_snapshots = snapshots[train_indices]
        train_norms = np.linalg.norm(train_snapshots, axis=1)
        stop_scale = float(np.max(train_norms)) if train_norms.size else 1.0
        if stop_scale <= 0.0:
            stop_scale = 1.0

        for eps in epsilons:
            stop_tolerance = float(eps) * stop_scale
            cpg_out = config_dir / f"cpg_eps_{eps:.0e}"
            adg_out = config_dir / f"adg_eps_{eps:.0e}"
            ensure_dir(cpg_out)
            ensure_dir(adg_out)

            t0 = time.perf_counter()
            cpg = CPG(snapshots=train_snapshots, epsilon=float(eps))
            cpg.compute_phases()
            elapsed = time.perf_counter() - t0
            basis_mat = (
                np.asarray(cpg.basis_matrix, dtype=float)
                if cpg.basis_matrix is not None
                else np.empty((0, snapshots.shape[1]))
            )
            selected_train_indices = [int(i) for i in cpg.selected_indices] if cpg.selected_indices is not None else []
            selected_indices = [int(train_indices[i]) for i in selected_train_indices]
            approximations, coefficients, residuals = project_snapshots_onto_basis(snapshots, basis_mat)
            relative_residuals = safe_relative_residuals(snapshots, residuals)

            final_max = float(np.max(residuals)) if residuals.size else 0.0
            final_rel_max = float(np.max(relative_residuals)) if relative_residuals.size else 0.0
            train_rel = relative_residuals[train_indices] if train_indices.size else np.array([])
            test_rel = relative_residuals[test_indices] if test_indices.size else np.array([])

            np.save(cpg_out / "cpg_basis.npy", basis_mat)
            np.save(cpg_out / "cpg_coefficients.npy", coefficients)
            np.save(cpg_out / "cpg_projections.npy", approximations)
            np.savetxt(cpg_out / "cpg_selected_indices.txt", np.array(selected_indices, dtype=int), fmt="%d")
            np.savetxt(cpg_out / "cpg_convergence.txt", np.asarray(cpg.residual_history), fmt="%.18e")
            (cpg_out / "cpg_report.txt").write_text(
                f"CPG synthetic report\n"
                f"configuration: {config_name}\n"
                f"means: {', '.join(str(c) for c in centers)}\n"
                f"relative_epsilon: {eps}\n"
                f"stopping_scale_max_train_norm: {stop_scale:.18e}\n"
                f"absolute_stopping_tolerance: {stop_tolerance:.18e}\n"
                f"R: {basis_mat.shape[0]}\n"
                f"final_max_residual: {final_max:.18e}\n"
                f"final_max_relative_residual: {final_rel_max:.18e}\n"
                f"train_max_relative: {float(np.max(train_rel)) if train_rel.size else float('nan'):.18e}\n"
                f"test_max_relative: {float(np.max(test_rel)) if test_rel.size else float('nan'):.18e}\n"
                f"elapsed_seconds: {elapsed:.6f}\n"
            )

            summary_row = (
                f"{config_name},CPG,{eps:.18e},{basis_mat.shape[0]},{final_max:.18e},{final_rel_max:.18e},"
                f"{float(np.max(train_rel)) if train_rel.size else float('nan'):.18e},"
                f"{float(np.max(test_rel)) if test_rel.size else float('nan'):.18e},{elapsed:.6f}\n"
            )
            save_summary_row(summary_path, summary_row)

            err_iters_rel, err_vals_rel = max_projection_error_history(snapshots, basis_mat, relative=True)
            err_iters_abs, err_vals_abs = max_projection_error_history(snapshots, basis_mat, relative=False)
            cpg_error_history = {"CPG": (err_iters_rel, err_vals_rel)}
            cpg_abs_history = {"CPG": (err_iters_abs, err_vals_abs)}
            write_error_gain_csv(cpg_error_history, cpg_out / "cpg_error_gain.csv")
            plot_error_gain(
                cpg_error_history,
                cpg_out / "cpg_error_gain.png",
                title="CPG relative error gain between greedy iterations",
                error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
            )
            write_error_gain_csv(cpg_abs_history, cpg_out / "cpg_absolute_error_gain.csv")
            plot_error_gain(
                cpg_abs_history,
                cpg_out / "cpg_absolute_error_gain.png",
                title="CPG absolute error gain between greedy iterations",
                error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
            )

            if basis_mat.size:
                plot_basis_heatmap(basis_mat, cpg_out / "cpg_basis_heatmap.png", title="CPG basis heatmap")
                plot_basis_profiles(
                    basis_mat,
                    cpg_out / "cpg_basis_profiles.png",
                    title="CPG basis profiles",
                    ylabel="basis value",
                )
                plot_coefficients(
                    coefficients,
                    radii,
                    cpg_out / "cpg_coefficients.png",
                    title="CPG projection coefficients by radius",
                )

            plot_reconstruction_examples(
                snapshots,
                approximations,
                radii,
                residuals,
                cpg_out / "cpg_projection_examples.png",
                title="CPG projection examples",
            )
            plot_residuals_by_radius(
                radii,
                residuals,
                relative_residuals,
                cpg_out / "cpg_residuals_by_radius.png",
                title="CPG projection residuals by radius",
            )
            plot_summary(
                snapshots,
                radii,
                basis_mat,
                residuals,
                relative_residuals,
                cpg_out / "cpg_summary.png",
                method_name="CPG",
            )

            t0 = time.perf_counter()
            # Standard ADG: per-snapshot normalization + angular redundancy.
            adg = AngularDefectGreedy(
                snapshots=train_snapshots,
                epsilon=float(eps),
                normalize_snapshots=True,
                angle_redundancy_tol=0.05,
            )
            adg.compute_phases()
            elapsed = time.perf_counter() - t0
            basis_mat = (
                np.asarray(adg.basis_matrix, dtype=float)
                if adg.basis_matrix is not None
                else np.empty((0, snapshots.shape[1]))
            )
            selected_train_indices = [int(i) for i in adg.selected_indices] if adg.selected_indices is not None else []
            selected_indices = [int(train_indices[i]) for i in selected_train_indices]
            reconstructions, coefficients, residuals = project_snapshots_onto_basis(snapshots, basis_mat)
            relative_residuals = safe_relative_residuals(snapshots, residuals)

            final_max = float(np.max(residuals)) if residuals.size else 0.0
            final_rel_max = float(np.max(relative_residuals)) if relative_residuals.size else 0.0
            train_rel = relative_residuals[train_indices] if train_indices.size else np.array([])
            test_rel = relative_residuals[test_indices] if test_indices.size else np.array([])

            np.save(adg_out / "adg_basis.npy", basis_mat)
            np.save(adg_out / "adg_coefficients.npy", coefficients)
            np.save(adg_out / "adg_projections.npy", reconstructions)
            np.savetxt(adg_out / "adg_selected_indices.txt", np.array(selected_indices, dtype=int), fmt="%d")
            np.savetxt(adg_out / "adg_convergence.txt", np.asarray(adg.residual_history), fmt="%.18e")
            (adg_out / "adg_report.txt").write_text(
                f"ADG synthetic report\n"
                f"configuration: {config_name}\n"
                f"means: {', '.join(str(c) for c in centers)}\n"
                f"relative_epsilon: {eps}\n"
                f"stopping_scale_max_train_norm: {stop_scale:.18e}\n"
                f"absolute_stopping_tolerance: {stop_tolerance:.18e}\n"
                f"R: {basis_mat.shape[0]}\n"
                f"final_max_residual: {final_max:.18e}\n"
                f"final_max_relative_residual: {final_rel_max:.18e}\n"
                f"train_max_relative: {float(np.max(train_rel)) if train_rel.size else float('nan'):.18e}\n"
                f"test_max_relative: {float(np.max(test_rel)) if test_rel.size else float('nan'):.18e}\n"
                f"elapsed_seconds: {elapsed:.6f}\n"
            )

            summary_row = (
                f"{config_name},ADG,{eps:.18e},{basis_mat.shape[0]},{final_max:.18e},{final_rel_max:.18e},"
                f"{float(np.max(train_rel)) if train_rel.size else float('nan'):.18e},"
                f"{float(np.max(test_rel)) if test_rel.size else float('nan'):.18e},{elapsed:.6f}\n"
            )
            save_summary_row(summary_path, summary_row)

            err_iters_rel, err_vals_rel = max_projection_error_history(snapshots, basis_mat, relative=True)
            err_iters_abs, err_vals_abs = max_projection_error_history(snapshots, basis_mat, relative=False)
            adg_error_history = {"Angular Defect Greedy": (err_iters_rel, err_vals_rel)}
            adg_abs_history = {"Angular Defect Greedy": (err_iters_abs, err_vals_abs)}
            write_error_gain_csv(adg_error_history, adg_out / "adg_error_gain.csv")
            plot_error_gain(
                adg_error_history,
                adg_out / "adg_error_gain.png",
                title="ADG relative error gain between greedy iterations",
                error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
            )
            write_error_gain_csv(adg_abs_history, adg_out / "adg_absolute_error_gain.csv")
            plot_error_gain(
                adg_abs_history,
                adg_out / "adg_absolute_error_gain.png",
                title="ADG absolute error gain between greedy iterations",
                error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
            )

            if basis_mat.size:
                plot_basis_heatmap(basis_mat, adg_out / "adg_basis_heatmap.png", title="ADG basis heatmap")
                plot_basis_profiles(
                    basis_mat,
                    adg_out / "adg_basis_profiles.png",
                    title="ADG basis profiles",
                    ylabel="basis value",
                )
                plot_coefficients(
                    coefficients,
                    radii,
                    adg_out / "adg_coefficients.png",
                    title="ADG projection coefficients by radius",
                )

            plot_reconstruction_examples(
                snapshots,
                reconstructions,
                radii,
                residuals,
                adg_out / "adg_projection_examples.png",
                title="ADG projection examples",
            )
            plot_residuals_by_radius(
                radii,
                residuals,
                relative_residuals,
                adg_out / "adg_residuals_by_radius.png",
                title="ADG projection residuals by radius",
            )
            plot_summary(
                snapshots,
                radii,
                basis_mat,
                residuals,
                relative_residuals,
                adg_out / "adg_summary.png",
                method_name="Angular Defect Greedy",
            )

            mcpg_out = config_dir / f"mcpg_eps_{eps:.0e}"
            ensure_dir(mcpg_out)

            t0 = time.perf_counter()
            mcpg = mCPG(snapshots=train_snapshots, epsilon=float(eps))
            mcpg.compute_phases()
            elapsed = time.perf_counter() - t0
            basis_mat = (
                np.asarray(mcpg.basis_matrix, dtype=float)
                if mcpg.basis_matrix is not None
                else np.empty((0, snapshots.shape[1]))
            )
            selected_train_indices = [int(i) for i in mcpg.selected_indices] if mcpg.selected_indices is not None else []
            selected_indices = [int(train_indices[i]) for i in selected_train_indices]
            reconstructions, coefficients, residuals = project_snapshots_onto_basis(snapshots, basis_mat)
            relative_residuals = safe_relative_residuals(snapshots, residuals)

            final_max = float(np.max(residuals)) if residuals.size else 0.0
            final_rel_max = float(np.max(relative_residuals)) if relative_residuals.size else 0.0
            train_rel = relative_residuals[train_indices] if train_indices.size else np.array([])
            test_rel = relative_residuals[test_indices] if test_indices.size else np.array([])

            np.save(mcpg_out / "mcpg_basis.npy", basis_mat)
            np.save(mcpg_out / "mcpg_coefficients.npy", coefficients)
            np.save(mcpg_out / "mcpg_projections.npy", reconstructions)
            np.savetxt(mcpg_out / "mcpg_selected_indices.txt", np.array(selected_indices, dtype=int), fmt="%d")
            np.savetxt(mcpg_out / "mcpg_convergence.txt", np.asarray(mcpg.residual_history), fmt="%.18e")
            (mcpg_out / "mcpg_report.txt").write_text(
                f"mCPG synthetic report\n"
                f"configuration: {config_name}\n"
                f"means: {', '.join(str(c) for c in centers)}\n"
                f"relative_epsilon: {eps}\n"
                f"stopping_scale_max_train_norm: {stop_scale:.18e}\n"
                f"absolute_stopping_tolerance: {stop_tolerance:.18e}\n"
                f"R: {basis_mat.shape[0]}\n"
                f"final_max_residual: {final_max:.18e}\n"
                f"final_max_relative_residual: {final_rel_max:.18e}\n"
                f"train_max_relative: {float(np.max(train_rel)) if train_rel.size else float('nan'):.18e}\n"
                f"test_max_relative: {float(np.max(test_rel)) if test_rel.size else float('nan'):.18e}\n"
                f"elapsed_seconds: {elapsed:.6f}\n"
                f"mean_shift_norm: {float(np.mean(mcpg.shift_norm_history)) if mcpg.shift_norm_history else float('nan'):.18e}\n"
                f"mean_orth_defect: {float(np.mean(mcpg.orth_defect_history)) if mcpg.orth_defect_history else float('nan'):.18e}\n"
            )

            summary_row = (
                f"{config_name},mCPG,{eps:.18e},{basis_mat.shape[0]},{final_max:.18e},{final_rel_max:.18e},"
                f"{float(np.max(train_rel)) if train_rel.size else float('nan'):.18e},"
                f"{float(np.max(test_rel)) if test_rel.size else float('nan'):.18e},{elapsed:.6f}\n"
            )
            save_summary_row(summary_path, summary_row)

            err_iters_rel, err_vals_rel = max_projection_error_history(snapshots, basis_mat, relative=True)
            err_iters_abs, err_vals_abs = max_projection_error_history(snapshots, basis_mat, relative=False)
            mcpg_error_history = {"mCPG": (err_iters_rel, err_vals_rel)}
            mcpg_abs_history = {"mCPG": (err_iters_abs, err_vals_abs)}
            write_error_gain_csv(mcpg_error_history, mcpg_out / "mcpg_error_gain.csv")
            plot_error_gain(
                mcpg_error_history,
                mcpg_out / "mcpg_error_gain.png",
                title="mCPG relative error gain between greedy iterations",
                error_ylabel=r"$e_n^{rel}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2/\|x\|_2$",
            )
            write_error_gain_csv(mcpg_abs_history, mcpg_out / "mcpg_absolute_error_gain.csv")
            plot_error_gain(
                mcpg_abs_history,
                mcpg_out / "mcpg_absolute_error_gain.png",
                title="mCPG absolute error gain between greedy iterations",
                error_ylabel=r"$e_n^{abs}=\max_x \|x-\mathrm{proj}_{K_n}(x)\|_2$",
            )

            if basis_mat.size:
                plot_basis_heatmap(basis_mat, mcpg_out / "mcpg_basis_heatmap.png", title="mCPG basis heatmap")
                plot_basis_profiles(
                    basis_mat,
                    mcpg_out / "mcpg_basis_profiles.png",
                    title="mCPG basis profiles",
                    ylabel="basis value",
                )
                plot_coefficients(
                    coefficients,
                    radii,
                    mcpg_out / "mcpg_coefficients.png",
                    title="mCPG projection coefficients by radius",
                )

            plot_reconstruction_examples(
                snapshots,
                reconstructions,
                radii,
                residuals,
                mcpg_out / "mcpg_projection_examples.png",
                title="mCPG projection examples",
            )
            plot_residuals_by_radius(
                radii,
                residuals,
                relative_residuals,
                mcpg_out / "mcpg_residuals_by_radius.png",
                title="mCPG projection residuals by radius",
            )
            plot_summary(
                snapshots,
                radii,
                basis_mat,
                residuals,
                relative_residuals,
                mcpg_out / "mcpg_summary.png",
                method_name="mCPG",
            )

    print(f"Synthetic experiments complete. Results in: {out_root}")


if __name__ == "__main__":
    run_experiments()
