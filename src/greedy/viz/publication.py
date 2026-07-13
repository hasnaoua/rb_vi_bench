from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from greedy.datasets.physics_dataset import (
    DEFAULT_DATASET,
    DEFAULT_GRID_SHAPE,
    HEIGHT_MM,
    SECTOR_ANGLE_RAD,
    contact_surface_axes,
    reshape_contact_surface,
)


DEFAULT_RESULTS_DIR = Path("results/physics/reduction")
DEFAULT_OUTPUT_DIR = Path("results/physics/publication")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    title: str
    filename: str
    color: str
    linestyle: str
    marker: str
    zorder: int


METHODS = {
    "adg": MethodSpec(
        key="adg",
        label="ADG",
        title="ADG reconstruction",
        filename="adg_reconstructions.npy",
        color="#d97706",
        linestyle="-",
        marker="o",
        zorder=7,
    ),
    "cpg": MethodSpec(
        key="cpg",
        label="CPG",
        title="CPG reconstruction",
        filename="cpg_reconstructions.npy",
        color="#2563eb",
        linestyle="--",
        marker="s",
        zorder=6,
    ),
}

HF_LABEL = "Actual HF"
HF_COLOR = "#111827"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#20242a",
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_physics_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    snapshots = np.asarray(data["snapshots"], dtype=float)
    displacements = np.asarray(data["displacements"], dtype=float)

    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got {snapshots.shape}")
    if displacements.ndim != 1 or displacements.size != snapshots.shape[0]:
        raise ValueError("displacements must be a 1-D vector matching snapshots")
    if snapshots.shape[1] != int(np.prod(DEFAULT_GRID_SHAPE)):
        raise ValueError(
            f"expected {int(np.prod(DEFAULT_GRID_SHAPE))} contact nodes, "
            f"got {snapshots.shape[1]}"
        )
    return snapshots, displacements


def load_reconstructions(
    results_dir: Path,
    method_keys: list[str],
    snapshot_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    reconstructions: dict[str, np.ndarray] = {}
    for key in method_keys:
        spec = METHODS[key]
        path = results_dir / spec.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python -m greedy.pipelines.physics_reduction first."
            )
        values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        if values.shape != snapshot_shape:
            raise ValueError(
                f"{path} has shape {values.shape}, expected {snapshot_shape}"
            )
        reconstructions[key] = values
    return reconstructions


def resolve_snapshot_index(index: int, snapshot_count: int) -> int:
    if index < 0:
        index = snapshot_count + index
    if index < 0 or index >= snapshot_count:
        raise ValueError(f"snapshot index must be in [0, {snapshot_count - 1}]")
    return index


def select_representative_snapshot(
    snapshots: np.ndarray,
    *,
    profile_reduction: str,
    active_threshold_ratio: float,
    target_active_fraction: float,
) -> int:
    candidates: list[tuple[float, int]] = []
    for row, snapshot in enumerate(snapshots):
        profile = profile_from_snapshot(snapshot, profile_reduction)
        max_force = float(np.max(np.abs(profile))) if profile.size else 0.0
        if max_force <= 0.0:
            continue
        active, _ = active_mask_from_profile(profile, active_threshold_ratio)
        active_fraction = float(np.mean(active))
        if active_fraction <= 0.0 or active_fraction >= 1.0:
            continue
        score = abs(active_fraction - target_active_fraction)
        candidates.append((score, row))

    if not candidates:
        norms = np.linalg.norm(snapshots, axis=1)
        return int(np.argmax(norms))
    return min(candidates)[1]


def parse_snapshot_indices(
    raw_indices: list[str],
    snapshots: np.ndarray,
    *,
    profile_reduction: str,
    active_threshold_ratio: float,
    target_active_fraction: float,
) -> list[int]:
    indices: list[int] = []
    for raw_index in raw_indices:
        if raw_index.lower() == "auto":
            indices.append(
                select_representative_snapshot(
                    snapshots,
                    profile_reduction=profile_reduction,
                    active_threshold_ratio=active_threshold_ratio,
                    target_active_fraction=target_active_fraction,
                )
            )
            continue
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError("--snapshot-index must contain integers or 'auto'") from exc
        indices.append(resolve_snapshot_index(index, snapshots.shape[0]))
    return indices


def snapshot_suffix(snapshot_indices: list[int], *, limit: int = 8) -> str:
    visible = "_".join(str(index) for index in snapshot_indices[:limit])
    if len(snapshot_indices) > limit:
        visible = f"{visible}_plus_{len(snapshot_indices) - limit}"
    return visible


def profile_from_snapshot(values: np.ndarray, reduction: str) -> np.ndarray:
    surface = reshape_contact_surface(values, DEFAULT_GRID_SHAPE)
    if reduction == "mean":
        return np.mean(surface, axis=0)
    if reduction == "max":
        return np.max(surface, axis=0)
    raise ValueError(f"unsupported profile reduction: {reduction}")


def force_scale(max_force: float) -> tuple[float, int | None]:
    if not np.isfinite(max_force) or max_force <= 0.0:
        return 1.0, None
    exponent = int(np.floor(np.log10(max_force)))
    scale = 10.0**exponent
    return scale, exponent


def force_label(base_label: str, exponent: int | None) -> str:
    if exponent is None or exponent == 0:
        return base_label
    return rf"{base_label} ($\times 10^{{{exponent}}}$)"


def active_mask_from_profile(
    profile: np.ndarray,
    threshold_ratio: float,
) -> tuple[np.ndarray, float]:
    max_force = float(np.max(np.abs(profile))) if profile.size else 0.0
    threshold = threshold_ratio * max_force
    return np.abs(profile) > threshold, threshold


def mask_spans(z: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    if z.size == 0 or mask.size == 0:
        return []

    spans: list[tuple[float, float]] = []
    start: float | None = None
    for i, is_active in enumerate(mask):
        if is_active and start is None:
            start = float(z[i])
        if start is not None and (not is_active or i == mask.size - 1):
            end_index = i - 1 if not is_active else i
            spans.append((start, float(z[end_index])))
            start = None
    return spans


def mesh_segments(
    *,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    nx: int = 18,
    nz: int = 42,
) -> list[list[tuple[float, float]]]:
    x = np.linspace(x_min, x_max, nx + 1)
    z = np.linspace(z_min, z_max, nz + 1)
    segments: list[list[tuple[float, float]]] = []

    for value in x:
        segments.append([(float(value), float(z_min)), (float(value), float(z_max))])
    for value in z:
        segments.append([(float(x_min), float(value)), (float(x_max), float(value))])

    for i in range(nx):
        for j in range(nz):
            x0, x1 = float(x[i]), float(x[i + 1])
            z0, z1 = float(z[j]), float(z[j + 1])
            if (i + j) % 2 == 0:
                segments.append([(x0, z0), (x1, z1)])
            else:
                segments.append([(x0, z1), (x1, z0)])
    return segments


def annotate_span(
    ax: plt.Axes,
    z: np.ndarray,
    mask: np.ndarray,
    *,
    active: bool,
    text: str,
    color: str,
    x_start: float,
    x_text: float,
) -> None:
    target = mask if active else ~mask
    spans = mask_spans(z, target)
    if not spans:
        return

    start, end = max(spans, key=lambda span: span[1] - span[0])
    center = 0.5 * (start + end)
    ax.annotate(
        text,
        xy=(x_start, center),
        xytext=(x_text, center),
        ha="left",
        va="center",
        color=color,
        fontsize=8.8,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": color,
            "lw": 1.0,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )


def plot_contact_schematic(ax: plt.Axes, z: np.ndarray, active: np.ndarray) -> None:
    x_min, x_max = 0.0, 1.0
    contact_x = 0.32
    segments = mesh_segments(x_min=x_min, x_max=x_max, z_min=0.0, z_max=HEIGHT_MM)
    mesh = LineCollection(segments, colors="#2f343a", linewidths=0.23, alpha=0.72)
    ax.add_collection(mesh)

    red_segments: list[list[tuple[float, float]]] = []
    green_segments: list[list[tuple[float, float]]] = []
    for i in range(z.size - 1):
        segment = [(contact_x, float(z[i])), (contact_x, float(z[i + 1]))]
        if bool(active[i]) or bool(active[i + 1]):
            green_segments.append(segment)
        else:
            red_segments.append(segment)
    ax.add_collection(LineCollection(red_segments, colors="#d7191c", linewidths=2.0))
    ax.add_collection(LineCollection(green_segments, colors="#1a8f21", linewidths=2.4))

    annotate_span(
        ax,
        z,
        active,
        active=True,
        text="Active contact",
        color="#176b1b",
        x_start=contact_x,
        x_text=contact_x + 0.07,
    )
    annotate_span(
        ax,
        z,
        active,
        active=False,
        text="Inactive contact",
        color="#d7191c",
        x_start=contact_x,
        x_text=contact_x + 0.05,
    )
    ax.annotate(
        "Contact surface",
        xy=(contact_x, 0.0),
        xytext=(contact_x + 0.14, -0.34),
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#111111",
        arrowprops={
            "arrowstyle": "->",
            "color": "#111111",
            "lw": 0.9,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.42, HEIGHT_MM + 0.05)
    ax.set_aspect(0.76)
    ax.axis("off")


def add_active_band(ax: plt.Axes, z: np.ndarray, active: np.ndarray) -> None:
    ax.fill_betweenx(
        z,
        0.0,
        1.0,
        where=active,
        transform=ax.get_yaxis_transform(),
        color="#dbeafe",
        alpha=0.34,
        zorder=0,
        linewidth=0.0,
    )


def relative_residual(
    actual: np.ndarray,
    reconstruction: np.ndarray,
    *,
    norm_floor: float,
) -> float:
    residual = float(np.linalg.norm(actual - reconstruction))
    denominator = max(float(np.linalg.norm(actual)), norm_floor)
    return residual / denominator if denominator > 0.0 else 0.0


def absolute_residual(actual: np.ndarray, reconstruction: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reconstruction))


def format_error(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.2e}"


def method_relative_errors(
    *,
    actual: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    method_keys: list[str],
    snapshot_index: int,
    norm_floor: float,
) -> dict[str, float]:
    return {
        key: relative_residual(
            actual,
            reconstructions[key][snapshot_index],
            norm_floor=norm_floor,
        )
        for key in method_keys
    }


def method_absolute_errors(
    *,
    actual: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    method_keys: list[str],
    snapshot_index: int,
) -> dict[str, float]:
    return {
        key: absolute_residual(actual, reconstructions[key][snapshot_index])
        for key in method_keys
    }


def best_method_label(relative_errors: dict[str, float]) -> str:
    if not relative_errors:
        return "n/a"
    key = min(relative_errors, key=relative_errors.get)
    return METHODS[key].label


def plot_overlay_lines(
    ax: plt.Axes,
    *,
    z: np.ndarray,
    hf_profile: np.ndarray,
    rom_profiles: dict[str, np.ndarray],
    method_keys: list[str],
    active: np.ndarray,
    scale: float,
    x_limit: float,
    x_label: str,
    show_ylabel: bool,
    show_xlabel: bool,
    relative_errors: dict[str, float] | None = None,
    absolute_errors: dict[str, float] | None = None,
    legend: bool = True,
) -> None:
    add_active_band(ax, z, active)
    markevery = max(1, z.size // 12)
    ax.plot(
        hf_profile / scale,
        z,
        color=HF_COLOR,
        linewidth=3.0,
        linestyle="-",
        alpha=0.58,
        label=HF_LABEL,
        zorder=4,
    )
    for key in method_keys:
        spec = METHODS[key]
        label = spec.label
        metric_parts: list[str] = []
        if relative_errors is not None:
            metric_parts.append(f"rel={format_error(relative_errors[key])}")
        if absolute_errors is not None:
            metric_parts.append(f"abs={format_error(absolute_errors[key])}")
        if metric_parts:
            label = f"{label} " + ", ".join(metric_parts)
        ax.plot(
            rom_profiles[key] / scale,
            z,
            color=spec.color,
            linewidth=2.35,
            linestyle=spec.linestyle,
            marker=spec.marker,
            markevery=markevery,
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=label,
            zorder=spec.zorder,
        )

    ax.axvline(0.0, color="#27272a", linewidth=0.8, zorder=2)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(0.0, HEIGHT_MM)
    if show_xlabel:
        ax.set_xlabel(x_label)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(r"$z$ [mm]")
    else:
        ax.set_yticklabels([])
    ax.grid(True, color="#d8dce0", linestyle="--", linewidth=0.6, alpha=0.82)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    if legend:
        ax.legend(loc="lower right")


def plot_profile_panel(
    ax: plt.Axes,
    z: np.ndarray,
    profile: np.ndarray,
    *,
    title: str,
    color_norm: Normalize,
    color_map: str,
    scale: float,
    x_label: str,
    x_limit: float,
    active: np.ndarray,
    show_ylabel: bool,
    line_color: str | None = None,
    line_style: str = "-",
    line_width: float = 2.35,
) -> LineCollection:
    x = profile / scale
    add_active_band(ax, z, active)

    points = np.column_stack([x, z]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_values = 0.5 * (np.abs(profile[:-1]) + np.abs(profile[1:]))
    if line_color is None:
        collection = LineCollection(
            segments,
            cmap=color_map,
            norm=color_norm,
            linewidths=line_width,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        collection.set_array(line_values / scale)
        ax.scatter(
            x,
            z,
            c=np.abs(profile) / scale,
            cmap=color_map,
            norm=color_norm,
            s=8.0,
            linewidths=0.0,
            zorder=4,
        )
    else:
        collection = LineCollection(
            segments,
            colors=line_color,
            linewidths=line_width,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        ax.plot(
            x,
            z,
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=0.98,
            zorder=4,
        )
        ax.scatter(
            x,
            z,
            color=line_color,
            s=8.0,
            linewidths=0.0,
            zorder=5,
        )
    ax.add_collection(collection)

    ax.axvline(0.0, color="#27272a", linewidth=0.85, zorder=2)
    ax.set_title(title, pad=5.0)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_xlabel(x_label)
    if show_ylabel:
        ax.set_ylabel(r"$z$ [mm]")
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, color="#d8dce0", linestyle="--", linewidth=0.6, alpha=0.82)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    return collection


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=dpi)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def plot_profile_figure(
    *,
    fom_profile: np.ndarray,
    rom_profile: np.ndarray,
    z: np.ndarray,
    active: np.ndarray,
    method: MethodSpec,
    displacement: float,
    snapshot_index: int,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    max_force = float(np.max(np.abs(np.concatenate([fom_profile, rom_profile]))))
    scale, exponent = force_scale(max_force)
    x_label = force_label(r"$F_N$", exponent)
    colorbar_label = force_label("Normal force", exponent)
    x_limit = max(1.0, 1.08 * max_force / scale)
    color_norm = Normalize(vmin=0.0, vmax=max(max_force / scale, 1.0))

    fig = plt.figure(figsize=(9.2, 4.65), constrained_layout=True)
    grid = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.03, 1.0, 1.0, 0.055],
        wspace=0.22,
    )
    ax_mesh = fig.add_subplot(grid[0, 0])
    ax_rom = fig.add_subplot(grid[0, 1])
    ax_fom = fig.add_subplot(grid[0, 2], sharey=ax_rom)
    ax_cbar = fig.add_subplot(grid[0, 3])

    plot_contact_schematic(ax_mesh, z, active)
    rom_line = plot_profile_panel(
        ax_rom,
        z,
        rom_profile,
        title=method.title,
        color_norm=color_norm,
        color_map="viridis",
        scale=scale,
        x_label=x_label,
        x_limit=x_limit,
        active=active,
        show_ylabel=True,
        line_color=method.color,
    )
    plot_profile_panel(
        ax_fom,
        z,
        fom_profile,
        title=HF_LABEL,
        color_norm=color_norm,
        color_map="viridis",
        scale=scale,
        x_label=x_label,
        x_limit=x_limit,
        active=active,
        show_ylabel=False,
        line_color="#111827",
    )
    if rom_line.get_cmap() is not None:
        cbar = fig.colorbar(rom_line, cax=ax_cbar)
        cbar.set_label(colorbar_label, rotation=90, labelpad=10)
    else:
        ax_cbar.axis("off")

    fig.suptitle(
        rf"Contact normal-force profile, $u_z={displacement:.3f}$ mm "
        rf"(snapshot {snapshot_index})",
        fontsize=11.2,
        y=1.02,
    )
    return save_figure(
        fig,
        output_dir / f"physics_publication_{method.key}_profile_snapshot_{snapshot_index}",
        dpi,
    )


def log_surface(values: np.ndarray) -> np.ndarray:
    return np.log10(1.0 + np.abs(values))


def robust_limits(values: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, percentiles)
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def plot_surface_comparison(
    *,
    fom: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    method_keys: list[str],
    snapshot_index: int,
    displacement: float,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    theta_max = float(np.degrees(SECTOR_ANGLE_RAD))
    fom_surface = reshape_contact_surface(fom, DEFAULT_GRID_SHAPE)
    fom_log = log_surface(fom_surface)
    rom_logs: dict[str, np.ndarray] = {}
    error_logs: dict[str, np.ndarray] = {}

    value_panels = [fom_log]
    error_panels: list[np.ndarray] = []
    for key in method_keys:
        rom_surface = reshape_contact_surface(
            reconstructions[key][snapshot_index],
            DEFAULT_GRID_SHAPE,
        )
        rom_log = log_surface(rom_surface)
        error_log = log_surface(fom_surface - rom_surface)
        rom_logs[key] = rom_log
        error_logs[key] = error_log
        value_panels.append(rom_log)
        error_panels.append(error_log)

    value_limits = robust_limits(np.asarray(value_panels), (0.5, 99.7))
    error_limits = robust_limits(np.asarray(error_panels), (0.5, 99.7))
    rows = len(method_keys)

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(8.8, max(2.9, 2.45 * rows)),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [0.0, HEIGHT_MM, 0.0, theta_max]
    last_value = None
    last_error = None

    for row, key in enumerate(method_keys):
        spec = METHODS[key]
        panels = [
            (HF_LABEL, fom_log, "viridis", value_limits),
            (f"{spec.label} ROM", rom_logs[key], "viridis", value_limits),
            (rf"$|$HF - {spec.label}$|$", error_logs[key], "magma", error_limits),
        ]
        for col, (title, values, cmap, limits) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            if col < 2:
                last_value = image
            else:
                last_error = image
            ax.set_title(title, pad=5.0)
            if col == 0:
                ax.set_ylabel(r"$\theta$ [deg]")
            else:
                ax.set_yticklabels([])
            if row == rows - 1:
                ax.set_xlabel(r"$z$ [mm]")
            ax.tick_params(length=3.0, width=0.75)

    if last_value is not None:
        cbar = fig.colorbar(last_value, ax=axes[:, :2], shrink=0.92, pad=0.015)
        cbar.set_label(r"$\log_{10}(1+|F_N|)$")
    if last_error is not None:
        cbar = fig.colorbar(last_error, ax=axes[:, 2], shrink=0.92, pad=0.015)
        cbar.set_label(r"$\log_{10}(1+|\Delta F_N|)$")

    fig.suptitle(
        rf"Unwrapped quarter-cylinder contact surface, $u_z={displacement:.3f}$ mm "
        rf"(snapshot {snapshot_index})",
        fontsize=11.2,
    )
    return save_figure(
        fig,
        output_dir / f"physics_publication_surface_comparison_snapshot_{snapshot_index}",
        dpi,
    )


def plot_profile_overlay(
    *,
    fom_profile: np.ndarray,
    rom_profiles: dict[str, np.ndarray],
    method_keys: list[str],
    z: np.ndarray,
    active: np.ndarray,
    displacement: float,
    snapshot_index: int,
    output_dir: Path,
    dpi: int,
    relative_errors: dict[str, float] | None = None,
    absolute_errors: dict[str, float] | None = None,
) -> list[Path]:
    all_profiles = [fom_profile] + [rom_profiles[key] for key in method_keys]
    max_force = float(np.max(np.abs(np.concatenate(all_profiles))))
    scale, exponent = force_scale(max_force)
    x_label = force_label("Normal force", exponent)
    x_limit = max(1.0, 1.08 * max_force / scale)

    fig, ax = plt.subplots(figsize=(6.0, 4.35), constrained_layout=True)
    plot_overlay_lines(
        ax,
        z=z,
        hf_profile=fom_profile,
        rom_profiles=rom_profiles,
        method_keys=method_keys,
        active=active,
        scale=scale,
        x_limit=x_limit,
        x_label=x_label,
        show_ylabel=True,
        show_xlabel=True,
        relative_errors=relative_errors,
        absolute_errors=absolute_errors,
    )
    title = rf"HF/ROM superposition, $u_z={displacement:.3f}$ mm (snapshot {snapshot_index})"
    if relative_errors:
        title = f"{title}, best {best_method_label(relative_errors)}"
    ax.set_title(title)
    return save_figure(
        fig,
        output_dir / f"physics_publication_hf_rom_superposition_snapshot_{snapshot_index}",
        dpi,
    )


def plot_multi_snapshot_overlay(
    *,
    snapshots: np.ndarray,
    reconstructions: dict[str, np.ndarray],
    method_keys: list[str],
    snapshot_indices: list[int],
    displacements: np.ndarray,
    z: np.ndarray,
    profile_reduction: str,
    active_threshold_ratio: float,
    output_dir: Path,
    dpi: int,
    relative_norm_floor: float,
) -> list[Path]:
    if not snapshot_indices:
        return []

    profiles: dict[int, tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = {}
    all_profiles: list[np.ndarray] = []
    errors_by_snapshot: dict[int, dict[str, float]] = {}
    absolute_errors_by_snapshot: dict[int, dict[str, float]] = {}
    for snapshot_index in snapshot_indices:
        hf_profile = profile_from_snapshot(snapshots[snapshot_index], profile_reduction)
        active, _ = active_mask_from_profile(hf_profile, active_threshold_ratio)
        rom_profiles = {
            key: profile_from_snapshot(
                reconstructions[key][snapshot_index],
                profile_reduction,
            )
            for key in method_keys
        }
        profiles[snapshot_index] = (hf_profile, rom_profiles, active)
        all_profiles.append(hf_profile)
        all_profiles.extend(rom_profiles.values())
        errors_by_snapshot[snapshot_index] = method_relative_errors(
            actual=snapshots[snapshot_index],
            reconstructions=reconstructions,
            method_keys=method_keys,
            snapshot_index=snapshot_index,
            norm_floor=relative_norm_floor,
        )
        absolute_errors_by_snapshot[snapshot_index] = method_absolute_errors(
            actual=snapshots[snapshot_index],
            reconstructions=reconstructions,
            method_keys=method_keys,
            snapshot_index=snapshot_index,
        )

    max_force = float(np.max(np.abs(np.concatenate(all_profiles))))
    scale, exponent = force_scale(max_force)
    x_label = force_label("Normal force", exponent)
    x_limit = max(1.0, 1.08 * max_force / scale)
    ncols = 2 if len(snapshot_indices) > 1 else 1
    nrows = int(np.ceil(len(snapshot_indices) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 3.35 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for panel_index, snapshot_index in enumerate(snapshot_indices):
        row = panel_index // ncols
        col = panel_index % ncols
        ax = axes[row, col]
        hf_profile, rom_profiles, active = profiles[snapshot_index]
        relative_errors = errors_by_snapshot[snapshot_index]
        absolute_errors = absolute_errors_by_snapshot[snapshot_index]
        plot_overlay_lines(
            ax,
            z=z,
            hf_profile=hf_profile,
            rom_profiles=rom_profiles,
            method_keys=method_keys,
            active=active,
            scale=scale,
            x_limit=x_limit,
            x_label=x_label,
            show_ylabel=col == 0,
            show_xlabel=row == nrows - 1,
            relative_errors=None,
            absolute_errors=None,
            legend=panel_index == 0,
        )
        error_lines = "\n".join(
            f"{METHODS[key].label}: rel {format_error(relative_errors[key])}, "
            f"abs {format_error(absolute_errors[key])}"
            for key in method_keys
        )
        ax.text(
            0.98,
            0.97,
            f"best {best_method_label(relative_errors)}\n{error_lines}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d4d4d8",
                "alpha": 0.86,
            },
            zorder=10,
        )
        ax.set_title(
            rf"snapshot {snapshot_index}, $u_z={float(displacements[snapshot_index]):.3f}$ mm"
        )

    for panel_index in range(len(snapshot_indices), nrows * ncols):
        axes[panel_index // ncols, panel_index % ncols].axis("off")

    fig.suptitle("HF/ADG/CPG contact-force profile superposition", fontsize=11.4)
    return save_figure(
        fig,
        output_dir
        / f"physics_publication_hf_rom_superposition_snapshots_{snapshot_suffix(snapshot_indices)}",
        dpi,
    )


def write_report(
    *,
    output_dir: Path,
    dataset_path: Path,
    results_dir: Path,
    method_keys: list[str],
    snapshot_index: int,
    displacement: float,
    profile_reduction: str,
    active_threshold: float,
    active: np.ndarray,
    z: np.ndarray,
    generated: list[Path],
    relative_errors: dict[str, float] | None = None,
    absolute_errors: dict[str, float] | None = None,
    comparison_mode: str = "full",
) -> Path:
    report_path = (
        output_dir / f"physics_publication_visualization_report_snapshot_{snapshot_index}.txt"
    )
    spans = mask_spans(z, active)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Physics HF/ROM visualization\n")
        handle.write(f"dataset: {dataset_path}\n")
        handle.write(f"results_dir: {results_dir}\n")
        handle.write(f"methods: {', '.join(METHODS[key].label for key in method_keys)}\n")
        handle.write(f"comparison_mode: {comparison_mode}\n")
        handle.write(f"snapshot_index: {snapshot_index}\n")
        handle.write(f"displacement_mm: {displacement:.12g}\n")
        handle.write(f"profile_reduction: {profile_reduction}\n")
        handle.write(f"active_threshold: {active_threshold:.12e}\n")
        if relative_errors:
            handle.write("relative_residuals:\n")
            for key in method_keys:
                handle.write(
                    f"- {METHODS[key].label}: {relative_errors[key]:.18e}\n"
                )
            handle.write(f"best_method: {best_method_label(relative_errors)}\n")
        if absolute_errors:
            handle.write("absolute_residuals:\n")
            for key in method_keys:
                handle.write(
                    f"- {METHODS[key].label}: {absolute_errors[key]:.18e}\n"
                )
        handle.write("active_z_spans_mm: ")
        if spans:
            handle.write(", ".join(f"[{start:.4g}, {end:.4g}]" for start, end in spans))
        else:
            handle.write("none")
        handle.write("\n")
        handle.write("generated_files:\n")
        for path in generated:
            handle.write(f"- {path}\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-style physics contact-force visualizations from "
            "saved HF snapshots and CPG/ADG reconstructions."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Physics .npz dataset with snapshots and displacements.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing cpg_reconstructions.npy and adg_reconstructions.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for publication figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=["adg", "cpg"],
        help="Reduced models to visualize.",
    )
    parser.add_argument(
        "--snapshot-index",
        nargs="+",
        default=["auto"],
        help=(
            "One or more snapshot rows to visualize, or 'auto' for a representative "
            "contact-front state. Negative integers count from the end."
        ),
    )
    parser.add_argument(
        "--profile-reduction",
        choices=["mean", "max"],
        default="mean",
        help="How to collapse the theta direction into a z-profile.",
    )
    parser.add_argument(
        "--active-threshold-ratio",
        type=float,
        default=1.0e-3,
        help="Active contact threshold as a fraction of the HF profile maximum.",
    )
    parser.add_argument(
        "--target-active-fraction",
        type=float,
        default=0.55,
        help="Target active z-fraction used when --snapshot-index auto.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution for PNG files.",
    )
    parser.add_argument(
        "--relative-norm-floor",
        type=float,
        default=0.1,
        help="Denominator floor used for displayed relative residuals.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["auto", "full", "overlay-only"],
        default="auto",
        help=(
            "auto uses overlay-only for multi-snapshot method comparisons; "
            "full also writes the detailed per-method and surface figures."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    snapshots, displacements = load_physics_dataset(args.dataset)
    snapshot_indices = parse_snapshot_indices(
        args.snapshot_index,
        snapshots,
        profile_reduction=args.profile_reduction,
        active_threshold_ratio=args.active_threshold_ratio,
        target_active_fraction=args.target_active_fraction,
    )
    reconstructions = load_reconstructions(
        args.results_dir,
        args.methods,
        snapshots.shape,
    )
    if args.relative_norm_floor < 0.0:
        raise ValueError("--relative-norm-floor must be non-negative")

    _, z = contact_surface_axes(DEFAULT_GRID_SHAPE)
    overlay_only = args.comparison_mode == "overlay-only" or (
        args.comparison_mode == "auto"
        and len(snapshot_indices) > 1
        and len(args.methods) > 1
    )

    multi_overlay_generated: list[Path] = []
    if overlay_only and len(args.methods) > 1 and len(snapshot_indices) > 1:
        multi_overlay_generated = plot_multi_snapshot_overlay(
            snapshots=snapshots,
            reconstructions=reconstructions,
            method_keys=args.methods,
            snapshot_indices=snapshot_indices,
            displacements=displacements,
            z=z,
            profile_reduction=args.profile_reduction,
            active_threshold_ratio=args.active_threshold_ratio,
            output_dir=args.output_dir,
            dpi=args.dpi,
            relative_norm_floor=args.relative_norm_floor,
        )

    generated_all: list[Path] = []
    for snapshot_index in snapshot_indices:
        fom = snapshots[snapshot_index]
        fom_profile = profile_from_snapshot(fom, args.profile_reduction)
        active, active_threshold = active_mask_from_profile(
            fom_profile,
            args.active_threshold_ratio,
        )
        rom_profiles = {
            key: profile_from_snapshot(
                reconstructions[key][snapshot_index],
                args.profile_reduction,
            )
            for key in args.methods
        }
        relative_errors = method_relative_errors(
            actual=fom,
            reconstructions=reconstructions,
            method_keys=args.methods,
            snapshot_index=snapshot_index,
            norm_floor=args.relative_norm_floor,
        )
        absolute_errors = method_absolute_errors(
            actual=fom,
            reconstructions=reconstructions,
            method_keys=args.methods,
            snapshot_index=snapshot_index,
        )

        generated: list[Path] = list(multi_overlay_generated)
        if overlay_only:
            if len(args.methods) > 1 and len(snapshot_indices) == 1:
                generated.extend(
                    plot_profile_overlay(
                        fom_profile=fom_profile,
                        rom_profiles=rom_profiles,
                        method_keys=args.methods,
                        z=z,
                        active=active,
                        displacement=float(displacements[snapshot_index]),
                        snapshot_index=snapshot_index,
                        output_dir=args.output_dir,
                        dpi=args.dpi,
                        relative_errors=relative_errors,
                        absolute_errors=absolute_errors,
                    )
                )
        else:
            for key in args.methods:
                generated.extend(
                    plot_profile_figure(
                        fom_profile=fom_profile,
                        rom_profile=rom_profiles[key],
                        z=z,
                        active=active,
                        method=METHODS[key],
                        displacement=float(displacements[snapshot_index]),
                        snapshot_index=snapshot_index,
                        output_dir=args.output_dir,
                        dpi=args.dpi,
                    )
                )
            if len(args.methods) > 1:
                generated.extend(
                    plot_profile_overlay(
                        fom_profile=fom_profile,
                        rom_profiles=rom_profiles,
                        method_keys=args.methods,
                        z=z,
                        active=active,
                        displacement=float(displacements[snapshot_index]),
                        snapshot_index=snapshot_index,
                        output_dir=args.output_dir,
                        dpi=args.dpi,
                        relative_errors=relative_errors,
                        absolute_errors=absolute_errors,
                    )
                )
            generated.extend(
                plot_surface_comparison(
                    fom=fom,
                    reconstructions=reconstructions,
                    method_keys=args.methods,
                    snapshot_index=snapshot_index,
                    displacement=float(displacements[snapshot_index]),
                    output_dir=args.output_dir,
                    dpi=args.dpi,
                )
            )
        report = write_report(
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            results_dir=args.results_dir,
            method_keys=args.methods,
            snapshot_index=snapshot_index,
            displacement=float(displacements[snapshot_index]),
            profile_reduction=args.profile_reduction,
            active_threshold=active_threshold,
            active=active,
            z=z,
            generated=generated,
            relative_errors=relative_errors,
            absolute_errors=absolute_errors,
            comparison_mode="overlay-only" if overlay_only else "full",
        )
        generated_all.extend(generated)
        print(f"Snapshot {snapshot_index} (u_z={float(displacements[snapshot_index]):.3f} mm)")
        for path in generated:
            print(f"- {path}")
        print(f"Report: {report}")

    print("Completed publication visualization for all requested snapshots.")


if __name__ == "__main__":
    main()
