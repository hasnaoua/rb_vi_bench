from __future__ import annotations

import argparse
import csv
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np

# Same archive and same rad= layout as lambda_snapshots; see fem_sols_io.
from greedy.datasets.fem_sols_io import (
    SnapshotRecord,
    default_input_path,  # noqa: F401  (re-exported alongside lambda_snapshots)
    flatten_snapshot as _flatten_snapshot,
    load_npy_bytes as _load_npy_bytes,
    parse_radius_from_parts as _parse_radius_from_parts,
)

DEFAULT_OUTPUT_DIR = Path("results/lambda/primal_dataset")
DEFAULT_PREFIX = "res"


def _choose_snapshot_file(files: dict[str, str | Path]) -> tuple[str, str | Path]:
    if "npy" in files:
        return "npy", files["npy"]
    if "txt" in files:
        return "txt", files["txt"]
    raise ValueError("Missing res.npy/res.txt entry")


def load_records_from_zip(path: Path, field_name: str) -> list[SnapshotRecord]:
    groups: dict[float, dict[str, str | Path | float]] = {}

    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not (
                name.endswith(f"/{field_name}.npy")
                or name.endswith(f"/{field_name}.txt")
            ):
                continue

            radius_info = _parse_radius_from_parts(PurePosixPath(name).parts)
            if radius_info is None:
                continue

            radius, label = radius_info
            suffix = "npy" if name.endswith(".npy") else "txt"
            group = groups.setdefault(radius, {"label": label})
            group[suffix] = name

        records: list[SnapshotRecord] = []
        for radius in sorted(groups):
            group = groups[radius]
            suffix, member = _choose_snapshot_file(group)  # type: ignore[arg-type]
            assert isinstance(member, str)
            raw = archive.read(member)
            if suffix == "npy":
                array = _load_npy_bytes(raw)
            else:
                array = np.loadtxt(BytesIO(raw), dtype=float)

            values, original_shape = _flatten_snapshot(array, member)
            records.append(
                SnapshotRecord(
                    radius=radius,
                    radius_label=str(group["label"]),
                    source=f"{path}!{member}",
                    original_shape=original_shape,
                    values=values,
                )
            )

    return records


def load_records_from_directory(path: Path, field_name: str) -> list[SnapshotRecord]:
    groups: dict[float, dict[str, str | Path | float]] = {}

    for file_path in path.rglob(f"{field_name}.*"):
        if file_path.suffix not in {".npy", ".txt"}:
            continue

        radius_info = _parse_radius_from_parts(file_path.parts)
        if radius_info is None:
            continue

        radius, label = radius_info
        suffix = file_path.suffix.lstrip(".")
        group = groups.setdefault(radius, {"label": label})
        group[suffix] = file_path

    records: list[SnapshotRecord] = []
    for radius in sorted(groups):
        group = groups[radius]
        suffix, file_path = _choose_snapshot_file(group)  # type: ignore[arg-type]
        assert isinstance(file_path, Path)
        if suffix == "npy":
            array = np.load(file_path, allow_pickle=False)
        else:
            array = np.loadtxt(file_path, dtype=float)

        values, original_shape = _flatten_snapshot(array, str(file_path))
        records.append(
            SnapshotRecord(
                radius=radius,
                radius_label=str(group["label"]),
                source=str(file_path),
                original_shape=original_shape,
                values=values,
            )
        )

    return records


def load_records(input_path: Path, field_name: str = DEFAULT_PREFIX) -> list[SnapshotRecord]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        records = load_records_from_zip(input_path, field_name)
    elif input_path.is_dir():
        records = load_records_from_directory(input_path, field_name)
    else:
        raise FileNotFoundError(
            f"Expected a .zip archive or extracted FEM_SOLS directory, got {input_path}"
        )

    if not records:
        raise ValueError(f"No rad=*/{field_name}.npy or {field_name}.txt files found")

    snapshot_size = records[0].values.size
    mismatches = [
        (record.source, record.values.size)
        for record in records
        if record.values.size != snapshot_size
    ]
    if mismatches:
        details = ", ".join(f"{source}: {size}" for source, size in mismatches[:5])
        raise ValueError(
            "All primal snapshots must have the same flattened size. "
            f"Expected {snapshot_size}; mismatches: {details}"
        )

    return records


def records_to_arrays(records: list[SnapshotRecord]) -> tuple[np.ndarray, np.ndarray]:
    radii = np.array([record.radius for record in records], dtype=float)
    snapshots = np.vstack([record.values for record in records])
    return snapshots, radii


def _component_header(component_count: int, prefix: str) -> str:
    columns = ["radius"] + [f"{prefix}_{i}" for i in range(component_count)]
    return ",".join(columns)


def save_merged_outputs(
    records: list[SnapshotRecord],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots, radii = records_to_arrays(records)

    paths = {
        "dataset_npz": output_dir / f"{prefix}_dataset.npz",
        "snapshots_npy": output_dir / f"{prefix}_snapshots.npy",
        "radii_npy": output_dir / f"{prefix}_radii.npy",
        "snapshots_txt": output_dir / f"{prefix}_snapshots.txt",
        "radii_txt": output_dir / f"{prefix}_radii.txt",
        "table_txt": output_dir / f"{prefix}_table.txt",
        "sources_csv": output_dir / f"{prefix}_sources.csv",
        "metadata_txt": output_dir / f"{prefix}_metadata.txt",
    }

    np.savez(
        paths["dataset_npz"],
        snapshots=snapshots,
        radii=radii,
        radius_labels=np.array([record.radius_label for record in records]),
        sources=np.array([record.source for record in records]),
    )
    np.save(paths["snapshots_npy"], snapshots)
    np.save(paths["radii_npy"], radii)
    np.savetxt(paths["snapshots_txt"], snapshots, fmt="%.18e")
    np.savetxt(paths["radii_txt"], radii, fmt="%.12g")
    np.savetxt(
        paths["table_txt"],
        np.column_stack([radii, snapshots]),
        fmt="%.18e",
        delimiter=",",
        header=_component_header(snapshots.shape[1], prefix),
        comments="",
    )

    with paths["sources_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "radius", "radius_folder", "original_shape", "source"])
        for row, record in enumerate(records):
            writer.writerow(
                [
                    row,
                    f"{record.radius:.12g}",
                    f"rad={record.radius_label}",
                    "x".join(str(v) for v in record.original_shape),
                    record.source,
                ]
            )

    metadata = [
        "Merged primal solution snapshot dataset",
        "",
        f"snapshot_count: {snapshots.shape[0]}",
        f"components_per_snapshot: {snapshots.shape[1]}",
        f"snapshot_matrix_shape: {snapshots.shape}",
        f"radii_min: {float(np.min(radii)):.12g}",
        f"radii_max: {float(np.max(radii)):.12g}",
        f"value_min: {float(np.min(snapshots)):.18e}",
        f"value_max: {float(np.max(snapshots)):.18e}",
        "",
        "Files:",
        f"- {paths['snapshots_npy'].name}: rows are flattened primal solution snapshots",
        f"- {paths['radii_txt'].name}: radius value for each snapshot row",
        f"- {paths['table_txt'].name}: first column is radius, remaining columns are primal values",
        f"- {paths['dataset_npz'].name}: snapshots, radii, radius_labels, sources",
        "",
        "Note:",
        "- These plots use stored primal indices or inferred index-space grids.",
        "- Physical mesh coordinates/connectivity are not present in the archive.",
    ]
    paths["metadata_txt"].write_text("\n".join(metadata) + "\n", encoding="utf-8")

    return paths


def _nice_indices(length: int, max_count: int) -> np.ndarray:
    if length <= max_count:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_count, dtype=int))


def _robust_range(values: np.ndarray, clip: tuple[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, clip)
    if vmin == vmax:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def infer_grid_shape(component_count: int) -> tuple[int, int]:
    best_rows, best_cols = 1, component_count
    best_gap = component_count - 1
    for candidate in range(1, int(np.sqrt(component_count)) + 1):
        if component_count % candidate == 0:
            other = component_count // candidate
            gap = abs(other - candidate)
            if gap < best_gap:
                best_rows, best_cols = candidate, other
                best_gap = gap
    return best_rows, best_cols


def plot_heatmap(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    *,
    percentile_clip: tuple[float, float],
) -> None:
    height = max(5.0, min(12.0, 2.2 + 0.16 * len(radii)))
    fig, ax = plt.subplots(figsize=(13, height))
    vmin, vmax = _robust_range(snapshots, percentile_clip)
    image = ax.imshow(
        snapshots,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    y_indices = _nice_indices(len(radii), 12)
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f"{radii[i]:.3g}" for i in y_indices])
    ax.set_xlabel("stored primal solution index")
    ax.set_ylabel("radius")
    ax.set_title(
        "Primal solution snapshots heatmap "
        f"({snapshots.shape[0]} radii x {snapshots.shape[1]} components)"
    )

    cbar = fig.colorbar(image, ax=ax, pad=0.012)
    cbar.set_label("primal solution value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_profiles(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    max_curves: int,
) -> None:
    selected = _nice_indices(len(radii), max(1, max_curves))
    x = np.arange(snapshots.shape[1])

    fig, ax = plt.subplots(figsize=(13, 6))
    for row in selected:
        ax.plot(x, snapshots[row], linewidth=1.25, label=f"r={radii[row]:.3g}")

    ax.set_xlabel("stored primal solution index")
    ax.set_ylabel("primal solution value")
    ax.set_title("Selected primal solution profiles")
    ax.grid(True, alpha=0.28)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(snapshots: np.ndarray, radii: np.ndarray, output_path: Path) -> None:
    l2_norms = np.linalg.norm(snapshots, axis=1)
    min_values = np.min(snapshots, axis=1)
    max_values = np.max(snapshots, axis=1)
    mean_values = np.mean(snapshots, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9.5), sharex=True)
    axes[0].plot(radii, l2_norms, marker="o", linewidth=1.5)
    axes[0].set_ylabel("L2 norm")
    axes[1].plot(radii, min_values, marker="o", linewidth=1.5, color="#2563eb")
    axes[1].set_ylabel("min")
    axes[2].plot(radii, max_values, marker="o", linewidth=1.5, color="#0f766e")
    axes[2].set_ylabel("max")
    axes[3].plot(radii, mean_values, marker="o", linewidth=1.5, color="#c2410c")
    axes[3].set_ylabel("mean")
    axes[3].set_xlabel("radius")

    for ax in axes:
        ax.grid(True, alpha=0.28)

    fig.suptitle("Primal solution snapshot summary versus radius")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_index_grid_maps(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_path: Path,
    *,
    grid_shape: tuple[int, int],
    max_maps: int,
    percentile_clip: tuple[float, float],
) -> None:
    rows, cols = grid_shape
    if rows * cols != snapshots.shape[1]:
        raise ValueError(
            f"grid_shape={grid_shape} is incompatible with {snapshots.shape[1]} values"
        )

    selected = _nice_indices(snapshots.shape[0], max(1, max_maps))
    ncols = min(4, selected.size)
    nrows = int(np.ceil(selected.size / ncols))
    vmin, vmax = _robust_range(snapshots, percentile_clip)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.3 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    last_image = None
    for axis_index, ax in enumerate(axes.ravel()):
        if axis_index >= selected.size:
            ax.axis("off")
            continue

        row = int(selected[axis_index])
        grid = snapshots[row].reshape(grid_shape)
        last_image = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"row {row}, r={radii[row]:.5g}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Primal solution index-space maps "
        f"(inferred grid {rows} x {cols}, not physical mesh coordinates)"
    )
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.72)
        cbar.set_label("primal solution value")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_visualizations(
    snapshots: np.ndarray,
    radii: np.ndarray,
    output_dir: Path,
    prefix: str,
    *,
    max_curves: int,
    max_maps: int,
    grid_shape: tuple[int, int] | None,
    percentile_clip: tuple[float, float],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if grid_shape is None:
        grid_shape = infer_grid_shape(snapshots.shape[1])

    paths = {
        "heatmap_png": output_dir / f"{prefix}_heatmap.png",
        "profiles_png": output_dir / f"{prefix}_profiles.png",
        "summary_png": output_dir / f"{prefix}_summary_vs_radius.png",
        "index_grid_maps_png": output_dir / f"{prefix}_index_grid_maps.png",
    }
    plot_heatmap(
        snapshots,
        radii,
        paths["heatmap_png"],
        percentile_clip=percentile_clip,
    )
    plot_profiles(snapshots, radii, paths["profiles_png"], max_curves)
    plot_summary(snapshots, radii, paths["summary_png"])
    plot_index_grid_maps(
        snapshots,
        radii,
        paths["index_grid_maps_png"],
        grid_shape=grid_shape,
        max_maps=max_maps,
        percentile_clip=percentile_clip,
    )
    return paths


def parse_grid_shape(raw: str | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid shape must look like ROWSxCOLS")
    try:
        rows, cols = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid shape must use integers") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("grid dimensions must be positive")
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and visualize primal solution res snapshots from FEM_SOLS."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="FEM_SOLS zip/archive directory containing rad=*/res.npy or res.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for merged primal data and visualizations.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--field-name",
        default=DEFAULT_PREFIX,
        help="Snapshot filename stem to load, usually 'res'.",
    )
    parser.add_argument(
        "--max-curves",
        type=int,
        default=8,
        help="Maximum primal profiles to overlay.",
    )
    parser.add_argument(
        "--max-maps",
        type=int,
        default=8,
        help="Maximum index-space maps to draw.",
    )
    parser.add_argument(
        "--grid-shape",
        type=parse_grid_shape,
        default=None,
        help="Optional index-space grid shape like 56x66. Defaults to a factorization of component count.",
    )
    parser.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentiles for color clipping in heatmaps/maps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input, args.field_name)
    snapshots, radii = records_to_arrays(records)

    data_paths = save_merged_outputs(records, args.output_dir, args.prefix)
    plot_paths = make_visualizations(
        snapshots,
        radii,
        args.output_dir,
        args.prefix,
        max_curves=args.max_curves,
        max_maps=args.max_maps,
        grid_shape=args.grid_shape,
        percentile_clip=tuple(args.clip_percentiles),
    )

    grid_shape = args.grid_shape or infer_grid_shape(snapshots.shape[1])
    print(f"Loaded primal snapshots from: {args.input}")
    print(f"Snapshot matrix: {snapshots.shape}")
    print(f"Original snapshot shape examples: {sorted(set(r.original_shape for r in records))}")
    print(f"Radius range: {float(np.min(radii)):.12g} -> {float(np.max(radii)):.12g}")
    print(f"Value range: {float(np.min(snapshots)):.6e} -> {float(np.max(snapshots)):.6e}")
    print(f"Index-space grid shape: {grid_shape[0]} x {grid_shape[1]}")
    print(f"Outputs written to: {args.output_dir}")
    for label, path in {**data_paths, **plot_paths}.items():
        print(f"  {label}: {path}")
    print("Note: no physical mesh coordinates/connectivity were found; maps are index-space only.")


if __name__ == "__main__":
    main()
