from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = Path("results/lambda/dataset/lambda_dataset.npz")
DEFAULT_OUTPUT_DIR = Path("results/lambda/plots")


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    snapshots = np.asarray(data["snapshots"], dtype=float)
    radii = np.asarray(data["radii"], dtype=float)

    if snapshots.ndim != 2:
        raise ValueError(f"snapshots must be 2-D, got shape {snapshots.shape}")
    if radii.ndim != 1 or radii.size != snapshots.shape[0]:
        raise ValueError(
            f"radii must be 1-D with {snapshots.shape[0]} values, got {radii.shape}"
        )
    return snapshots, radii


def choose_row(radii: np.ndarray, row: int | None, radius: float | None) -> int:
    if row is not None and radius is not None:
        raise ValueError("Use either --row or --radius, not both.")
    if radius is not None:
        return int(np.argmin(np.abs(radii - radius)))
    if row is None:
        row = 0
    if row < 0 or row >= radii.size:
        raise ValueError(f"--row must be between 0 and {radii.size - 1}, got {row}")
    return int(row)


def plot_snapshot(
    values: np.ndarray,
    radius: float,
    row: int,
    output_path: Path,
) -> None:
    node_index = np.arange(values.size)
    norm = float(np.linalg.norm(values))
    max_index = int(np.argmax(values))
    max_value = float(values[max_index])

    fig, ax = plt.subplots(figsize=(11, 5.6))
    markerline, stemlines, baseline = ax.stem(
        node_index,
        values,
        linefmt="#7aa6c2",
        markerfmt="o",
        basefmt="#8a95a1",
    )
    plt.setp(markerline, markersize=5, markerfacecolor="#0f766e", markeredgewidth=0)
    plt.setp(stemlines, linewidth=1.0, alpha=0.85)
    plt.setp(baseline, linewidth=1.0, alpha=0.7)

    ax.plot(node_index, values, color="#17202a", linewidth=1.4, alpha=0.85)
    ax.scatter(
        [max_index],
        [max_value],
        s=70,
        color="#c2410c",
        zorder=4,
        label=f"max at node {max_index}: {max_value:.4g}",
    )

    ax.set_title(f"One lambda snapshot  (row {row}, parameter={radius:.5g})")
    ax.set_xlabel("stored lambda/contact index")
    ax.set_ylabel("lambda value at node")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right")

    text = (
        f"nodes: {values.size}\n"
        f"L2 norm: {norm:.5g}\n"
        f"mean: {float(np.mean(values)):.5g}\n"
        f"min: {float(np.min(values)):.5g}"
    )
    ax.text(
        0.985,
        0.72,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d9e0e7"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one lambda snapshot as a 1D contact-boundary nodal profile."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Merged .npz file containing snapshots and radii.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Snapshot row to plot. Defaults to row 0.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Plot the snapshot whose stored radius is nearest to this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to results/lambda/plots/single_snapshot_*.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots, radii = load_dataset(args.dataset)
    row = choose_row(radii, args.row, args.radius)
    radius = float(radii[row])

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"single_snapshot_row_{row:03d}_radius_{radius:.5g}.png"

    plot_snapshot(snapshots[row], radius, row, output)
    print(f"Plotted row {row}, radius {radius:.12g}")
    print(f"Snapshot shape: {snapshots[row].shape}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
