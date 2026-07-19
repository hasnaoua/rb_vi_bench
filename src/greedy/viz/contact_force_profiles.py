from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = Path("results/lambda/dataset/lambda_dataset.npz")
DEFAULT_OUTPUT = Path("results/lambda/plots/contact_force_profiles.png")


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


def load_x_coordinates(path: Path, expected_size: int) -> np.ndarray:
    if path.suffix == ".npy":
        x = np.load(path, allow_pickle=False)
    else:
        x = np.loadtxt(path)
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != expected_size:
        raise ValueError(
            f"{path} has {x.size} coordinates, but snapshots have {expected_size} nodes"
        )
    return x


def nearest_rows(radii: np.ndarray, target_radii: list[float]) -> list[int]:
    rows: list[int] = []
    for target in target_radii:
        row = int(np.argmin(np.abs(radii - target)))
        if row not in rows:
            rows.append(row)
    return rows


def default_rows(snapshot_count: int) -> list[int]:
    return np.unique(np.linspace(0, snapshot_count - 1, 6, dtype=int)).tolist()


def select_rows(
    radii: np.ndarray,
    rows: list[int] | None,
    target_radii: list[float] | None,
) -> list[int]:
    if rows and target_radii:
        raise ValueError("Use either --rows or --radii, not both.")
    if target_radii:
        return nearest_rows(radii, target_radii)
    if rows:
        for row in rows:
            if row < 0 or row >= radii.size:
                raise ValueError(f"row {row} is outside [0, {radii.size - 1}]")
        return rows
    return default_rows(radii.size)


def proxy_abscissa(
    node_count: int,
    radius: float,
    mode: str,
    fixed_extent: float,
) -> np.ndarray:
    if mode == "index":
        return np.arange(node_count, dtype=float)
    if mode == "unit":
        return np.linspace(-1.0, 1.0, node_count)
    if mode == "radius":
        return np.linspace(-radius, radius, node_count)
    if mode == "fixed":
        return np.linspace(-fixed_extent, fixed_extent, node_count)
    raise ValueError(f"Unknown x mode: {mode}")


def centered_contact_view(values: np.ndarray, threshold: float) -> np.ndarray:
    """
    Roll a stored lambda vector so the nonzero contact patch is visually centered.

    This is only a display aid for datasets where the stored vector order is not the
    physical left-to-right contact-boundary order. It does not replace mesh coordinates.
    """
    active = np.flatnonzero(np.abs(values) > threshold)
    if active.size == 0:
        return values.copy()

    active_center = int(round(0.5 * (int(active[0]) + int(active[-1]))))
    target_center = (values.size - 1) // 2
    shift = target_center - active_center
    return np.roll(values, shift)


def plot_profiles(
    snapshots: np.ndarray,
    radii: np.ndarray,
    rows: list[int],
    output: Path,
    x_mode: str,
    fixed_extent: float,
    x_coordinates: np.ndarray | None,
    center_contact: bool,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for curve_index, row in enumerate(rows):
        values = snapshots[row]
        if x_coordinates is not None:
            x = x_coordinates
            plot_values = values
        else:
            x = proxy_abscissa(values.size, float(radii[row]), x_mode, fixed_extent)
            plot_values = (
                centered_contact_view(values, threshold) if center_contact else values
            )

        ax.plot(
            x,
            plot_values,
            marker=markers[curve_index % len(markers)],
            markersize=3.4,
            linewidth=1.35,
            label=f"r={radii[row]:.5g}  (row {row})",
        )

    if x_coordinates is not None:
        xlabel = "contact coordinate from supplied file"
    elif x_mode == "index":
        xlabel = "stored lambda/contact index"
    elif x_mode == "radius":
        xlabel = "proxy coordinate, linearly spaced in [-parameter, parameter]"
    elif x_mode == "unit":
        xlabel = "proxy coordinate, linearly spaced in [-1, 1]"
    else:
        xlabel = f"proxy coordinate, linearly spaced in [-{fixed_extent:g}, {fixed_extent:g}]"

    title = "Contact-force profiles from lambda snapshots"
    if center_contact and x_coordinates is None and x_mode != "index":
        title += "  (contact patch centered for display)"

    ax.axhline(0.0, color="#8a95a1", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("lambda value")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot several lambda snapshots in the style of contact-force curves "
            "against stored lambda/contact index by default."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Merged .npz file containing snapshots and radii.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=None,
        help="Snapshot rows to plot.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=None,
        help="Plot snapshots nearest to these radius values.",
    )
    parser.add_argument(
        "--x-coordinates",
        type=Path,
        default=None,
        help="Optional .npy/.txt array with the physical x-coordinate of each contact node.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["index", "unit", "radius", "fixed"],
        default="index",
        help="X-axis to use when --x-coordinates is not supplied.",
    )
    parser.add_argument(
        "--fixed-extent",
        type=float,
        default=1.0,
        help="Half-width used by --x-mode fixed.",
    )
    parser.add_argument(
        "--center-contact",
        action="store_true",
        help="Roll each vector so its nonzero contact patch is visually centered.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-10,
        help="Threshold used to identify active contact nodes for --center-contact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots, radii = load_dataset(args.dataset)
    rows = select_rows(radii, args.rows, args.radii)
    x_coordinates = (
        load_x_coordinates(args.x_coordinates, snapshots.shape[1])
        if args.x_coordinates is not None
        else None
    )

    plot_profiles(
        snapshots=snapshots,
        radii=radii,
        rows=rows,
        output=args.output,
        x_mode=args.x_mode,
        fixed_extent=args.fixed_extent,
        x_coordinates=x_coordinates,
        center_contact=args.center_contact,
        threshold=args.threshold,
    )

    print(f"Plotted rows: {rows}")
    print(f"Radii: {[float(radii[row]) for row in rows]}")
    print(f"Output: {args.output}")
    if x_coordinates is None:
        print("Note: no physical contact-node coordinates were supplied.")
        print(f"Used x-axis mode: {args.x_mode}")
        if args.center_contact:
            print("The contact patch was centered for display only.")


if __name__ == "__main__":
    main()
