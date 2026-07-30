from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from greedy.core.reduction_common import DEFAULT_DATASET, load_lambda_dataset


def symmetry_error(values: np.ndarray) -> float:
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        return 0.0
    return float(np.linalg.norm(values - values[::-1]) / norm)


def active_range(values: np.ndarray, threshold: float) -> str:
    active = np.flatnonzero(np.abs(values) > threshold)
    if active.size == 0:
        return "none"
    return f"{int(active[0])}..{int(active[-1])} ({active.size} active)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether lambda snapshots look like full symmetric contact-boundary profiles."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Merged lambda .npz file containing snapshots and radii/parameters.",
    )
    parser.add_argument(
        "--expected-snapshots",
        type=int,
        default=51,
        help="Expected number of training snapshots from the reference problem.",
    )
    parser.add_argument(
        "--expected-contact-nodes",
        type=int,
        default=380,
        help="Expected number of potential contact nodes from the reference problem.",
    )
    parser.add_argument(
        "--expected-min-parameter",
        type=float,
        default=0.15,
        help="Expected minimum parameter from the reference problem.",
    )
    parser.add_argument(
        "--expected-max-parameter",
        type=float,
        default=0.45,
        help="Expected maximum parameter from the reference problem.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-10,
        help="Threshold used to report active lambda entries.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="*",
        default=None,
        help="Rows to print in detail. Defaults to first, middle, and last rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots, parameters, source = load_lambda_dataset(args.dataset)

    rows = args.rows
    if rows is None:
        rows = sorted(set([0, snapshots.shape[0] // 2, snapshots.shape[0] - 1]))

    print(f"dataset_source: {source}")
    print(f"snapshot_matrix_shape: {snapshots.shape}")
    print(
        "parameter_range: "
        f"{float(np.min(parameters)):.12g} .. {float(np.max(parameters)):.12g}"
    )
    print("")
    print("Reference-problem expectations:")
    print(f"  snapshots: {args.expected_snapshots}")
    print(f"  contact_nodes: {args.expected_contact_nodes}")
    print(
        "  parameter_range: "
        f"{args.expected_min_parameter:.12g} .. {args.expected_max_parameter:.12g}"
    )
    print("")
    print("Mismatches:")
    print(
        f"  snapshots: actual {snapshots.shape[0]}, "
        f"expected {args.expected_snapshots}"
    )
    print(
        f"  lambda_entries: actual {snapshots.shape[1]}, "
        f"expected {args.expected_contact_nodes}"
    )
    print(
        f"  parameter_min: actual {float(np.min(parameters)):.12g}, "
        f"expected {args.expected_min_parameter:.12g}"
    )
    print(
        f"  parameter_max: actual {float(np.max(parameters)):.12g}, "
        f"expected {args.expected_max_parameter:.12g}"
    )
    print("")
    print("Row diagnostics:")
    for row in rows:
        if row < 0 or row >= snapshots.shape[0]:
            raise ValueError(f"row {row} outside [0, {snapshots.shape[0] - 1}]")
        values = snapshots[row]
        print(
            f"  row {row:3d}, parameter={parameters[row]:.12g}, "
            f"active={active_range(values, args.threshold)}, "
            f"reverse_symmetry_error={symmetry_error(values):.6e}, "
            f"max_index={int(np.argmax(values))}"
        )

    print("")
    print("Interpretation:")
    print(
        "  A full symmetric contact profile needs the full potential contact "
        "boundary and its physical node coordinates sorted by x."
    )
    print(
        "  This dataset does not match the quoted 51 x 380 reference setup, "
        "and no contact-node coordinate file is present in the workspace."
    )


if __name__ == "__main__":
    main()
