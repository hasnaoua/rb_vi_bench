"""
Shared reading conventions for the FEM_SOLS snapshot archive.

``lambda_snapshots`` (the dual multipliers) and ``primal_snapshots`` (the
displacements) read the same archive, laid out as ``rad=<value>/`` folders, and
had identical copies of the parsing below. The two differ only in *which* file
they pull out of each folder, so only that part stays in each module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

DEFAULT_ARCHIVE = Path("data/FEM_SOLS.zip")
DEFAULT_DIRECTORY = Path("data/FEM_SOLS")

# Folder names carry the parameter, e.g. "rad=0.8".
RADIUS_PATTERN = re.compile(r"^rad=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$")


@dataclass(frozen=True)
class SnapshotRecord:
    """One snapshot read from the archive, flattened to a vector."""

    radius: float
    radius_label: str
    source: str
    original_shape: tuple[int, ...]
    values: np.ndarray


def default_input_path() -> Path:
    """Prefer the zip archive, fall back to an unpacked directory."""
    if DEFAULT_ARCHIVE.exists():
        return DEFAULT_ARCHIVE
    return DEFAULT_DIRECTORY


def parse_radius_from_parts(parts: tuple[str, ...]) -> tuple[float, str] | None:
    """Find the ``rad=<value>`` component of a path, as (value, label)."""
    for part in parts:
        match = RADIUS_PATTERN.match(part)
        if match:
            label = match.group(1)
            return float(label), label
    return None


def load_npy_bytes(data: bytes) -> np.ndarray:
    return np.load(BytesIO(data), allow_pickle=False)


def flatten_snapshot(array: np.ndarray, source: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Flatten a snapshot to 1-D, keeping its original shape for reporting.

    Rejects empty and non-finite data here rather than letting a NaN reach the
    greedies, where it would silently poison every projection downstream.
    """
    values = np.asarray(array, dtype=float)
    original_shape = tuple(values.shape)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f"{source} is empty")
    if not np.all(np.isfinite(values)):
        bad_count = int(values.size - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"{source} contains {bad_count} non-finite values")
    return values, original_shape
