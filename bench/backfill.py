"""Add exactly-derivable columns to a grid CSV written before they existed.

A metric column normally arrives by re-running the grid, which is the honest default and
what should happen whenever a column carries new *information*. This module is for the
narrower case where a column is a **closed-form function of columns the CSV already
has**, so re-running would spend an hour recomputing NNLS solves to arrive at a number
already determined by the file on disk.

Exactly one column qualifies today. ``section_width_ratio`` is
``(V / V_ortho)^(1/(R-1))``, and ``section_log_volume``, ``R`` and ``dim`` are all
recorded, so the reference volume can be rebuilt from its closed form and divided out.
Nothing is estimated, interpolated, or defaulted.

**The derivation is checked, not trusted.** ``section_vol_ratio`` is a different function
of the same three inputs and is already in the file, so re-deriving *it* and comparing
against the stored value tests the arithmetic against data the runner produced
independently. A mismatch beyond float round-off aborts the backfill rather than writing
a plausible-looking column. Cells where the stored ratio has underflowed to 0 are skipped
by that check -- they carry no digits to compare -- but are still filled, since the log
form does not underflow and is exactly where the derived column earns its place.

Rows that predate ``section_log_volume`` entirely, or where R < 2, get an empty cell. An
absent value reads as absent; it must not read as zero.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import _paths
from .tabular import num, read_rows, write_csv

RESULTS = _paths.ROOT / "results"

#: Round-off ceiling for the self-check. The derivation and the runner evaluate the same
#: quantity through different expressions, so they agree to a few ulps of the exponential,
#: not bit-for-bit; observed worst case across 399 cells is 1.1e-13.
CHECK_RTOL = 1e-9


def log_reference_volume(R: int, m: float, height: float = 1.0) -> float:
    """Log section volume of ``R`` orthogonal directions, cut at ``<x, u> = height``.

    Vertices are ``sqrt(m) * height * e_i``, whose edge Gram is ``m height^2 (I + J)``
    with determinant ``m^(R-1) height^(2(R-1)) R``. Kept here in closed form so the
    backfill does not need the generators, which the CSV does not store.
    """
    return (-math.lgamma(R)
            + 0.5 * ((R - 1) * math.log(m) + math.log(R))
            + (R - 1) * math.log(height))


def derive_section_width(row: dict) -> tuple[float | None, float | None]:
    """``(width_ratio, rederived_vol_ratio)`` for one row, or ``(None, None)``."""
    R, m, log_vol = num(row, "R"), num(row, "dim"), num(row, "section_log_volume")
    if any(math.isnan(v) for v in (R, m, log_vol)) or R < 2 or m <= 0:
        return None, None
    if not math.isfinite(log_vol):
        # -inf is a degenerate (rank-deficient) section: zero volume, zero width.
        return (0.0, 0.0) if log_vol < 0 else (None, None)
    log_ratio = log_vol - log_reference_volume(int(R), m)
    return math.exp(log_ratio / (int(R) - 1)), math.exp(log_ratio)


def backfill(path: Path) -> tuple[int, int]:
    """Fill ``section_width_ratio`` in ``path``. Returns ``(filled, checked)``."""
    rows = read_rows(path)
    filled = checked = 0
    for row in rows:
        if row.get("section_width_ratio") not in (None, ""):
            continue
        width, vol = derive_section_width(row)
        if width is None:
            row.setdefault("section_width_ratio", "")
            continue
        stored = num(row, "section_vol_ratio")
        if not math.isnan(stored) and stored > 1e-280:
            if abs(vol - stored) > CHECK_RTOL * stored:
                raise AssertionError(
                    f"{path}: re-derived section_vol_ratio {vol!r} disagrees with the "
                    f"stored {stored!r} for {row.get('dataset')}/{row.get('method')} "
                    f"at R={row.get('R')}; the backfill formula does not match the "
                    "runner and nothing has been written"
                )
            checked += 1
        row["section_width_ratio"] = repr(width)
        filled += 1
    write_csv(path, rows)
    return filled, checked


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csvs", nargs="*", type=Path,
                   default=[RESULTS / "grid.csv", RESULTS / "sweep_dense" / "grid.csv"])
    args = p.parse_args(argv)
    for path in args.csvs:
        if not path.is_file():
            print(f"[skip] {path} not found")
            continue
        filled, checked = backfill(path)
        print(f"{path}: filled {filled} cells, {checked} verified against "
              "the independently stored section_vol_ratio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
