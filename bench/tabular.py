"""Reading and writing the benchmark's CSV artefacts.

One home for the handful of helpers that ``runner``, ``report``, ``figures`` and
``decrement`` all need. They were duplicated across those modules -- ``_num`` was
byte-identical in two of them -- which is the kind of duplication that goes wrong
quietly: a fix to how a CSV value is parsed, applied to one copy, would leave the
tables and the figures disagreeing about the same cell with nothing raised.

The CSV is the artefact; every consumer reads it back rather than sharing in-memory
state, so these are the only functions that need to agree on its conventions.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


#: What to tell a user whose grid CSV is not there. One string, because it names a
#: command: ``__main__`` exposes the runner as ``python -m bench run``, and a second copy
#: of this hint elsewhere would keep naming whatever form was current when it was written.
MISSING_GRID_HINT = "not found; run `python -m bench run` first"


def read_rows(path: Path) -> list[dict]:
    """Every row of a grid CSV, unparsed.

    Raises ``FileNotFoundError`` rather than returning empty: an absent grid and a grid
    of zero usable rows need different messages, and only the caller knows whether a
    missing file is fatal (``figures``, ``report``) or merely one of several optional
    inputs (``decrement``, which takes two grids and skips whichever is absent).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} {MISSING_GRID_HINT}")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> float:
    """A cell as a float, with absent/blank/unparseable all mapping to ``nan``.

    ``nan`` rather than an exception or 0.0: cells legitimately differ in which columns
    they carry -- a dataset without an inf-sup input has no beta columns, one without a
    train/test split has no test columns -- and a 0.0 there would read as a perfect
    score rather than as absent.
    """
    raw = row.get(key, "")
    if raw in ("", None):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def fmt(v: float, spec: str = ".4g") -> str:
    """A float for a text table; ``nan`` prints as ``-`` rather than 'nan'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return format(v, spec)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows, taking the header as the union of every row's keys.

    Cells legitimately produce different columns, so a fixed header taken from the first
    row would silently drop whatever later rows added.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def rows_by_dataset_and_method(path: Path, mode: str) -> dict[str, dict[str, list[dict]]]:
    """``{dataset: {method: [row, ...]}}`` for one run mode, skipped cells dropped."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in read_rows(path):
        if r.get("mode") != mode or r.get("skip_reason"):
            continue
        out[r["dataset"]][r["method"]].append(r)
    return out
