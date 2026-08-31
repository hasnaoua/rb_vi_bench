"""The argument fragments the five entry points share.

``runner``, ``report``, ``figures``, ``reconstruction`` and ``decrement`` are separate
commands on purpose -- they have genuinely different costs, and a half-hour grid run
should not be coupled to a two-second re-render. But they take overlapping arguments,
and those had drifted: ``--out`` was declared five times with five different help
strings (three of them absent), ``--datasets`` twice with different defaults, and
``--subsample`` twice with different ones. A reader could not tell which differences
were deliberate.

Each helper here declares one argument with one wording, and takes the default as a
parameter -- because *that* is what legitimately differs between commands. ``runner``
writes to ``results/`` while the figure modules write to ``results/figures/``; that is
a real difference and stays visible at the call site.

**On ``--separate``.** It was called ``--split``, which now collides with the thing
every dataset has: a train/test split. Reading ``--split`` on a command whose figures
report held-out error suggests it selects the evaluation set, when all it does is write
one PNG per metric instead of one combined panel. ``--split`` still works and is
undocumented rather than removed, so saved commands keep running.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import _paths


def add_results(p: argparse.ArgumentParser, default: Path = _paths.RESULTS) -> None:
    """Where to read a finished run from."""
    p.add_argument("--results", type=Path, default=default,
                   help="directory holding grid.csv (default: %(default)s)")


def add_out(p: argparse.ArgumentParser, default: Path | None, *, what: str) -> None:
    """Where to write. ``default=None`` means the command derives it from --results."""
    p.add_argument("--out", type=Path, default=default, help=f"where to write {what}")


def add_datasets(p: argparse.ArgumentParser, default: list[str]) -> None:
    p.add_argument("--datasets", nargs="*", default=default,
                   help=f"dataset keys to run (default: {default})")


def add_methods(p: argparse.ArgumentParser, default: list[str], *, known: list[str]) -> None:
    p.add_argument("--methods", nargs="*", default=default,
                   help="default: one canonical implementation per algorithm; "
                        f"pass names to widen (all: {known})")


def add_subsample(p: argparse.ArgumentParser, default: int | None) -> None:
    p.add_argument("--subsample", type=int, default=default,
                   help="cap training snapshots (changes the numbers; recorded in output)")


def add_separate(p: argparse.ArgumentParser, *, what: str) -> None:
    """``--separate``: one PNG per {what} instead of one combined figure."""
    p.add_argument("--separate", "--split", dest="separate", action="store_true",
                   help=f"also write one PNG per {what} under <out>/<dataset>/")


__all__ = ["add_datasets", "add_methods", "add_out", "add_results", "add_separate",
           "add_subsample"]
