"""What both adapter families need, and neither should own.

``family_a`` and ``family_b`` wrap two independent implementations and share no code by
design -- that independence is what makes ``metrics.agreement`` meaningful. But the
*contract with the runner* is shared, and it belongs in one place: ``run_cell`` catches
whatever an adapter raises and records it verbatim as the cell's ``skip_reason``, so a
message that differs between the families would split one cause into two rows in
``report.skip_summary``.

This module holds only that contract. It deliberately does not import either family, so
``adapters/__init__`` can keep importing both without a cycle.
"""

from __future__ import annotations


def require_delta(delta: float | None) -> float:
    """``delta`` is mandatory whenever ``R`` is not given -- say so once, for both families.

    Every adapter takes exactly one of the two knobs, and the runner always supplies one.
    Stated here rather than at a dozen call sites, and as a named error rather than the
    ``TypeError`` that ``float(None)`` would raise three frames deeper.

    The wording is load-bearing: ``runner.run_cell`` writes it straight into
    ``grid.csv``'s ``skip_reason`` column, and ``report.skip_summary`` groups cells by
    that string.
    """
    if delta is None:
        raise ValueError("pass delta= when R= is not given")
    return float(delta)
