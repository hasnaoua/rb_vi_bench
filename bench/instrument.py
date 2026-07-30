"""Count the solver calls the cone algorithms actually make.

Wall-clock time alone is a poor performance metric here: the methods differ mainly
in *how many* and *which kind* of constrained subproblem they solve per iteration,
and that count is machine-independent and reproducible where seconds are not.

* CPG (both transcriptions) spends its time in NNLS -- one solve per training
  snapshot per iteration, to evaluate the selection criterion.
* mCPG adds an inequality-constrained least-squares solve per accepted generator:
  [NDEE22] Algorithm 2 line 9, ``Upsilon_r in K_{r-1} cap (theta_q - W^+)``. Family A
  solves it with SLSQP (``minimize``), family B with its own
  ``solve_cone_shift_projection``.
* ADG additionally evaluates angles, which are NNLS-backed projections.

Every module binds its solvers at import time (``from scipy.optimize import nnls``),
so patching ``scipy.optimize.nnls`` would not be seen. The counters therefore replace
the attribute *on each importing module*, which is why the target list below names
modules rather than the scipy namespace. Adding a solver import to any of those
modules without adding it here silently undercounts.
"""

from __future__ import annotations

import contextlib
from collections import Counter

from . import _paths  # noqa: F401  -- sys.path side effect, must precede the imports

# (module_path, attribute) pairs. Kept explicit so an undercount is a visible
# omission rather than a silent one.
_TARGETS: tuple[tuple[str, str], ...] = (
    ("rb_vi_common.cone_projection", "nnls"),
    ("rb_vi_common.cone_greedy", "minimize"),
    ("rb_vi_common.reduction", "minimize"),
    ("greedy.core.reduction_common", "nnls"),
    ("greedy.core.reduction_common", "minimize"),
    ("greedy.core.reduction_common", "lsq_linear"),
)


@contextlib.contextmanager
def count_solver_calls():
    """Yield a ``Counter`` of solver invocations made inside the block.

    Keys are ``"<solver>"`` aggregated across modules (e.g. ``"nnls"``), because a
    method's cost profile is about the kind of subproblem, not which file issued it.
    Per-module detail is kept under ``"<module>.<solver>"`` for debugging.
    """
    import importlib

    counts: Counter[str] = Counter()
    patched: list[tuple[object, str, object]] = []

    for module_path, attr in _TARGETS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # A module that is not importable in this environment cannot be
            # called either, so there is nothing to miscount.
            continue
        original = getattr(module, attr, None)
        if original is None:
            continue

        def make_counter(fn=original, name=attr, mod=module_path):
            def wrapper(*args, **kwargs):
                counts[name] += 1
                counts[f"{mod}.{name}"] += 1
                return fn(*args, **kwargs)

            return wrapper

        setattr(module, attr, make_counter())
        patched.append((module, attr, original))

    try:
        yield counts
    finally:
        for module, attr, original in patched:
            setattr(module, attr, original)


def summarize(counts) -> dict[str, int]:
    """Drop the per-module detail keys, keeping the aggregated solver totals."""
    return {k: int(v) for k, v in sorted(counts.items()) if "." not in k}
