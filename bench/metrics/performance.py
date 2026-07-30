"""Performance: what does the offline stage actually cost, and what do you get for it?

Wall-clock is reported but is the *weakest* number here -- it varies with machine, BLAS
threading and the SLSQP/NNLS implementations. The load-bearing metrics are the ones
that are reproducible:

* **Solver calls** (``bench.instrument``). The methods differ in how many constrained
  subproblems they solve, not in their arithmetic. CPG's cost is one NNLS per training
  snapshot per iteration -- ``O(R * n)`` NNLS solves -- to evaluate its selection
  criterion. mCPG adds one inequality-constrained solve per accepted generator
  ([NDEE22] Alg. 2 line 9). ADG adds angle evaluations, themselves NNLS-backed. These
  counts are machine-independent and are what a cost model should be built on.

* **``R`` at a target tolerance.** The quantity [BEE20] §5 argues is the *point* of a
  tolerance-driven method: the user asks for accuracy and the algorithm reports the
  cardinality needed. A method reaching a given accuracy with fewer generators is
  cheaper online forever after, which usually dominates any offline difference.

* **Online projection cost.** Because ``Pi_{K^+}`` is an NNLS solve and *not* linear in
  its argument, there is no projection matrix to precompute -- the online stage pays a
  constrained solve per query. This is the one cost that recurs after deployment, so it
  is measured separately from fitting.

**Cost per unit accuracy** ties the two together: two methods at the same tolerance can
differ several-fold in solver calls, and that ratio is the honest offline comparison.
"""

from __future__ import annotations

import time

import numpy as np

from .. import _paths  # noqa: F401
from ..instrument import count_solver_calls, summarize
from ..types import BasisResult, Dataset

from rb_vi_common.cone_projection import project_onto_cone


def online_projection_cost(dataset: Dataset, result: BasisResult,
                           n_queries: int = 25) -> dict[str, float]:
    """Time and solver calls for projecting unseen snapshots onto the fitted cone.

    Uses test snapshots where available -- projecting a *training* snapshot is the easy
    case, since it lies in the cone by construction for the selection-based methods and
    NNLS converges almost immediately.
    """
    G = np.asarray(result.generators, float)
    if G.shape[1] == 0:
        return {}
    source = dataset.test()
    if source is None:
        source = dataset.train()
    cols = min(n_queries, source.shape[1])
    if cols == 0:
        return {}
    queries = source[:, :cols]

    t0 = time.perf_counter()
    with count_solver_calls() as counts:
        for q in range(cols):
            project_onto_cone(queries[:, q], G, mass=None)
    elapsed = time.perf_counter() - t0
    totals = summarize(counts)
    return {
        "online_ms_per_query": 1e3 * elapsed / cols,
        "online_nnls_per_query": totals.get("nnls", 0) / cols,
    }


def evaluate(dataset: Dataset, result: BasisResult, *,
             measure_online: bool = True) -> dict[str, float]:
    """Performance row for one (dataset, method) cell."""
    row: dict[str, float] = {
        "fit_seconds": float(result.fit_seconds),
        "R": float(result.R),
    }
    for solver in ("nnls", "minimize", "lsq_linear"):
        row[f"calls_{solver}"] = float(result.solver_calls.get(solver, 0))
    total_calls = sum(result.solver_calls.get(s, 0)
                      for s in ("nnls", "minimize", "lsq_linear"))
    row["calls_total"] = float(total_calls)

    # Offline cost amortized over the basis actually produced: a method that spends
    # twice the solves to reach the same R is twice as expensive per generator.
    row["calls_per_generator"] = float(total_calls / result.R) if result.R else float("nan")

    n_train = dataset.train().shape[1]
    row["dim"] = float(dataset.dim)
    row["n_train"] = float(n_train)
    # The theoretical CPG budget, for reading the measured counts against: one NNLS
    # per training snapshot per iteration.
    row["calls_over_Rn"] = (float(total_calls / (result.R * n_train))
                            if result.R and n_train else float("nan"))

    if measure_online:
        row.update(online_projection_cost(dataset, result))
    return row
