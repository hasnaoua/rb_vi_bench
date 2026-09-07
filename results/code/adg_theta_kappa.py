"""Gram condition number against ADG's own angular-defect criterion, per dataset.

Produces the three figures behind docs/rb_vi_bench.tex, Section 6.5 (``Conditioning against
ADG's own stopping criterion``): one PNG per dataset, each plotting kappa -- the Gram
condition number of ADG's cone -- against theta_max, the largest remaining angular
defect ADG itself computes every round and compares to its tolerance to decide whether
to keep going.

**Why this needs its own script rather than reusing bench.figures.** Every other figure
in this project plots a column already sitting in grid.csv against R. theta_max is not
such a column -- it is an intermediate the greedy computes on the way to a generator and
throws away once ``compute_phases()`` returns, so it does not exist anywhere a CSV reader
could find it. Getting it back means re-running ``AngularDefectGreedy`` directly, which
this script does once per dataset (to near-full convergence, epsilon and zero_tol both
tiny so the run does not stop early), and reading off ``theta_max_history`` --
``batch_size_history`` -- one entry per round, aligned, so ``2 + cumsum(batch_size_history)``
gives the cardinality R that round produced.

**kappa is not recomputed.** Once the round's R is known, kappa is looked up from
``results/sweep_dense/grid.csv`` at method="adg", that exact R -- the identical number
every other conditioning figure in the report plots. This is what keeps the figure
honest: it relocates an existing, already-checked measurement onto a new axis, rather
than producing a second, potentially-diverging estimate of the same quantity. That
lookup can miss (a round's R exceeds the sweep's cardinality cap, currently 40); missed
rounds are dropped rather than filled with a substitute.

Run from anywhere:

    .venv/bin/python results/code/adg_theta_kappa.py
    .venv/bin/python results/code/adg_theta_kappa.py --out results/figures/_conditioning

Requires the dense sweep to already exist (``bench.runner --deltas --cardinalities ...
--out results/sweep_dense``, see the README's reproduction command); it reads that CSV,
it does not run the grid itself.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

# Scripts under results/code/ sit two levels below the repo root, and are typically
# invoked as a bare file path rather than a module, so sys.path[0] is this file's own
# directory, not the root `import bench` needs. Fix that up before anything else.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from bench import _paths  # noqa: F401  -- forces the Agg backend before pyplot

import matplotlib.pyplot as plt

from bench import datasets as ds_mod
from bench.tabular import num, read_rows

from greedy.core.angle_defect_greedy import AngularDefectGreedy

#: One panel per dataset key, in the order they appear in the report.
DATASETS: tuple[tuple[str, str, str, str], ...] = (
    # (registry key, output file stem, line colour, title used in the figure)
    ("fem_lambda", "theta_kappa_halfdisks", "#1f4e9c", "Half-disks of Hertz"),
    ("physics", "theta_kappa_pellet", "#1b7f4f", "3D Pellet-Cladding"),
    ("membrane_2d", "theta_kappa_membrane", "#e8760a", "membrane (2-D)"),
)

#: How tightly ADG is run before its history is read off. Tiny relative to any
#: tolerance used elsewhere in the benchmark, so the run approaches full cardinality
#: (bounded by n_train) rather than stopping while theta_max is still large.
EPSILON = 1e-12
ZERO_TOL = 1e-14

#: Cardinalities available to look kappa up against -- the dense sweep's cap.
MAX_R_IN_SWEEP = 40


def gram_cond_by_R(sweep_grid: pathlib.Path, dataset_name: str) -> dict[int, float]:
    """``{R: gram_cond}`` for ADG's own rows in the matched-cardinality sweep."""
    rows = [
        r for r in read_rows(sweep_grid)
        if not r.get("skip_reason")
        and r.get("mode") == "cardinality"
        and r.get("method") == "adg"
        and r.get("dataset") == dataset_name
    ]
    out: dict[int, float] = {}
    for r in rows:
        k = num(r, "gram_cond")
        if k == k and k > 0:  # excludes NaN
            out[int(num(r, "R"))] = k
    return out


def theta_kappa_trajectory(dataset_key: str, cond_by_R: dict[int, float]
                           ) -> list[tuple[float, float, int]]:
    """``[(theta_max_deg, kappa, R), ...]`` in ITERATION order (theta_max decreasing).

    Runs ``AngularDefectGreedy`` once, directly, rather than going through
    ``bench.adapters.fit_greedy_adg`` -- the adapter returns only the finished
    ``BasisResult`` and discards the per-round histories this plot needs.
    """
    dataset = ds_mod.load(dataset_key)
    rows = np.ascontiguousarray(dataset.train().T)
    model = AngularDefectGreedy(
        snapshots=rows, epsilon=EPSILON, zero_tol=ZERO_TOL, normalize_snapshots=True,
    )
    model.compute_phases()

    theta_deg = np.degrees(np.asarray(model.theta_max_history))
    cumulative_R = 2 + np.cumsum(model.batch_size_history)  # 2 = the initial pair

    points = [
        (float(theta_deg[i]), cond_by_R[R], int(R))
        for i, R in enumerate(cumulative_R)
        if R in cond_by_R
    ]
    points.sort(key=lambda p: -p[0])
    return points


def plot_one(points: list[tuple[float, float, int]], color: str, title: str,
             out_path: pathlib.Path) -> None:
    if not points:
        raise ValueError(f"no matched-R points for {title!r}; is the dense sweep present "
                         f"and does it cover R up to {MAX_R_IN_SWEEP}?")
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    Rs = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(x, y, color=color, marker="o", ms=4.2, lw=1.5, alpha=0.9)
    step = max(1, len(x) // 6)
    for i in range(0, len(x), step):
        ax.annotate(f"$R$={Rs[i]}", (x[i], y[i]), fontsize=6.8, color="#555555",
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()

    eps_inv = 1.0 / np.finfo(float).eps
    if min(y) < eps_inv < max(y) * 10:
        ax.axhline(eps_inv, color="#b03a2e", ls=":", lw=1.1)
        ax.text(0.02, eps_inv, "numerically singular →", color="#b03a2e",
                fontsize=7.5, ha="left", va="bottom", transform=ax.get_yaxis_transform())

    ax.set_xlabel(r"ADG's reported $\theta_{\max}$ [deg]  (iteration progresses $\rightarrow$)")
    ax.set_ylabel(r"Gram condition number $\kappa$")
    ax.set_title(f"{title}: $\\kappa$ vs. ADG's own angular-error criterion", fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=pathlib.Path, default=ROOT / "results",
                   help="directory holding sweep_dense/grid.csv (default: %(default)s)")
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "figs",
                   help="directory to write the three PNGs into (default: %(default)s)")
    args = p.parse_args(argv)

    sweep_grid = args.results / "sweep_dense" / "grid.csv"
    if not sweep_grid.is_file():
        raise SystemExit(
            f"{sweep_grid} not found; run the dense cardinality sweep first -- see "
            "the reproduction command in README.md / docs/rb_vi_bench.tex."
        )

    written = []
    for key, stem, color, title in DATASETS:
        dataset_name = ds_mod.load(key).name
        cond_by_R = gram_cond_by_R(sweep_grid, dataset_name)
        points = theta_kappa_trajectory(key, cond_by_R)
        out_path = args.out / f"{stem}.png"
        plot_one(points, color, title, out_path)
        written.append(out_path)
        print(f"{dataset_name:22s} {len(points):3d} rounds  "
              f"theta_max {points[-1][0]:.4g}..{points[0][0]:.4g} deg  "
              f"kappa {min(p[1] for p in points):.3e}..{max(p[1] for p in points):.3e}  "
              f"-> {out_path}")

    print(f"\n{len(written)} figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
