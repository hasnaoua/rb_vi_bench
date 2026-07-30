"""Best- and worst-case reconstruction figures, per method, per dataset.

The metric figures say *how much* error a cone leaves. These say *what* it leaves: which
snapshot each method represents best, which it represents worst, and where in the field
the discrepancy sits. On contact problems that is usually the interesting part -- a cone
built from selected snapshots tends to fail on a support that has moved rather than on
amplitude, and only the profile shows it.

**Best and worst are ranked by per-snapshot relative error**,
``||theta - Pi_K(theta)|| / ||theta||``, not by the shared-denominator column the grid
reports. The grid divides every snapshot by one global normalizer
(``max_q ||theta_q||`` over the training set) so that methods stay comparable and the
number can be read against the tolerance that produced it. That is the wrong ranking
here: under a shared denominator a small-magnitude snapshot looks well reconstructed
merely because it is small, so "worst" would systematically pick the largest snapshot
rather than the least well represented. The two differ by 1.5x on ``hertz_pressure`` and
70x on ``physics``, whose snapshot norms span a factor of 604.

Snapshots whose norm is numerically zero are excluded from the ranking: their relative
error is 0/0. ``physics`` has two such columns, and they also violate the ADG spec's
precondition ``S subset R_+^m \\ {0}``.

The reconstruction drawn is always the one the error column actually scored -- NNLS cone
projection for the cone methods and NMF, unconstrained least squares for the POD control
-- via ``metrics.precision.reconstruct``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from . import _paths  # noqa: F401  -- forces Agg before pyplot

import matplotlib.pyplot as plt
import numpy as np

from . import datasets as ds_mod, layout
from .adapters import METHODS
from .figures import STYLE
from .metrics.precision import reconstruct, uses_cone_projection
from .runner import _subsample

RESULTS = _paths.ROOT / "results"


def _ranking(dataset, result, columns) -> tuple[np.ndarray, np.ndarray]:
    """Per-snapshot relative errors and the reconstructions, over ``columns``."""
    approx = reconstruct(columns, result.generators, cone=uses_cone_projection(result))
    err = np.linalg.norm(columns - approx, axis=0)
    norms = np.linalg.norm(columns, axis=0)
    rel = np.divide(err, norms, out=np.full_like(err, np.nan),
                    where=norms > 1e-8 * max(norms.max(), 1e-300))
    return rel, approx


def _draw(ax, x, truth, approx, title, color):
    ax.plot(x, truth, color="#222222", lw=1.6, label="snapshot")
    ax.plot(x, approx, color=color, lw=1.3, ls="--", label="reconstruction")
    ax.fill_between(x, truth, approx, color=color, alpha=0.18, lw=0)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel("component index", fontsize=8)
    ax.tick_params(labelsize=7)


def figures_for_method(dataset, name, method_key, result, columns, out_dir) -> list[Path]:
    """One standalone PNG per case, at ``<out>/reconstruction/<dataset>/<method>/``.

    ``best.png`` and ``worst.png`` are written separately rather than as two panels of
    one image, so either can be dropped into a document without cropping. The
    reconstruction tree is kept apart from the metric figures: they are indexed by
    different things (a snapshot here, a cardinality there) and interleaving them in one
    dataset folder made the directory hard to read.
    """
    rel, approx = _ranking(dataset, result, columns)
    if np.all(np.isnan(rel)):
        return []
    color = STYLE.get(method_key, {}).get("color", "#c0392b")
    x = np.arange(columns.shape[0])
    method_dir = layout.ensure(layout.method_dir(out_dir, name, method_key))

    written: list[Path] = []
    for label, idx in (("best", int(np.nanargmin(rel))), ("worst", int(np.nanargmax(rel)))):
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        _draw(ax, x, columns[:, idx], approx[:, idx],
              f"{label}: snapshot {idx}, rel. err {rel[idx]:.3e}", color)
        ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{name} — {METHODS[method_key].label} (R={result.R})", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = method_dir / f"{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def figure_for_dataset(dataset, name, fitted, columns, out_dir) -> Path | None:
    """All methods in one figure: rows are methods, columns are best / worst."""
    keys = [k for k in fitted]
    if not keys:
        return None
    fig, axes = plt.subplots(len(keys), 2, figsize=(11.0, 2.5 * len(keys)),
                             squeeze=False)
    x = np.arange(columns.shape[0])
    for row, key in enumerate(keys):
        result = fitted[key]
        rel, approx = _ranking(dataset, result, columns)
        color = STYLE.get(key, {}).get("color", "#c0392b")
        if np.all(np.isnan(rel)):
            for ax in axes[row]:
                ax.axis("off")
            continue
        best, worst = int(np.nanargmin(rel)), int(np.nanargmax(rel))
        for ax, idx, label in ((axes[row][0], best, "best"), (axes[row][1], worst, "worst")):
            _draw(ax, x, columns[:, idx], approx[:, idx],
                  f"{METHODS[key].label} — {label}: #{idx}, {rel[idx]:.2e}", color)
    axes[0][0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"{name} — best / worst reconstruction per method "
                 f"(R={next(iter(fitted.values())).R}, ranked by per-snapshot rel. error)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = layout.ensure(layout.reconstruction_dir(out_dir, name)) / "all_methods.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="best/worst reconstruction figures")
    p.add_argument("--datasets", nargs="*", default=list(ds_mod.FAST) + list(ds_mod.HEAVY))
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--R", type=int, default=8,
                   help="matched cardinality to fit every method at (default 8)")
    p.add_argument("--subsample", type=int, default=200)
    p.add_argument("--split", action="store_true",
                   help="also write one PNG per method under <out>/<dataset>/")
    p.add_argument("--out", type=Path, default=RESULTS / "figures")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for key in args.datasets:
        try:
            dataset = _subsample(ds_mod.load(key), args.subsample)
        except Exception as exc:                       # noqa: BLE001
            print(f"[skip] {key}: {type(exc).__name__}: {exc}")
            continue

        # Evaluate on held-out snapshots where they exist: reconstructing a TRAINING
        # snapshot is the easy case, since for the selection-based methods a chosen
        # snapshot lies in the cone exactly and its "best" panel is a tautology.
        columns = dataset.test()
        if columns is None:
            columns = dataset.train()

        fitted = {}
        for m in args.methods:
            try:
                fitted[m] = METHODS[m].fit(dataset, R=args.R)
            except Exception as exc:                   # noqa: BLE001
                print(f"[skip] {key}/{m}: {type(exc).__name__}: {exc}")
        if not fitted:
            continue

        path = figure_for_dataset(dataset, dataset.name, fitted, columns, args.out)
        if path:
            written.append(path)
        if args.split:
            for m, result in fitted.items():
                written.extend(figures_for_method(
                    dataset, dataset.name, m, result, columns, args.out))

    for path in written:
        print(path)
    print(f"\n{len(written)} figures at R={args.R}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
