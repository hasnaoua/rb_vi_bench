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
from pathlib import Path

from . import _paths  # noqa: F401  -- forces Agg before pyplot

import matplotlib.pyplot as plt
import numpy as np

from . import datasets as ds_mod, geometry, layout
from .adapters import DEFAULT_METHODS, METHODS
from .figures import FIGURE_EXCLUDED, STYLE
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


def _geom(dataset):
    """The dataset's field geometry, defaulting to an index-ordered curve."""
    return getattr(dataset, "geometry", None) or geometry.line_geometry()


def _draw(ax, x, truth, approx, title, color, geom=None):
    """Curve rendering: the truth and its reconstruction overlaid.

    ``x`` is the physical abscissa when the dataset supplies one, and the node index
    otherwise -- a distinction that only matters where the nodes are unevenly spaced,
    as on the Hertz contact arc.

    A ``mirrored_line`` geometry reflects both curves about the symmetry plane first, so
    the figure shows the full contact line the way [BEE20] Fig. 7-8 does. The same
    transform is applied to the snapshot and to its reconstruction, so nothing about
    their agreement changes.
    """
    geom = geom or geometry.line_geometry()
    if geom.kind == "mirrored_line":
        truth = geometry.mirror_half_profile(truth)
        approx = geometry.mirror_half_profile(approx)
        x = geom.coords
    ax.plot(x, truth, color="#222222", lw=1.6, label="snapshot")
    ax.plot(x, approx, color=color, lw=1.3, ls="--", label="reconstruction")
    ax.fill_between(x, truth, approx, color=color, alpha=0.18, lw=0)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel(geom.xlabel, fontsize=8)
    if geom.ylabel:
        ax.set_ylabel(geom.ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def _draw_axial_overlay(ax, truth, approx, geom, color, label):
    """HF vs ROM as an axial profile -- ``greedy.viz.publication``'s convention.

    ``z`` runs up the y-axis and force along x, the angular direction collapsed by its
    mean. The field is essentially axisymmetric, so this loses nothing and shows the
    quantity the 2-D maps make you infer: where along the cladding the contact sits and
    how far the reduced model's front is displaced from the HF one.

    The band marks the active contact span at the same 1e-3 threshold ratio the
    publication figures use, and the errors quoted are for this profile.
    """
    hf = geometry.axial_profile(truth, geom)
    rom = geometry.axial_profile(approx, geom)
    z = geometry.axial_coordinate(geom)
    scale, exponent = geometry.force_scale(float(np.max(np.abs(hf))) if hf.size else 0.0)

    mask = geometry.active_span(hf)
    if mask.any():
        ax.axhspan(float(z[mask].min()), float(z[mask].max()),
                   color="#dbe4f5", alpha=0.55, zorder=0)

    ax.plot(hf / scale, z, color="#7f8c8d", lw=3.0, alpha=0.6, label="HF snapshot",
            zorder=3)
    ax.plot(rom / scale, z, color=color, lw=2.0, ls="--", marker="o",
            markevery=max(1, z.size // 12), ms=3.6, markerfacecolor="white",
            markeredgewidth=0.9, label=label, zorder=4)

    denom = float(np.linalg.norm(hf)) or 1.0
    rel = float(np.linalg.norm(hf - rom)) / denom
    ax.axvline(0.0, color="#27272a", lw=0.8, zorder=2)
    ax.set_ylim(z[0], z[-1])
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Normal force"
                  + (rf" ($\times 10^{{{exponent}}}$)" if exponent else ""), fontsize=8)
    ax.set_ylabel(r"$z$ [mm]", fontsize=8)
    ax.set_title(f"axial profile — rel {rel:.2e}", fontsize=9)
    ax.grid(True, color="#d8dce0", ls="--", lw=0.6, alpha=0.8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower right", fontsize=7, frameon=True)


def _draw_field_triptych(fig, axes, truth, approx, geom, title, color="#e8760a",
                         label="reconstruction"):
    """HF | ROM | error maps, plus the axial-profile superposition.

    The three maps share the first two colour limits so HF and reconstruction are
    visually comparable; the error panel gets its own, since it is typically orders of
    magnitude smaller and would be a flat field on the shared scale.

    The fourth panel is the axial profile, in ``greedy.viz.publication``'s layout. The
    maps and the profile answer different questions: the maps show *where on the surface*
    the discrepancy sits and confirm the field is axisymmetric; the profile shows how far
    the reduced model's contact front is displaced along the cladding, which is the
    quantity the publication figures report and is hard to read off a colour map.
    """
    vmin, vmax = geometry.field_limits([truth, approx], geom)
    rel = geometry.relative_error_field(truth, approx)
    rlo, rhi = geometry.field_limits([rel], geom)
    for ax, values, lab, lo, hi in (
        (axes[0], truth, "HF snapshot", vmin, vmax),
        (axes[1], approx, "reconstruction", vmin, vmax),
        (axes[2], rel, "relative error (/ HF peak)", rlo, rhi),
    ):
        im = geometry.draw_field(ax, values, geom, vmin=lo, vmax=hi)
        ax.set_title(lab, fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel(geom.ylabel, fontsize=8)
    if len(axes) > 3:
        _draw_axial_overlay(axes[3], truth, approx, geom, color, label)
    fig.suptitle(title, fontsize=10)


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
    geom = _geom(dataset)
    x = geom.coords if geom.coords is not None else np.arange(columns.shape[0])
    method_dir = layout.ensure(layout.method_dir(out_dir, name, method_key))

    written: list[Path] = []
    for label, idx in (("best", int(np.nanargmin(rel))), ("worst", int(np.nanargmax(rel)))):
        head = (f"{name} — {METHODS[method_key].label} (R={result.R})\n"
                f"{label}: snapshot {idx}, rel. err {rel[idx]:.3e}")
        if geom.is_field:
            # A structured grid also gets the axial-profile panel; a scatter geometry
            # has no axis to collapse along, so it keeps the three maps.
            n = 4 if geom.kind == "grid" else 3
            fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 2.0, 3.6))
            _draw_field_triptych(fig, axes, columns[:, idx], approx[:, idx], geom, head,
                                 color=color, label=METHODS[method_key].label)
            fig.tight_layout(rect=(0, 0, 1, 0.87))
        else:
            fig, ax = plt.subplots(figsize=(6.6, 4.0))
            _draw(ax, x, columns[:, idx], approx[:, idx],
                  f"{label}: snapshot {idx}, rel. err {rel[idx]:.3e}", color, geom)
            ax.legend(fontsize=8, frameon=False)
            fig.suptitle(f"{name} — {METHODS[method_key].label} (R={result.R})", fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = method_dir / f"{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _figure_for_dataset_field(dataset, name, fitted, columns, out_dir, geom) -> Path | None:
    """Field version: rows are methods, columns are the best and worst error fields.

    Error fields rather than reconstructions, on **one shared colour scale** across every
    method, because that is what makes the panels comparable -- each method's own
    reconstruction looks near-identical to the HF snapshot at this scale, and the
    difference is the whole content. The HF snapshot itself is drawn once on the top row
    for reference.

    The error is **relative**, normalized by each snapshot's own peak, so a row can be
    compared against another row whose snapshot has a different magnitude. An absolute
    field could not: the best and worst cases are different snapshots.
    """
    keys = list(fitted)
    errs: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
    for key in keys:
        rel, approx = _ranking(dataset, fitted[key], columns)
        if np.all(np.isnan(rel)):
            continue
        b, w = int(np.nanargmin(rel)), int(np.nanargmax(rel))
        errs[key] = (b, w,
                     geometry.relative_error_field(columns[:, b], approx[:, b]),
                     geometry.relative_error_field(columns[:, w], approx[:, w]))
    if not errs:
        return None

    lo, hi = geometry.field_limits([e for v in errs.values() for e in v[2:]], geom)
    n = len(errs) + 1
    fig, axes = plt.subplots(n, 2, figsize=(9.6, 2.6 * n), squeeze=False)

    any_best = next(iter(errs.values()))[0]
    any_worst = next(iter(errs.values()))[1]
    for ax, idx, lbl in ((axes[0][0], any_best, "best"), (axes[0][1], any_worst, "worst")):
        vmin, vmax = geometry.field_limits([columns[:, idx]], geom)
        im = geometry.draw_field(ax, columns[:, idx], geom, vmin=vmin, vmax=vmax)
        ax.set_title(f"HF snapshot #{idx} ({lbl} case)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    for row, (key, (b, w, eb, ew)) in enumerate(errs.items(), start=1):
        for ax, idx, e, lbl in ((axes[row][0], b, eb, "best"), (axes[row][1], w, ew, "worst")):
            im = geometry.draw_field(ax, e, geom, vmin=lo, vmax=hi, cmap="magma")
            ax.set_title(f"{METHODS[key].label} — rel. error, {lbl} #{idx}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

    fig.suptitle(f"{name} — relative reconstruction error fields "
                 f"(R={next(iter(fitted.values())).R}, / HF peak, shared scale)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = layout.ensure(layout.reconstruction_dir(out_dir, name)) / "all_methods.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_for_dataset(dataset, name, fitted, columns, out_dir) -> Path | None:
    """All methods in one figure: rows are methods, columns are best / worst."""
    keys = [k for k in fitted]
    if not keys:
        return None
    geom = _geom(dataset)
    if geom.is_field:
        return _figure_for_dataset_field(dataset, name, fitted, columns, out_dir, geom)
    fig, axes = plt.subplots(len(keys), 2, figsize=(11.0, 2.5 * len(keys)),
                             squeeze=False)
    x = geom.coords if geom.coords is not None else np.arange(columns.shape[0])
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
                  f"{METHODS[key].label} — {label}: #{idx}, {rel[idx]:.2e}", color, geom)
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
    p.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
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
        for m in [k for k in args.methods if k not in FIGURE_EXCLUDED]:
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
