"""How a snapshot vector lays out in space, and how to draw it.

A dual snapshot is a vector of nodal multipliers, but the nodes are *not* a sequence --
they sit on a contact surface. Plotting such a vector against its component index is only
honest when the contact set really is one-dimensional and ordered. For the physics
dataset it is neither: its 7676 entries are a structured **76 x 101 grid** over a
quarter-cylinder (76 angular stations across a 90-degree sector, 101 axial stations over
5 mm), so an index plot cuts the surface into 76 arbitrary strips and lays them end to
end. Adjacent points on that curve are usually not adjacent on the cladding, and a
contact patch that is compact in (theta, z) appears as scattered spikes.

Three layouts are represented, one per kind of contact discretization in the merge:

``grid``    a structured tensor grid, drawn with ``imshow``. ``physics``: theta x z on
            the pellet-cladding sector, reshaped exactly as
            ``greedy.datasets.physics_dataset.reshape_contact_surface`` does, so these
            figures and the repository's own publication figures show the same surface.
``scatter`` unstructured nodes with 2-D coordinates, drawn with ``tripcolor``.
            ``membrane_2d``: contact nodes inside the obstacle disc.
``line``    genuinely 1-D contact, drawn against its **physical abscissa** rather than an
            index. ``hertz_2d`` along the contact arc; the 1-D obstacle toys.

Contact pressures span decades, so ``log=True`` selects the ``log10(1 + |v|)`` scaling the
publication pipeline uses -- on a linear scale the peak saturates and everything else
reads as zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldGeometry:
    kind: str                              # "grid" | "scatter" | "line"
    shape: tuple[int, int] | None = None   # (n_rows, n_cols) for "grid"
    coords: np.ndarray | None = None       # (n, 2) for "scatter", (n,) for "line"
    extent: tuple[float, float, float, float] | None = None   # x0, x1, y0, y1
    xlabel: str = "component index"
    ylabel: str = ""
    clabel: str = "multiplier"
    log: bool = False

    @property
    def is_field(self) -> bool:
        """Does this need a 2-D rendering rather than a curve?"""
        return self.kind in ("grid", "scatter")


def scale(values: np.ndarray, geom: FieldGeometry) -> np.ndarray:
    """Apply the geometry's colour scaling."""
    v = np.asarray(values, float)
    return np.log10(1.0 + np.abs(v)) if geom.log else v


def as_surface(values: np.ndarray, geom: FieldGeometry) -> np.ndarray:
    """Reshape a flat snapshot onto its grid.

    Row-major into ``(n_theta, n_z)``, matching
    ``greedy.datasets.physics_dataset.reshape_contact_surface``. Transposing here would
    silently produce a plausible-looking but wrong surface, so the convention is pinned
    by a test against that function.
    """
    v = np.asarray(values, float)
    if geom.shape is None:
        raise ValueError("geometry has no grid shape")
    expected = int(np.prod(geom.shape))
    if v.size != expected:
        raise ValueError(f"expected {expected} values for {geom.shape}, got {v.size}")
    return v.reshape(geom.shape)


def draw_field(ax, values: np.ndarray, geom: FieldGeometry, *,
               vmin=None, vmax=None, cmap="viridis"):
    """Render one snapshot (or error field) and return the mappable."""
    if geom.kind == "grid":
        surf = scale(as_surface(values, geom), geom)
        im = ax.imshow(surf, origin="lower", aspect="auto", extent=geom.extent,
                       vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    elif geom.kind == "scatter":
        xy = np.asarray(geom.coords, float)
        im = ax.tripcolor(xy[:, 0], xy[:, 1], scale(values, geom),
                          shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_aspect("equal")
    else:
        raise ValueError(f"{geom.kind} is not a field geometry")
    ax.set_xlabel(geom.xlabel, fontsize=8)
    ax.set_ylabel(geom.ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def field_limits(panels, geom: FieldGeometry, percentiles=(0.5, 99.7)):
    """Robust shared colour limits, as the publication pipeline uses.

    Percentile-clipped rather than min/max: a single saturated node would otherwise set
    the scale and flatten the whole field.
    """
    stacked = np.concatenate([np.asarray(scale(p, geom), float).ravel() for p in panels])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return None, None
    lo, hi = np.percentile(stacked, percentiles)
    if lo == hi:
        lo, hi = float(stacked.min()), float(stacked.max())
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# The geometries the merged datasets actually have
# ---------------------------------------------------------------------------

def physics_geometry() -> FieldGeometry:
    """Pellet-cladding quarter sector: 76 angular x 101 axial nodes.

    Constants come from ``greedy.datasets.physics_dataset`` rather than being restated,
    so this cannot drift from the repository's own figures.
    """
    from greedy.datasets.physics_dataset import (
        DEFAULT_GRID_SHAPE, HEIGHT_MM, SECTOR_ANGLE_RAD,
    )

    return FieldGeometry(
        kind="grid",
        shape=tuple(DEFAULT_GRID_SHAPE),
        extent=(0.0, float(HEIGHT_MM), 0.0, float(np.degrees(SECTOR_ANGLE_RAD))),
        xlabel="axial $z$ [mm]",
        ylabel=r"$\theta$ [deg]",
        clabel=r"$\log_{10}(1+|\lambda|)$",
        log=True,
    )


def line_geometry(coords=None, xlabel="component index") -> FieldGeometry:
    return FieldGeometry(kind="line", coords=coords, xlabel=xlabel)


def mirror_half_profile(values: np.ndarray) -> np.ndarray:
    """Reflect a stored half contact line into the full symmetric one.

    ``[v_m ... v_1, v_0, v_1 ... v_m]`` from ``[v_0 ... v_m]``, where ``v_0`` sits on the
    symmetry plane and is therefore written once, not twice.
    """
    v = np.asarray(values, float)
    if v.size < 2:
        return v
    return np.concatenate([v[:0:-1], v])


def mirrored_abscissa(n_half: int) -> np.ndarray:
    """Reference contact abscissas in ``[-1, 1]`` for a mirrored half profile."""
    return np.linspace(-1.0, 1.0, 2 * n_half - 1)


def mirrored_line_geometry(n_half: int,
                           xlabel="Reference contact abscissas") -> FieldGeometry:
    """Half-disk contact: stored as a half profile, drawn as the full symmetric one.

    [BEE20] §6.2 plots the normal contact stress against the **reference contact
    abscissa in [-1, 1]**, with the contact zone centred on 0 and zeros stacked on *both*
    sides (its Figures 7 and 8; §6.2 states "the effective contact zone is centered
    around the zero abscissa"). The half-disk is symmetric about that plane, so a solver
    stores only one half -- which is what ``FEM_SOLS`` contains: 57 nodes running from the
    symmetry axis outward, peak first, zeros last.

    Plotting the stored array directly therefore shows half the physics, with the peak
    jammed against the left edge. Reflecting it about node 0 recovers the profile the
    paper draws. Node 0 is written once, not twice -- it lies *on* the symmetry plane,
    which is the same fact that gives it a half-support shape function and makes
    ``lambda_0 / lambda_1 = 0.5035 +/- 0.019`` across all 50 snapshots.

    Display only. The mirror is a permutation-with-duplication applied identically to a
    snapshot and its reconstruction, so no metric sees it and no number changes.
    """
    return FieldGeometry(
        kind="mirrored_line",
        coords=mirrored_abscissa(n_half),
        xlabel=xlabel,
        ylabel="normal contact stress",
    )


def scatter_geometry(coords, xlabel="$x$", ylabel="$y$") -> FieldGeometry:
    return FieldGeometry(kind="scatter", coords=np.asarray(coords, float),
                         xlabel=xlabel, ylabel=ylabel)
