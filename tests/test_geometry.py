"""``bench.geometry``: how a snapshot lays out in space, and how it is drawn.

Every metric is basis-independent and ignores geometry entirely -- it only affects
figures. That is exactly why it needs tests: a wrong geometry produces a plausible
picture rather than an error. Drawn against a component index, ``physics``' 76x101
contact surface becomes 76 strips laid end to end, and ``fem_lambda``'s stored half
profile becomes half a contact line with the peak at the edge.

Also pinned here: that the field renderer applies no colour transformation of its own,
and that error panels are relative rather than absolute -- both changes that would
silently redraw every published figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import datasets as ds_mod
from bench.adapters import METHODS


def test_half_disk_is_drawn_as_the_full_symmetric_contact_line():
    """[BEE20] Fig. 7-8 plot the contact stress over abscissas in [-1, 1], centred on 0.

    FEM_SOLS stores only the half the symmetry plane makes redundant: 57 nodes from the
    symmetry axis outward, peak first, zeros last. Drawn directly that is half the physics
    with the peak jammed against the left edge. The mirror recovers the paper's picture.
    """
    from bench import geometry as g

    for key in ("fem_lambda", "fem_lambda_pressure"):
        d = ds_mod.load(key)
        geom = d.geometry
        assert geom is not None and geom.coords is not None, key
        assert geom.kind == "mirrored_line", key
        assert len(geom.coords) == 2 * d.dim - 1, "node 0 is on the plane: written once"
        assert geom.coords[0] == pytest.approx(-1.0)
        assert geom.coords[-1] == pytest.approx(1.0)

        full = g.mirror_half_profile(d.snapshots[:, 0])
        assert full.size == 2 * d.dim - 1
        assert np.allclose(full, full[::-1]), "must be symmetric about the plane"
        # Zeros on BOTH sides, contact in the middle -- the paper's layout.
        assert full[0] < 1e-6 * full.max() and full[-1] < 1e-6 * full.max()
        centre = full[full.size // 2]
        if key == "fem_lambda_pressure":
            # Corrected, the centre carries essentially the peak value.
            assert centre > 0.9 * full.max()
        else:
            # Uncorrected, the centre node carries half its tributary weight, so it sits
            # at roughly half the peak. That is the half-support effect, not a defect.
            assert 0.4 * full.max() < centre < 0.6 * full.max()


def test_pressure_correction_lifts_the_centre_onto_the_peak():
    """The half-support correction is what makes the mirrored profile peak at abscissa 0.

    Measured over all 50 snapshots, centre value divided by the profile peak:

        uncorrected  0.5034  (0.4668 - 0.5487)   -- exactly the half-support factor
        corrected    0.9885  (0.9336 - 1.0000)

    The corrected centre is *not* always the strict argmax (26 of 50) -- node-level noise
    of a few percent means node 1 can edge above node 0 -- so this asserts the ratio,
    which is the robust statement, rather than the argmax, which is not.
    """
    from bench import geometry as g

    raw = ds_mod.load("fem_lambda")
    press = ds_mod.load("fem_lambda_pressure")
    c = raw.dim - 1                           # index of node 0 in the mirrored array

    def ratios(d):
        return np.array([g.mirror_half_profile(d.snapshots[:, k])[c]
                         / g.mirror_half_profile(d.snapshots[:, k]).max()
                         for k in range(d.n_snapshots)])

    r_raw, r_press = ratios(raw), ratios(press)
    assert 0.45 < r_raw.mean() < 0.55, r_raw.mean()
    assert r_press.mean() > 0.95, r_press.mean()
    assert r_press.min() > 0.9, "corrected centre should never fall far below the peak"


def test_axial_profile_matches_the_publication_reduction():
    """The z-profile must be greedy.viz.publication's, not a reinvention.

    Its figures collapse theta by the MEAN and plot force against axial z. Pinning this
    against the repository's own profile_from_snapshot keeps the benchmark's overlay and
    the publication figures showing the same curve; a max-reduction or a transpose would
    both produce a plausible but different profile.
    """
    from greedy.viz.publication import profile_from_snapshot

    from bench import geometry as g

    ds = ds_mod.load("physics")
    geom = ds.geometry
    assert geom is not None and geom.shape is not None
    v = ds.snapshots[:, 0]

    mine = g.axial_profile(v, geom)
    assert np.allclose(mine, profile_from_snapshot(v, "mean"))
    assert mine.size == geom.shape[1] == 101, "one value per axial station"

    z = g.axial_coordinate(geom)
    assert z.size == mine.size
    assert z[0] == pytest.approx(0.0) and z[-1] == pytest.approx(5.0)


def test_active_span_uses_the_publication_threshold():
    """The shaded band is the same active span the publication figures mark."""
    from bench import geometry as g

    assert g.ACTIVE_THRESHOLD_RATIO == 1.0e-3
    ds = ds_mod.load("physics")
    assert ds.geometry is not None
    p = g.axial_profile(ds.snapshots[:, 0], ds.geometry)
    mask = g.active_span(p)
    assert mask.any() and not mask.all(), "contact should cover part of the cladding"
    assert np.all(p[mask] > g.ACTIVE_THRESHOLD_RATIO * p.max())


def test_error_panels_are_relative_not_absolute():
    """The error map is normalized by the snapshot's own peak.

    Absolute error cannot be compared across snapshots, and the best and worst panels are
    by construction *different* snapshots -- so an absolute colour scale shared between
    them is misleading. Normalizing by each snapshot's peak makes the two rows comparable
    and puts the colour bar in readable units.
    """
    from bench import geometry as g

    truth = np.array([0.0, 2.0, 8.0, 4.0])
    approx = np.array([0.0, 2.4, 8.0, 3.2])
    rel = g.relative_error_field(truth, approx)
    assert np.allclose(rel, np.abs(truth - approx) / 8.0)
    assert rel.max() <= 1.0
    # Scale-invariant: a snapshot ten times larger with ten times the error scores equally.
    assert np.allclose(rel, g.relative_error_field(10 * truth, 10 * approx))


def test_relative_error_survives_the_numerically_zero_inactive_zone():
    """Pointwise |d|/|theta| would divide by noise over most of the field.

    Where contact is not established the multiplier is numerically zero -- most entries
    below 1e-6 of the peak, reaching 5e-16 -- so a pointwise ratio explodes exactly where
    the ROM's spurious contact appears, drowning the real failure. The peak-normalized
    form stays finite and bounded.
    """
    from bench import geometry as g

    ds = ds_mod.load("physics")
    truth = ds.snapshots[:, 0]
    peak = np.abs(truth).max()
    tiny = np.abs(truth) < 1e-6 * peak
    assert tiny.sum() > truth.size // 2, "large numerically-zero region expected"
    assert np.abs(truth)[tiny].min() < 1e-12 * peak, "and it reaches the noise floor"

    # A real reconstruction, not a uniform rescaling: a ROM's error is additive and
    # lands in the inactive zone, which is the case that breaks the pointwise ratio.
    from bench.metrics.precision import reconstruct
    approx = reconstruct(truth[:, None], METHODS["adg"].fit(ds, R=6).generators)[:, 0]

    rel = g.relative_error_field(truth, approx)
    assert np.all(np.isfinite(rel)) and rel.max() <= 1.0
    pointwise = np.abs(truth - approx) / np.maximum(np.abs(truth), 1e-300)
    assert pointwise.max() / rel.max() > 1e3, "pointwise is the ill-conditioned one"


def test_field_rendering_applies_no_colour_transformation():
    """Fields are drawn in their own units, so the colour bar means what it says.

    ``geometry.scale`` used to apply log10(1+|v|) to the physics field, on the grounds
    that raw contact pressures span decades. Two problems. A colour bar reading
    log10(1+|lambda|) cannot be read against a force, and the compression makes the field
    look far more uniform than it is. And it did not match the reference it was meant to
    reproduce: greedy.viz.publication normalizes with a plain Normalize and rescales by a
    decimal exponent -- a change of UNITS, not of shape.
    """
    from bench import geometry as g

    geom = ds_mod.load("physics").geometry
    assert geom is not None and geom.shape is not None
    assert not hasattr(geom, "log"), "the flag should be gone, not merely set False"
    assert "log" not in geom.clabel.lower()
    for values in (np.array([0.0, 1e-9, 1.0, 5.0, 1234.0]),
                   np.linspace(0.0, 0.5, geom.shape[0] * geom.shape[1])):
        assert np.array_equal(g.scale(values, geom), values)
