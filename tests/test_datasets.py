"""The dataset registry: what each source can honestly supply, and what it must refuse.

``Dataset`` normalizes eight genuinely different sources into one contract, and the
places it can go wrong are all silent: an index array left pointing at a dropped column,
a train/test split that does not partition, a source that reports 0.0 where it means
"not measured". Each is pinned here.

The heaviest cases are the two with real provenance to preserve -- ``physics``, whose
parameters and 50/49 split come from the pellet-cladding archive, and ``fem_lambda``,
whose ordering and half-support correction are established in ``FEM_LAMBDA_ORDERING``
rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import _paths
from bench import datasets as ds_mod
from bench import metrics
from bench.adapters import METHODS
from bench.types import Dataset


def test_dataset_rejects_negative_snapshots():
    """A negative multiplier means a broken HF solve, not a dataset to benchmark on."""
    S = np.abs(np.random.default_rng(0).random((10, 5)))
    S[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        Dataset(name="bad", snapshots=S)


def test_dataset_clips_roundoff_negatives():
    """Solver noise at 1e-18 must not be treated as a sign error.

    ``fem_lambda`` and ``physics`` both arrive with entries at ~1e-12 and ~1e-70.
    """
    S = np.abs(np.random.default_rng(0).random((10, 5)))
    S[0, 0] = -1e-16
    ds = Dataset(name="ok", snapshots=S)
    assert ds.snapshots.min() >= 0.0


def test_no_split_reports_nan_not_zero(bumps):
    """A missing test set must not read as a perfect generalization score."""
    ds = Dataset(name="nosplit", snapshots=bumps.snapshots)
    assert ds.test() is None
    row = metrics.precision.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=5))
    assert np.isnan(row["test_max_rel_err"])


def test_zero_snapshots_are_dropped_before_any_algorithm_runs(bumps):
    """A numerically zero snapshot is absence of data and must never reach a method.

    It is a parameter value at which no contact occurred, so lambda = 0 everywhere.
    Normalizing it is undefined, any per-snapshot relative criterion sees an arbitrary
    error on it, and the ADG spec excludes it outright (S subset R_+^m \\ {0}). Dropping
    it at construction is what keeps every downstream method from having to special-case
    it. physics carries five such columns -- the low end of its displacement sweep,
    before the imposed displacement has closed the initial pellet-cladding gap.
    """
    S = np.column_stack([bumps.train(), np.zeros(bumps.dim)])
    ds = Dataset(name="withzero", snapshots=S)
    assert ds.n_dropped_zero == 1
    assert ds.n_snapshots == bumps.train().shape[1]
    assert np.linalg.norm(ds.snapshots, axis=0).min() > 0

    physics = ds_mod.load("physics")
    assert physics.n_dropped_zero == 5, "physics' five zero snapshots should be gone"
    assert physics.n_snapshots == 94
    assert physics.name == "3D Pellet-Cladding"


def test_physics_reshape_matches_the_repository_convention():
    """The theta x z reshape must be greedy_algos', not a plausible-looking transpose.

    physics snapshots are 7676 = 76 x 101 nodes on a quarter-cylinder contact surface,
    not a sequence. Transposing would still produce a 2-D image, and a wrong one -- so
    this is pinned against the function the repository's own publication figures use.
    """
    from greedy.datasets.physics_dataset import reshape_contact_surface

    from bench import geometry

    ds = ds_mod.load("physics")
    geom = ds.geometry
    assert geom is not None
    assert geom.kind == "grid" and geom.shape == (76, 101)

    v = ds.snapshots[:, 0]
    assert np.array_equal(geometry.as_surface(v, geom), reshape_contact_surface(v))

    # extent is (z_min, z_max, theta_min, theta_max) in mm and degrees
    assert geom.extent == (0.0, 5.0, 0.0, 90.0)

    with pytest.raises(ValueError, match="expected"):
        geometry.as_surface(v[:-1], geom)


def test_field_datasets_declare_their_geometry():
    """A contact set that tiles a surface must never fall back to an index plot."""
    pytest.importorskip("cvxopt", reason="membrane_2d needs greedy_algos[qp]")
    for key, kind in (("physics", "grid"), ("membrane_2d", "scatter"),
                      ("fem_lambda", "mirrored_line")):
        ds = ds_mod.load(key)
        assert ds.geometry is not None, key
        assert ds.geometry.kind == kind, (key, ds.geometry.kind)
    # The half-disk is genuinely 1-D but symmetric about its plane: it carries a physical
    # abscissa spanning [-1, 1], not a node index.
    hd = ds_mod.load("fem_lambda")
    assert hd.geometry is not None and hd.geometry.coords is not None
    assert len(hd.geometry.coords) == 2 * hd.dim - 1


def test_physics_uses_the_shipped_train_test_split():
    """``physics`` fits on the archive's training columns and scores on its test ones.

    It was the last split-less source in the merge: ``greedy_algos``' pipeline built and
    evaluated on the whole dataset, so every ``test_*`` column came out nan and the
    figures fell back to training error. The split archive ships a 50/49 partition of the
    99 parameters, and the five no-contact columns ``Dataset`` drops leave 47/47.
    """
    ds = ds_mod.load("physics")

    assert ds.has_split
    test_cols = ds.test()
    assert test_cols is not None
    assert ds.train().shape[1] == 47
    assert test_cols.shape[1] == 47
    assert ds.train().shape[1] + test_cols.shape[1] == ds.n_snapshots

    assert ds.train_idx is not None and ds.test_idx is not None
    train, test = set(map(int, ds.train_idx)), set(map(int, ds.test_idx))
    assert not (train & test), "a snapshot must not be both fitted and scored"
    assert train | test == set(range(ds.n_snapshots)), "the split must cover every column"

    # The archive interleaves rather than holding out a tail, and it has to: the
    # parameter sweeps monotonically from no contact to full contact, so a contiguous
    # test block would be extrapolation. Both halves must span the same range.
    assert ds.params is not None
    p_train = ds.params[ds.train_idx].ravel()
    p_test = ds.params[ds.test_idx].ravel()
    assert abs(p_train.min() - p_test.min()) < 0.02
    assert abs(p_train.max() - p_test.max()) < 0.02

    # And the metric must now report a real held-out number rather than nan.
    row = metrics.precision.evaluate(ds, METHODS["cpg_bee20"].fit(ds, R=8))
    assert not np.isnan(row["test_max_rel_err"])
    assert not np.isnan(row["test_max_rel_err_persnap"])


def test_physics_split_does_not_move_the_tolerance_denominator():
    """``Dataset.scale`` is unchanged by introducing the split, and that is checkable.

    Both relative tolerances normalize by ``max_q ||theta_q||`` over the snapshots they
    are *fitted* to, so a split can silently rescale every tolerance-mode result. Here it
    does not: the largest-norm snapshot is the last column of the sweep, which the archive
    assigns to training. That does not make pre-split ``physics`` cells comparable one for
    one -- the training set halved, so the fits moved -- but it does mean a ``delta``
    still denotes the same absolute error it did before.
    """
    ds = ds_mod.load("physics")
    full_scale = float(np.max(np.linalg.norm(ds.snapshots, axis=0)))
    assert np.isclose(ds.scale, full_scale, rtol=1e-12), (
        "the max-norm snapshot must be in the training half, or every delta shifts")


def test_physics_parameters_come_from_the_archive_not_an_inferred_grid():
    """The 99 shipped parameters supersede the 96-point grid greedy_algos guessed.

    ``greedy.datasets.physics_dataset`` hard-codes a 96-value displacement grid from the
    problem statement and reconciles it with 99 columns by dropping three. The archive
    shows that was off by one grid step throughout -- its first kept column is 0.175 mm,
    not the 0.18 it was labelled. The snapshots were never wrong; their abscissa was.
    """
    from greedy.datasets.physics_dataset import displacement_grid

    shipped = ds_mod._load_cladding_split()["params"]
    ds = ds_mod.load("physics")

    # Two blocks, as shipped: 0.16-0.30 by 0.005 (29 points) then 0.31-1.00 by 0.01 (70).
    assert shipped.size == 99
    steps = np.unique(np.round(np.diff(shipped), 12))
    assert set(steps) == {0.005, 0.01}, steps
    assert np.isclose(shipped[0], 0.16) and np.isclose(shipped[-1], 1.00)

    # The old builder kept columns 3.. and labelled them with its 96-point grid. Line the
    # two up on the columns they share: every label is high by exactly one grid step --
    # not a constant offset, since the step itself changes between the blocks.
    inferred = displacement_grid()
    assert inferred.size == 96
    offset = np.round(inferred - shipped[3:], 12)
    assert set(np.unique(offset)) <= {0.005, 0.01}, offset
    assert (offset > 0).all(), "the inferred labels were high, uniformly and by one step"

    # Five columns come off the low end, so the dataset starts inside the fine block.
    assert ds.n_snapshots == 94
    assert ds.params is not None
    assert np.isclose(ds.params.ravel()[0], 0.185)
    assert np.isclose(ds.params.ravel()[-1], 1.00), "the sweep ends at 1.00, not 1.01"


def test_physics_archive_split_files_agree_with_the_full_matrix(monkeypatch):
    """A split that disagrees with the matrix it indexes must fail loudly.

    The archive ships the train and test blocks as separate files as well as the indices
    that produce them. If those ever disagreed, the benchmark would fit on one set of
    columns and score on another with no symptom anywhere downstream -- a plausible-looking
    test error that means nothing. The loader verifies rather than trusts.
    """
    archive = ds_mod._load_cladding_split()
    raw, params = archive["snapshots"], archive["params"]
    train_idx, test_idx = archive["train_idx"], archive["test_idx"]

    assert raw.shape == (7676, 99)
    assert params.size == 99
    assert train_idx.size == 50 and test_idx.size == 49
    assert np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(99))

    # The same 7676 x 99 array greedy_algos ships, only with its parameters attached.
    legacy = _paths.GREEDY_ROOT / "data" / "physics_data.txt"
    if legacy.is_file():
        assert np.array_equal(np.loadtxt(legacy), raw), (
            "the split archive must be the same matrix, not a different extraction")

    # A missing archive is an error with a fix in it, not an ImportError further down.
    monkeypatch.setattr(_paths, "CLADDING_SPLIT", _paths.ROOT / "no_such_archive")
    with pytest.raises(FileNotFoundError, match="RB_VI_CLADDING_SPLIT"):
        ds_mod._load_cladding_split()


def test_coordinate_rescaling_can_change_the_reduction():
    """Doubling one coordinate is not a no-op for a cone method.

    Rescaling a *coordinate* is a different operation from rescaling a *snapshot*, and
    the methods are sensitive to it. (They are sensitive to snapshot scaling too, which
    is easy to get backwards: span_+{c g} = span_+{g} makes the resulting CONE invariant,
    but the greedy SELECTION is an argmax over residual magnitudes, so scaling a column
    changes which snapshot wins. Only the cone is invariant, never the selection.)

    The rescaled view is built in the test rather than loaded. It used to be a shipped
    dataset -- the half-disk in pressure rather than force variables, which differ by the
    tributary length of the symmetry node, so exactly one coordinate is doubled. That
    dataset is gone, but the property is a statement about the METHODS and does not need
    it: doubling row 0 here reproduces the same transformation.
    """
    import dataclasses

    force = ds_mod.load("fem_lambda")
    S = force.snapshots.copy()
    S[0, :] *= 2.0                       # one coordinate, as the pressure view did
    rescaled = dataclasses.replace(force, name="fem_lambda[coord0 x2]", snapshots=S,
                                   geometry=None)

    differed = []
    for key in ("cpg_bee20", "mcpg_ndee22", "adg"):
        a = METHODS[key].fit(force, delta=0.02)
        b = METHODS[key].fit(rescaled, delta=0.02)
        if a.selected_indices != b.selected_indices:
            differed.append(key)
    assert differed, "no method saw a coordinate rescaling; that would be a bug"


@pytest.mark.parametrize("key", ds_mod.HEAVY)
def test_heavy_datasets_load_when_cvxopt_is_present(key):
    """The 2-D sources are advertised, so they must build -- given their dependency.

    Both import ``cvxopt`` at module scope. Skipping when it is absent keeps the suite
    green on a bare install, but a *different* failure must still fail the test rather
    than be swallowed as "optional".
    """
    pytest.importorskip("cvxopt", reason="heavy-tier datasets need greedy_algos[qp]")
    ds = ds_mod.load(key)
    assert ds.snapshots.min() >= 0.0
    assert ds.dim > 1 and ds.n_snapshots > 1
    assert np.all(np.isfinite(ds.snapshots))


@pytest.mark.parametrize("key", ds_mod.FAST)
def test_fast_datasets_load_and_are_valid(key):
    """Every advertised fast dataset must build and satisfy the Dataset contract."""
    ds = ds_mod.load(key)
    assert ds.snapshots.min() >= 0.0
    assert ds.dim > 0 and ds.n_snapshots > 1
    assert np.all(np.isfinite(ds.snapshots))
    assert ds.scale > 0.0
    if ds.supports_infsup:
        assert ds.A is not None and ds.B_of_mu is not None
        assert ds.B_of_mu(0).ndim == 2


def test_datasets_are_named_for_the_problem_they_solve():
    """Reported names say what the physics is; CLI keys stay stable.

    A reader of a figure should see the problem, not the file it came from. The registry
    key is the handle for --datasets and must not move, or every saved command breaks.
    """
    from bench import layout

    expected = {
        "fem_lambda": "Half-disks of Hertz",
        "physics": "3D Pellet-Cladding",
    }
    for key, name in expected.items():
        assert key in ds_mod.DATASETS, f"{key} key must stay stable"
        assert ds_mod.load(key).name == name
        # And the prose name must still produce a usable directory.
        slug = layout.slug(name)
        assert " " not in slug and "(" not in slug and "__" not in slug, slug
        assert slug
