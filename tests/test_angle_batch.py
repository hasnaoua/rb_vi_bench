import numpy as np

from greedy.core.angle_defect_greedy import AngularDefectGreedy


def _spread_dataset(seed: int = 0, n: int = 30, d: int = 40) -> np.ndarray:
    """Nonnegative snapshots pointing in a range of directions, with a few
    near-parallel duplicates the redundancy filter should trim."""
    rng = np.random.default_rng(seed)
    snapshots = np.abs(rng.normal(size=(n, d)))
    snapshots[1] = 3.0 * snapshots[0]
    snapshots[2] = 0.5 * snapshots[0]
    return snapshots


def _rho_max(model: AngularDefectGreedy) -> float:
    res = model.final_residuals()
    nn = np.linalg.norm(model.snapshots, axis=1)
    return float(np.divide(res, nn, out=np.zeros_like(res), where=nn > 0).max())


def _basis_min_pairwise_angle(basis: np.ndarray) -> float:
    units = basis / np.linalg.norm(basis, axis=1, keepdims=True)
    ang = np.arccos(np.clip(units @ units.T, -1.0, 1.0))
    np.fill_diagonal(ang, np.inf)
    return float(np.min(ang))


def test_redundancy_none_matches_default_behavior():
    snapshots = _spread_dataset()
    base = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-3)
    base.compute_phases()
    explicit_none = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-3, angle_redundancy_tol=None
    )
    explicit_none.compute_phases()
    assert base.selected_indices == explicit_none.selected_indices


def test_redundancy_filter_diminishes_R():
    snapshots = _spread_dataset(seed=2, n=40, d=50)
    plain = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-2)
    plain.compute_phases()
    filtered = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-2, angle_redundancy_tol=0.2
    )
    filtered.compute_phases()
    # The filter can only keep or reduce the number of generators.
    assert filtered.basis_matrix.shape[0] <= plain.basis_matrix.shape[0]
    # Larger tolerance is monotone: never more generators than a smaller one.
    tighter = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-2, angle_redundancy_tol=0.05
    )
    tighter.compute_phases()
    assert filtered.basis_matrix.shape[0] <= tighter.basis_matrix.shape[0]


def test_redundancy_filter_enforces_pairwise_separation():
    snapshots = _spread_dataset(seed=3, n=40, d=50)
    eps_angle = 0.2
    model = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-3, angle_redundancy_tol=eps_angle
    )
    model.compute_phases()
    # No two stored generators are closer than arcsin(eps_angle) in angle
    # (initialization picks the widest-angle pair, so this holds basis-wide).
    assert _basis_min_pairwise_angle(model.basis_matrix) > np.arcsin(eps_angle) - 1e-9


def test_redundancy_filter_trims_generators_and_improves_conditioning():
    # Smooth overlapping Gaussian bumps: plain ADG accumulates many near-parallel
    # generators (ill-conditioned); the redundancy filter trims them.
    x = np.linspace(0.0, 1.0, 120)
    centers = np.linspace(0.15, 0.85, 12)
    rng = np.random.default_rng(0)
    coeffs = rng.uniform(size=(50, centers.size))
    basis = np.array([np.exp(-((x - c) ** 2) / (2 * 0.07 ** 2)) for c in centers])
    snaps = np.clip(coeffs @ basis, 0.0, None)

    plain = AngularDefectGreedy(snapshots=snaps, epsilon=1e-2)
    plain.compute_phases()
    filtered = AngularDefectGreedy(snapshots=snaps, epsilon=1e-2, angle_redundancy_tol=0.15)
    filtered.compute_phases()

    assert filtered.basis_matrix.shape[0] < plain.basis_matrix.shape[0]
    assert np.linalg.cond(filtered.basis_matrix) < np.linalg.cond(plain.basis_matrix)


def test_redundancy_filter_composes_with_normalization():
    snapshots = _smooth_gaussian_dataset(seed=4)
    plain = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-2)
    plain.compute_phases()
    a = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-2, angle_redundancy_tol=0.2)
    a.compute_phases()
    b = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-2, angle_redundancy_tol=0.2,
        normalize_snapshots=True,
    )
    b.compute_phases()
    # The redundancy filter diminishes R with or without normalization, and every
    # stored generator is pairwise separated by more than arcsin(0.2).
    assert a.basis_matrix.shape[0] < plain.basis_matrix.shape[0]
    assert b.basis_matrix.shape[0] < plain.basis_matrix.shape[0]
    assert _basis_min_pairwise_angle(b.basis_matrix) > np.arcsin(0.2) - 1e-9


def _smooth_gaussian_dataset(n: int = 50, d: int = 120, seed: int = 0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, d)
    centers = np.linspace(0.15, 0.85, 12)
    rng = np.random.default_rng(seed)
    coeffs = rng.uniform(size=(n, centers.size))
    basis = np.array([np.exp(-((x - c) ** 2) / (2 * 0.07 ** 2)) for c in centers])
    return np.clip(coeffs @ basis, 0.0, None)


def test_batch_knob_reduces_rounds_without_growing_R():
    snaps = _smooth_gaussian_dataset()
    plain = AngularDefectGreedy(snapshots=snaps, epsilon=1e-2)
    plain.compute_phases()
    batched = AngularDefectGreedy(snapshots=snaps, epsilon=1e-2, angle_batch_tol=0.2)
    batched.compute_phases()
    # Fewer greedy rounds, and R is not materially larger than plain.
    assert len(batched.batch_size_history) < len(plain.batch_size_history)
    assert max(batched.batch_size_history) > 1
    assert batched.basis_matrix.shape[0] <= plain.basis_matrix.shape[0] + 1
    # Accuracy is not worse.
    assert _rho_max(batched) <= _rho_max(plain) + 1e-9


def test_batch_and_redundancy_compose():
    snaps = _smooth_gaussian_dataset(seed=1)
    plain = AngularDefectGreedy(snapshots=snaps, epsilon=1e-2)
    plain.compute_phases()
    both = AngularDefectGreedy(
        snapshots=snaps, epsilon=1e-2, angle_redundancy_tol=0.15, angle_batch_tol=0.2
    )
    both.compute_phases()
    # Redundancy still diminishes R; batching still cuts rounds.
    assert both.basis_matrix.shape[0] <= plain.basis_matrix.shape[0]
    assert len(both.batch_size_history) < len(plain.batch_size_history)


def test_batch_tol_out_of_range_rejected():
    snapshots = _spread_dataset()
    for bad in (-0.1, 1.5):
        try:
            AngularDefectGreedy(snapshots=snapshots, epsilon=1e-3, angle_batch_tol=bad)
        except ValueError:
            continue
        raise AssertionError(f"angle_batch_tol={bad} should have been rejected")


def test_redundancy_tol_out_of_range_rejected():
    snapshots = _spread_dataset()
    for bad in (-0.1, 1.5):
        try:
            AngularDefectGreedy(
                snapshots=snapshots, epsilon=1e-3, angle_redundancy_tol=bad
            )
        except ValueError:
            continue
        raise AssertionError(f"angle_redundancy_tol={bad} should have been rejected")
