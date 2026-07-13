import numpy as np

from greedy.core.angle_defect_greedy import AngularDefectGreedy


def _spread_dataset(seed: int = 0, n: int = 30, d: int = 40) -> np.ndarray:
    """Nonnegative snapshots pointing in a range of directions."""
    rng = np.random.default_rng(seed)
    snapshots = np.abs(rng.normal(size=(n, d)))
    snapshots[1] = 3.0 * snapshots[0]
    snapshots[2] = 0.5 * snapshots[0]
    return snapshots


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def test_angle_batch_none_matches_default_behavior():
    snapshots = _spread_dataset()
    base = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-3)
    base.compute_phases()
    explicit_none = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-3, angle_batch_tol=None
    )
    explicit_none.compute_phases()
    assert base.selected_indices == explicit_none.selected_indices


def test_separate_candidates_filters_maximum_angle_set():
    # Four nonnegative snapshot directions with known pairwise angles.
    d = 6
    e0 = np.zeros(d); e0[0] = 1.0
    e1 = np.zeros(d); e1[1] = 1.0
    dup = e0 + 0.01 * e1          # ~0.57 deg from e0 (near-duplicate maximizer)
    mid = e0 + e1                 # 45 deg from e0, 45 deg from e1
    snaps = np.array([e0, dup, mid, e1])

    # arcsin(0.5) = 30 deg separation floor.
    model = AngularDefectGreedy(snapshots=snaps, epsilon=1.0, angle_batch_tol=0.5)
    # Fabricated angle-to-cone values: e0 is the maximizer, then dup, mid, e1.
    angles = np.array([1.50, 1.49, 1.00, 0.90])

    kept = model._separate_candidates([0, 1, 2, 3], angles)

    assert kept[0] == 0                 # the maximizer is always admitted first
    assert 1 not in kept                # near-duplicate of maximizer is dropped
    assert set(kept) == {0, 2, 3}       # 45-deg-separated directions survive
    # Every admitted pair is separated by more than arcsin(0.5) = 30 deg.
    min_sep = np.arcsin(0.5)
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            assert _angle(snaps[kept[i]], snaps[kept[j]]) > min_sep - 1e-9


def test_separate_candidates_keeps_all_when_already_separated():
    d = 5
    a = np.array([1.0, 0.0, 0, 0, 0])
    b = np.array([0.0, 1.0, 0, 0, 0])
    c = np.array([0.0, 0.0, 1, 0, 0])   # mutually orthogonal (90 deg apart)
    model = AngularDefectGreedy(snapshots=np.array([a, b, c]), epsilon=1.0,
                                angle_batch_tol=0.5)
    kept = model._separate_candidates([0, 1, 2], np.array([1.0, 0.9, 0.8]))
    assert set(kept) == {0, 1, 2}


def test_angle_batch_single_candidate_is_noop():
    model = AngularDefectGreedy(snapshots=_spread_dataset(), epsilon=1e-3,
                                angle_batch_tol=0.3)
    assert model._separate_candidates([7], np.arange(30.0)) == [7]
    assert model._separate_candidates([], np.arange(30.0)) == []


def test_angle_batch_admits_maximizer_and_certificate_holds():
    model = AngularDefectGreedy(snapshots=_spread_dataset(), epsilon=1e-3,
                                angle_batch_tol=0.2)
    model.compute_phases()
    assert model.basis_matrix is not None and model.basis_matrix.shape[0] >= 2
    assert all(size >= 1 for size in model.batch_size_history)
    assert model.verify_angular_defect_certificate()["violations"] == 0


def test_angle_batch_band_engages_on_generic_data():
    # On floating-point data with directional spread the exact max-angle tie is
    # a singleton; the band must widen it so batches of >1 actually form.
    rng = np.random.default_rng(5)
    snapshots = np.abs(rng.normal(size=(40, 50)))  # generic, no colinear dupes
    plain = AngularDefectGreedy(snapshots=snapshots, epsilon=1e-3)
    plain.compute_phases()
    assert max(plain.batch_size_history) == 1  # base algorithm: one at a time

    batched = AngularDefectGreedy(
        snapshots=snapshots, epsilon=1e-3, angle_batch_tol=0.2
    )
    batched.compute_phases()
    assert max(batched.batch_size_history) > 1          # band engaged
    assert len(batched.batch_size_history) < len(plain.batch_size_history)  # fewer rounds
    # No accuracy penalty: same worst-case relative residual to the final cone.
    def rho_max(model):
        res = model.final_residuals()
        nn = np.linalg.norm(model.snapshots, axis=1)
        return np.divide(res, nn, out=np.zeros_like(res), where=nn > 0).max()
    assert rho_max(batched) <= rho_max(plain) + 1e-9


def test_angle_batch_tol_out_of_range_rejected():
    snapshots = _spread_dataset()
    for bad in (-0.1, 1.5):
        try:
            AngularDefectGreedy(snapshots=snapshots, epsilon=1e-3, angle_batch_tol=bad)
        except ValueError:
            continue
        raise AssertionError(f"angle_batch_tol={bad} should have been rejected")
