import numpy as np

from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.core.reduction_common import solve_cone_shift_projection


def _near_collinear_dataset(seed: int = 0, n: int = 15, d: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(size=d))
    snapshots = np.array(
        [base + 0.01 * i * np.abs(rng.normal(size=d)) for i in range(n)]
    )
    return np.clip(snapshots, 0.0, None)


def test_solve_cone_shift_projection_respects_upper_bound():
    rng = np.random.default_rng(1)
    system = np.abs(rng.normal(size=(25, 4)))
    target = np.abs(rng.normal(size=25)) * 2.0

    coeffs = solve_cone_shift_projection(system, target, upper_tol=1e-9)
    assert np.all(coeffs >= -1e-8)

    upsilon = system @ coeffs
    assert np.all(target - upsilon >= -1e-6)


def test_solve_cone_shift_projection_empty_system():
    target = np.array([1.0, 2.0, 3.0])
    coeffs = solve_cone_shift_projection(np.empty((3, 0)), target)
    assert coeffs.shape == (0,)


def test_mcpg_basis_is_nonnegative_and_converges():
    snapshots = _near_collinear_dataset()
    model = mCPG(snapshots=snapshots, epsilon=1e-6)
    model.compute_phases()

    assert model.basis_matrix is not None
    assert model.basis_matrix.shape[0] > 0
    assert np.all(model.basis_matrix >= -1e-6)

    # Residual history must be (non-strictly) decreasing, like CPG/ADG.
    residuals = np.asarray(model.residual_history)
    assert np.all(np.diff(residuals) <= 1e-9)
    assert residuals[-1] <= model.stopping_tolerance + 1e-6

    # Every selected row is distinct and every basis vector projects back
    # onto itself (up to numerical tolerance).
    assert len(set(model.selected_indices)) == len(model.selected_indices)
    for vector in model.basis_matrix:
        projected = model.project(vector)
        assert np.linalg.norm(projected - vector) < 1e-4 * max(1.0, np.linalg.norm(vector))


def test_mcpg_matches_cpg_accuracy_with_better_conditioning():
    snapshots = _near_collinear_dataset()
    epsilon = 1e-6

    cpg = CPG(snapshots=snapshots, epsilon=epsilon)
    cpg.compute_phases()
    mcpg = mCPG(snapshots=snapshots, epsilon=epsilon)
    mcpg.compute_phases()

    # Same tolerance should be reached with a comparable number of steps.
    assert abs(cpg.basis_matrix.shape[0] - mcpg.basis_matrix.shape[0]) <= 2

    def gram_condition(basis: np.ndarray) -> float:
        gram = basis @ basis.T
        return float(np.linalg.cond(gram))

    # This is the central claim of the mCPG paper (Section 4): removing the
    # already-representable part of each candidate before normalizing it
    # yields a markedly better-conditioned reduced basis than plain CPG.
    assert gram_condition(mcpg.basis_matrix) < gram_condition(cpg.basis_matrix)


def test_mcpg_reduces_to_cpg_first_generator():
    snapshots = _near_collinear_dataset(seed=2)
    model = mCPG(snapshots=snapshots, epsilon=1e-6)
    assert model._bootstrap()

    norms = np.linalg.norm(snapshots, axis=1)
    expected_first = snapshots[int(np.argmax(norms))]
    expected_first = expected_first / np.linalg.norm(expected_first)

    np.testing.assert_allclose(model._basis[0], expected_first, atol=1e-10)
