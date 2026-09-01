import numpy as np
import pytest
import scipy.linalg as la

from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.core.reduction_common import (
    gram_cholesky,
    solve_cone_shift_projection,
    w_norm,
)
from greedy.datasets.contact_forces import load_contact_force_dataset


def random_spd(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(d, d))
    return root @ root.T + d * np.eye(d)


def test_gram_cholesky_reproduces_w_norm():
    gram = random_spd(6, seed=0)
    vectors = np.abs(np.random.default_rng(1).normal(size=(5, 6)))
    upper = gram_cholesky(gram)

    np.testing.assert_allclose(upper.T @ upper, gram, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        np.linalg.norm(vectors @ upper.T, axis=1),
        w_norm(vectors, gram),
        rtol=1e-8,
    )


def test_cone_membership_is_preserved_by_the_transform():
    """The property the whole W-integration leans on: U(A c) == (U A) c, c >= 0."""
    gram = random_spd(5, seed=2)
    upper = gram_cholesky(gram)
    generators = np.abs(np.random.default_rng(3).normal(size=(5, 3)))
    coeffs = np.abs(np.random.default_rng(4).normal(size=3))

    np.testing.assert_allclose(
        upper @ (generators @ coeffs),
        (upper @ generators) @ coeffs,
        rtol=1e-10,
    )


def test_cone_shift_projection_defaults_are_unchanged():
    system = np.abs(np.random.default_rng(5).normal(size=(7, 3)))
    vector = np.abs(np.random.default_rng(6).normal(size=7)) + 1.0

    baseline = solve_cone_shift_projection(system, vector)
    explicit = solve_cone_shift_projection(
        system, vector, cone_system=system, cone_vector=vector
    )
    np.testing.assert_allclose(baseline, explicit, rtol=1e-8, atol=1e-10)


def test_cone_shift_projection_respects_the_supplied_constraint():
    """The upper bound must bind on cone_vector, not on vector."""
    system = np.array([[1.0], [1.0]])
    vector = np.array([10.0, 10.0])
    cone_vector = np.array([1.0, 1.0])

    coeffs = solve_cone_shift_projection(
        system, vector, cone_system=system, cone_vector=cone_vector
    )
    assert np.all(system @ coeffs <= cone_vector + 1e-6)


def test_mcpg_identity_constraint_transform_matches_default():
    snapshots = np.abs(np.random.default_rng(7).normal(size=(8, 6)))

    default = mCPG(snapshots=snapshots, epsilon=1e-2)
    default.compute_phases()
    assert default.basis_matrix is not None
    identity = mCPG(
        snapshots=snapshots, epsilon=1e-2, constraint_transform=np.eye(6)
    )
    identity.compute_phases()
    assert identity.basis_matrix is not None

    assert default.selected_indices == identity.selected_indices
    np.testing.assert_allclose(default.basis_matrix, identity.basis_matrix, atol=1e-8)


def test_mcpg_constraint_transform_validates_shape():
    snapshots = np.abs(np.random.default_rng(8).normal(size=(4, 6)))
    with pytest.raises(ValueError, match="constraint_transform columns"):
        mCPG(snapshots=snapshots, epsilon=1e-2, constraint_transform=np.eye(3))


def test_mcpg_in_w_keeps_the_shift_inside_the_physical_cone():
    """
    The bug this guards: running mCPG on U-transformed snapshots without a
    constraint_transform enforces U(theta - Upsilon) >= 0 instead of
    theta - Upsilon >= 0, so the shift leaves the physical cone W+.
    """
    gram = random_spd(6, seed=9)
    upper = gram_cholesky(gram)
    inverse = la.solve_triangular(upper, np.eye(6), lower=False)
    snapshots = np.abs(np.random.default_rng(10).normal(size=(7, 6))) + 0.1
    transformed = snapshots @ upper.T

    model = mCPG(snapshots=transformed, epsilon=1e-3, constraint_transform=inverse)
    model.compute_phases()
    assert model.basis_matrix is not None

    # Every generator, mapped back to physical coordinates, is a normalized
    # (theta - Upsilon) and so must be elementwise nonnegative.
    physical = la.solve_triangular(upper, model.basis_matrix.T, lower=False).T
    assert physical.shape[0] > 1
    assert physical.min() > -1e-6


def test_cpg_on_transformed_data_selects_by_w_norm():
    """CPG's picks on U-transformed data must match a direct W-norm argmax."""
    gram = random_spd(5, seed=11)
    upper = gram_cholesky(gram)
    snapshots = np.abs(np.random.default_rng(12).normal(size=(6, 5)))

    model = CPG(snapshots=snapshots @ upper.T, epsilon=0.5)
    model.compute_phases()
    assert model.basis_matrix is not None

    # The first pick maximizes the distance to the zero cone, i.e. the W-norm.
    assert model.selected_indices[0] == int(np.argmax(w_norm(snapshots, gram)))


@pytest.mark.parametrize("case", ["membrane", "hertz", "hertz_a", "hertz_b"])
def test_datasets_load_in_the_cone_with_an_spd_gram(case):
    dataset = load_contact_force_dataset(case)

    assert dataset.snapshots.shape[0] == dataset.parameters.size
    assert dataset.snapshots.shape[1] == dataset.gram.shape[0]
    assert dataset.snapshots.min() >= -1e-6
    # An SPD Gram is what makes the Cholesky route valid at all.
    assert np.all(la.eigvalsh(dataset.gram) > 0.0)


def test_arc_mass_matrix_integrates_p1_fields_exactly():
    """
    The Gram rebuilt for hertz_a / hertz_b must be the real L2 mass matrix of
    the arc, not a nodal stand-in: 1' M 1 is the arc length, and x' M x the
    exact integral of x^2 over it. If this drifts, every W-norm on those two
    cases is quietly measuring the wrong thing.
    """
    from greedy.datasets.contact_forces import arc_mass_matrix

    x = np.linspace(-1.0, 1.0, 51)
    mass = arc_mass_matrix(x)

    ones = np.ones_like(x)
    assert ones @ mass @ ones == pytest.approx(2.0)          # length of [-1, 1]
    # P1 mass is exact on linears; x^2 carries the standard O(h^2) quadrature error.
    assert x @ mass @ x == pytest.approx(2.0 / 3.0, rel=1e-3)
    np.testing.assert_allclose(mass, mass.T, rtol=1e-12)
    assert np.all(la.eigvalsh(mass) > 0.0)


def test_arc_mass_matrix_rejects_unordered_nodes():
    from greedy.datasets.contact_forces import arc_mass_matrix

    with pytest.raises(ValueError, match="strictly increasing"):
        arc_mass_matrix(np.array([0.0, 0.2, 0.1]))


@pytest.mark.parametrize(
    "case, filename",
    [
        ("hertz_a", "hertz_case_a_imposed_displacement.npz"),
        ("hertz_b", "hertz_case_b_parametric_geometry.npz"),
    ],
)
def test_row_wise_cases_are_loaded_without_a_spurious_transpose(case, filename):
    """
    hertz_a / hertz_b store snapshots row-wise (P x n_c) while membrane / hertz
    store them column-wise. Both n_c=51 and a transpose would not be caught by a
    shape check alone, so pin the actual values against the raw file.
    """
    raw = np.load(f"data/contact_forces/{filename}", allow_pickle=True)
    dataset = load_contact_force_dataset(case)

    np.testing.assert_allclose(dataset.snapshots, raw["snapshots"], rtol=1e-12)
    np.testing.assert_allclose(dataset.parameters, raw["params"], rtol=1e-12)
    # Row p must be one lambda(mu_p) over the arc, so it lines up with abscissas.
    assert dataset.coordinates is not None
    assert dataset.snapshots.shape[1] == dataset.coordinates.size
    # The rebuilt Gram is the arc mass of exactly those abscissas.
    from greedy.datasets.contact_forces import arc_mass_matrix

    np.testing.assert_allclose(dataset.gram, arc_mass_matrix(raw["abscissas"]), rtol=1e-12)


@pytest.mark.parametrize("case", ["hertz_a", "hertz_b"])
def test_mcpg_is_better_conditioned_at_matched_r_on_the_new_hertz_cases(case):
    """
    Both new cases are low-rank enough that all three methods reach the same
    accuracy, so conditioning is the only axis that separates them -- and it is
    the axis mCPG exists to fix.
    """
    from greedy.pipelines.contact_forces_compare import prefix_curve
    from greedy.core.reduction_common import parameter_train_test_split

    dataset = load_contact_force_dataset(case)
    upper = gram_cholesky(dataset.gram)
    inverse = la.solve_triangular(upper, np.eye(upper.shape[0]), lower=False)
    transformed = dataset.snapshots @ upper.T
    train, test = parameter_train_test_split(dataset.parameters, test_fraction=0.2)

    cpg = CPG(snapshots=transformed[train], epsilon=1e-3)
    cpg.compute_phases()
    assert cpg.basis_matrix is not None
    mcpg = mCPG(snapshots=transformed[train], epsilon=1e-3, constraint_transform=inverse)
    mcpg.compute_phases()
    assert mcpg.basis_matrix is not None

    budget = min(cpg.basis_matrix.shape[0], mcpg.basis_matrix.shape[0])
    cpg_at = prefix_curve(cpg.basis_matrix[:budget], transformed, train, test)[-1]
    mcpg_at = prefix_curve(mcpg.basis_matrix[:budget], transformed, train, test)[-1]

    assert mcpg_at["kappa"] < cpg_at["kappa"]
    assert mcpg_at["train_max"] <= cpg_at["train_max"] * 1.01


@pytest.mark.parametrize("case", ["hertz_a", "hertz_b"])
def test_mcpg_generators_stay_in_the_physical_cone_on_the_new_hertz_cases(case):
    """The W-transform guard from test_mcpg_in_w_..., on the real datasets."""
    dataset = load_contact_force_dataset(case)
    upper = gram_cholesky(dataset.gram)
    inverse = la.solve_triangular(upper, np.eye(upper.shape[0]), lower=False)

    model = mCPG(
        snapshots=dataset.snapshots @ upper.T,
        epsilon=1e-3,
        constraint_transform=inverse,
    )
    model.compute_phases()
    assert model.basis_matrix is not None

    physical = la.solve_triangular(upper, model.basis_matrix.T, lower=False).T
    assert physical.min() > -1e-6


def test_tensor_grid_interior_split_holds_out_the_interior():
    from greedy.core.reduction_common import tensor_grid_interior_split

    dataset = load_contact_force_dataset("membrane")
    train, test = tensor_grid_interior_split(dataset.mu_samples)

    # 5x5x5 grid -> 3x3x3 strictly interior points held out.
    assert test.size == 27
    assert train.size == 98
    assert not set(train) & set(test)

    # Every extreme of every parameter direction must stay in training, so the
    # test set is interpolation rather than extrapolation.
    for column in range(dataset.mu_samples.shape[1]):
        values = dataset.mu_samples[:, column]
        for extreme in (values.min(), values.max()):
            assert not np.any(np.isclose(dataset.mu_samples[test, column], extreme))


def test_tensor_grid_split_rejects_scalar_parameters():
    from greedy.core.reduction_common import tensor_grid_interior_split

    with pytest.raises(ValueError, match="multi-dimensional"):
        tensor_grid_interior_split(np.linspace(0.0, 1.0, 9).reshape(-1, 1))


def test_pod_max_relative_error_matches_a_direct_projection():
    """The POD curve must use the greedies' own measure: max relative W-residual."""
    from greedy.datasets.contact_forces import pod_max_relative_error

    dataset = load_contact_force_dataset("hertz")
    curve = pod_max_relative_error(dataset)

    upper = gram_cholesky(dataset.gram)
    transformed = dataset.snapshots @ upper.T
    scale = np.linalg.norm(transformed, axis=1).max()
    _, _, right = np.linalg.svd(transformed, full_matrices=False)

    assert curve[0] == pytest.approx(1.0)  # rank-0 subspace keeps the whole snapshot
    for R in (1, 4, 9):
        basis = right[:R]
        residual = transformed - transformed @ basis.T @ basis
        expected = np.linalg.norm(residual, axis=1).max() / scale
        assert curve[R] == pytest.approx(expected, rel=1e-8, abs=1e-12)


def test_pod_lower_bounds_the_cone_greedy_at_equal_R():
    """
    POD optimizes over a subspace; the greedies are confined to nonnegative
    combinations of one. So POD@R must be no worse than any greedy at that R --
    if this ever inverts, the comparison is measuring two different things.
    """
    from greedy.datasets.contact_forces import pod_max_relative_error
    from greedy.pipelines.contact_forces_compare import project_onto_cone

    dataset = load_contact_force_dataset("hertz")
    curve = pod_max_relative_error(dataset)

    upper = gram_cholesky(dataset.gram)
    transformed = dataset.snapshots @ upper.T
    scale = np.linalg.norm(transformed, axis=1).max()

    model = CPG(snapshots=transformed, epsilon=5e-2)
    model.compute_phases()
    assert model.basis_matrix is not None
    R = model.basis_matrix.shape[0]

    _, residuals = project_onto_cone(transformed, model.basis_matrix)
    assert curve[R] <= residuals.max() / scale + 1e-9


def test_adg_certificate_holds():
    """Every round admits a theta_max maximizer, so Theorem 3.5 always holds."""
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    dataset = load_contact_force_dataset("hertz")
    upper = gram_cholesky(dataset.gram)
    transformed = dataset.snapshots @ upper.T

    for normalize in (False, True):
        model = AngularDefectGreedy(
            snapshots=transformed, epsilon=1e-2, normalize_snapshots=normalize
        )
        model.compute_phases()
        assert model.verify_angular_defect_certificate()["violations"] == 0


def test_adg_converges_to_the_requested_tolerance():
    """
    Standard ADG has no early exit: it enriches until every snapshot is
    resolved, so the final residual must actually meet the tolerance.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    dataset = load_contact_force_dataset("hertz")
    upper = gram_cholesky(dataset.gram)
    transformed = dataset.snapshots @ upper.T

    model = AngularDefectGreedy(snapshots=transformed, epsilon=1e-2)
    model.compute_phases()
    assert model.residual_history[-1] <= model.stopping_tolerance


def test_adg_normalization_does_not_change_selection_order():
    """
    The invariance the code relies on: normalization rescales the unresolved-set
    test, never the argmax, so one selection sequence is a prefix of the other.
    """
    from greedy.core.angle_defect_greedy import AngularDefectGreedy

    dataset = load_contact_force_dataset("hertz")
    upper = gram_cholesky(dataset.gram)
    transformed = dataset.snapshots @ upper.T

    plain = AngularDefectGreedy(snapshots=transformed, epsilon=1e-2)
    plain.compute_phases()
    normalized = AngularDefectGreedy(
        snapshots=transformed, epsilon=1e-2, normalize_snapshots=True
    )
    normalized.compute_phases()

    shared = min(len(plain.selected_indices), len(normalized.selected_indices))
    assert plain.selected_indices[:shared] == normalized.selected_indices[:shared]


def test_mcpg_dominates_at_matched_cone_size_on_hertz():
    """
    The comparison that matters: at equal R, mCPG's cone is strictly better
    conditioned and strictly more accurate than CPG's. Comparing methods at
    their own stopping R instead would confuse 'better cone' with 'stopped
    sooner'.
    """
    from greedy.pipelines.contact_forces_compare import prefix_curve
    from greedy.core.reduction_common import parameter_train_test_split

    dataset = load_contact_force_dataset("hertz")
    upper = gram_cholesky(dataset.gram)
    inverse = la.solve_triangular(upper, np.eye(upper.shape[0]), lower=False)
    transformed = dataset.snapshots @ upper.T
    train, test = parameter_train_test_split(dataset.parameters, test_fraction=0.2)

    cpg = CPG(snapshots=transformed[train], epsilon=1e-2)
    cpg.compute_phases()
    assert cpg.basis_matrix is not None
    mcpg = mCPG(
        snapshots=transformed[train], epsilon=1e-2, constraint_transform=inverse
    )
    mcpg.compute_phases()
    assert mcpg.basis_matrix is not None

    budget = mcpg.basis_matrix.shape[0]
    cpg_at = prefix_curve(cpg.basis_matrix[:budget], transformed, train, test)[-1]
    mcpg_at = prefix_curve(mcpg.basis_matrix[:budget], transformed, train, test)[-1]

    assert mcpg_at["kappa"] < cpg_at["kappa"]
    assert mcpg_at["train_max"] < cpg_at["train_max"]
    assert mcpg_at["test_max"] <= cpg_at["test_max"]
