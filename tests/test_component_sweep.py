"""
Pins the contract between component_sweep and the greedies' internals.

component_sweep drives the algorithms incrementally rather than through
compute_phases(), so it reaches past the public API: it assigns ``model._basis``
directly and calls ``_compute_residuals`` / ``_project_onto_cone`` /
``_init_basis`` / ``_bootstrap`` / ``_next_candidate`` / ``_grow_by_one``. That
coupling is invisible to every other test, so a rename inside greedy.core would
break these pipelines silently. These tests make that break loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from greedy.core.angle_defect_greedy import AngularDefectGreedy
from greedy.core.cpg import CPG
from greedy.core.mcpg import mCPG
from greedy.pipelines.component_sweep import (
    fit_angle_fixed_components,
    fit_cpg_fixed_components,
    fit_mcpg_fixed_components,
)

FITTERS = [
    ("cpg", fit_cpg_fixed_components),
    ("angle", fit_angle_fixed_components),
    ("mcpg", fit_mcpg_fixed_components),
]


def snapshots(rows: int = 10, cols: int = 7, seed: int = 3) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).normal(size=(rows, cols))) + 0.1


@pytest.mark.parametrize("name, fit", FITTERS)
def test_fixed_component_fit_returns_the_requested_cone_size(name, fit):
    S = snapshots()
    basis, selected, _ = fit(S, components=4, zero_tol=1e-12)

    assert basis.shape == (4, S.shape[1])
    assert len(selected) == 4
    assert len(set(selected)) == 4  # no snapshot selected twice


@pytest.mark.parametrize("name, fit", FITTERS)
def test_warm_start_extends_an_existing_cone_without_rebuilding_it(name, fit):
    """The path that assigns model._basis directly: growing R by one must keep
    the earlier generators exactly, not re-run the greedy from scratch."""
    S = snapshots()
    basis, selected, _ = fit(S, components=4, zero_tol=1e-12)
    grown, grown_selected, _ = fit(
        S, components=5, zero_tol=1e-12, basis=basis, selected_indices=selected
    )

    assert grown.shape == (5, S.shape[1])
    assert grown_selected[:4] == selected
    np.testing.assert_allclose(grown[:4], basis, rtol=1e-12)


@pytest.mark.parametrize("name, fit", FITTERS)
def test_asking_for_more_components_than_snapshots_is_capped(name, fit):
    S = snapshots(rows=4, cols=5)
    basis, selected, _ = fit(S, components=99, zero_tol=1e-12)

    assert basis.shape[0] <= S.shape[0]
    assert len(selected) == basis.shape[0]


@pytest.mark.parametrize("name, fit", FITTERS)
def test_zero_components_yields_an_empty_cone(name, fit):
    S = snapshots()
    basis, selected, _ = fit(S, components=0, zero_tol=1e-12)

    assert basis.shape == (0, S.shape[1])
    assert selected == []


def test_internals_component_sweep_depends_on_still_exist():
    """
    The private attributes component_sweep reaches for. Asserted by name so a
    rename in greedy.core fails here instead of in an unrun pipeline.
    """
    S = snapshots()
    for model in (
        CPG(snapshots=S, epsilon=1e-2),
        mCPG(snapshots=S, epsilon=1e-2),
        AngularDefectGreedy(snapshots=S, epsilon=1e-2),
    ):
        assert hasattr(model, "_basis")
        assert callable(model._project_onto_cone)
        assert callable(model._compute_residuals) or callable(
            model._compute_projection_metrics
        )

    assert callable(AngularDefectGreedy(snapshots=S, epsilon=1e-2)._init_basis)
    assert callable(AngularDefectGreedy(snapshots=S, epsilon=1e-2)._next_angle_candidates)
    assert callable(AngularDefectGreedy(snapshots=S, epsilon=1e-2)._compute_projection_metrics)
    for name in ("_bootstrap", "_next_candidate", "_grow_by_one"):
        assert callable(getattr(mCPG(snapshots=S, epsilon=1e-2), name))


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda S: CPG(snapshots=S, epsilon=0.0),
        lambda S: mCPG(snapshots=S, epsilon=0.0),
        lambda S: AngularDefectGreedy(snapshots=S, epsilon=0.0),
    ],
    ids=["cpg", "mcpg", "adg"],
)
def test_zero_epsilon_terminates_instead_of_growing_forever(model_factory):
    """
    epsilon=0 asks for exact reconstruction. The residual bottoms out at
    rounding noise (~1e-15) and never reaches exactly 0, so a loop gated only on
    "residual > tolerance" re-selects the same snapshot forever. CPG did exactly
    that -- it reached R=1379 from 10 snapshots before this guard. R can never
    need to exceed the snapshot count, since the cone of all of them is exact.
    """
    S = snapshots()
    model = model_factory(S)
    model.compute_phases()

    assert model.basis_matrix.shape[0] <= S.shape[0]


def test_cpg_fixed_components_matches_compute_phases_prefix():
    """
    The incremental driver must reproduce the real algorithm: at R components it
    should select exactly what compute_phases() picks first, otherwise the
    component sweep is measuring a different method than the rest of the repo.
    """
    S = snapshots()
    model = CPG(snapshots=S, epsilon=0.0)
    model.compute_phases()

    basis, selected, _ = fit_cpg_fixed_components(S, components=4, zero_tol=1e-12)
    assert selected == model.selected_indices[:4]
    np.testing.assert_allclose(basis, model.basis_matrix[:4], rtol=1e-12)
