"""The membrane HF operators, and whether the reduced solve reproduces the HF one.

``membrane_2d`` is the one retained source that exposes its own assembly, so it is the
one dataset on which ``metrics.online`` can run at all. Everything the solved-error
metric reports rests on the (A, B, f, g) mapping in ``datasets._membrane_2d`` being the
same problem the HF model solves -- and the sign convention is the part that would fail
silently, since a flipped B still produces a well-posed QP with a plausible-looking
answer. These tests pin it against ``MembraneHF.solve`` itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench import datasets as ds_mod
from bench.metrics.online import solve_reduced_general


@pytest.fixture(scope="module")
def membrane():
    return ds_mod.load("membrane_2d")


def test_membrane_now_supports_the_solved_error_metric(membrane):
    """Caveat 1 said no retained dataset could drive a reduced solve. This one can."""
    assert membrane.supports_online
    assert membrane.A is not None
    assert membrane.B_of_mu(0).shape == (membrane.dim, membrane.A.shape[0])
    assert membrane.rhs_of_mu(0).shape == (membrane.A.shape[0],)
    assert membrane.gap_of_mu(0).shape == (membrane.dim,)


def test_reduced_solve_lands_at_or_below_the_HF_objective(membrane):
    """The mapping is exact; the residual against ``hf.solve`` is cvxopt's own slack.

    Comparing multipliers directly cannot separate "wrong operators" from "the reference
    solver stopped early", and on this problem the reduced Schur complement is
    rank-deficient (Appendix Step 5), so a small objective gap shows up as a much larger
    multiplier gap. Comparing on the HF's OWN objective settles it: if the mapping were
    wrong we would be minimizing a different functional and would generally land above.
    """
    import bench._paths  # noqa: F401
    from greedy.synthetic_data.contact_forces.membrane_hf import (
        MembraneHF, obstacle, training_grid,
    )

    hf = MembraneHF(n=36)
    params = training_grid()
    P = np.asarray(hf._P, float)
    Mc = np.asarray(hf._Mc_d, float)
    cn = np.asarray(hf.cnodes, int)
    V = np.eye(membrane.A.shape[0])
    Xi = np.eye(membrane.dim)

    for i in (0, 40, 99):
        psi = obstacle(hf.cnode_coords, params[i])
        b = Mc @ (psi - hf.Kinv_f[cn])
        _u, ours = solve_reduced_general(
            membrane.A, membrane.rhs_of_mu(i), membrane.gap_of_mu(i),
            V, Xi, membrane.B_of_mu(i))
        _u_hf, lam_hf = hf.solve(params[i])

        def J(L):
            return 0.5 * float(L @ P @ L) - float(L @ b)

        assert J(ours) <= J(lam_hf) + 1e-12, (i, J(ours), J(lam_hf))
        # and the displacement agrees to the reference solver's own accuracy
        assert np.linalg.norm(_u - _u_hf) / np.linalg.norm(_u_hf) < 1e-3, i


def test_gap_is_the_only_parameter_dependent_operator(membrane):
    """B and f are constant for this source; only the obstacle moves with mu.

    Asserted so that a later source with a genuinely varying B(mu) cannot be added by
    copying this one without noticing the difference.
    """
    assert np.allclose(membrane.B_of_mu(0), membrane.B_of_mu(50))
    assert np.allclose(membrane.rhs_of_mu(0), membrane.rhs_of_mu(50))
    assert not np.allclose(membrane.gap_of_mu(0), membrane.gap_of_mu(50))
