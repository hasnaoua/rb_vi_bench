from __future__ import annotations

import numpy as np

from greedy.core.cone_greedy import ConeGreedy
from greedy.core.reduction_common import (
    solve_cone_shift_projection,
    vector_projection_angle,
)


class mCPG(ConeGreedy):
    """
    Modified Cone-Projected Greedy (mCPG) algorithm.

    Builds a reduced positive cone  W^+_R = span_+{nu_1, ..., nu_R}  such that
    every training snapshot theta(mu) is approximated within a relative
    residual tolerance ``epsilon``, following Algorithm 2 of Niakh, Drouet,
    Ehrlacher & Ern (2022), "Stable model reduction for linear variational
    inequalities with parameter-dependent constraints".

    Unlike CPG, which stores the raw selected snapshot theta_qr as the r-th
    generator, mCPG first removes from theta_qr everything already
    representable by the current cone K_{r-1} while staying inside the
    positive cone: it solves

        Upsilon_r = argmin_{Upsilon in K_{r-1} \\cap (theta_qr - W^+)} ||theta_qr - Upsilon||,

    then sets nu_r = (theta_qr - Upsilon_r) / ||theta_qr - Upsilon_r||. The
    extra elementwise constraint theta_qr - Upsilon >= 0 (so that
    Upsilon in theta_qr - W^+) is what keeps the shift itself a valid,
    physically meaningful cone element; without it this would just be the
    plain CPG-style unconstrained cone projection. Normalizing the residual
    shift instead of the raw snapshot yields a generator that is closer to
    orthogonal to K_{r-1}, which the paper shows produces a
    better-conditioned Gram matrix than plain CPG (see ``orth_defect_history``
    below, matching eq. (41) / Figure 6 of the paper).

    ``epsilon`` is the paper's normalized tolerance δ: the loop stops when
    max_q ||theta_q - Pi_K(theta_q)||_2 / ||theta_q1||_2 <= δ, with q1 the
    first selected maximum-norm snapshot. CPG and AngularDefectGreedy use the
    same normalization in this repo so epsilon sweeps compare like with like.

    Parameters
    ----------
    snapshots : array-like, shape (N_train, d)
        The HF solution snapshots {theta(mu)}_{mu in P^tr}. Expected to be
        elementwise non-negative, i.e. already valid elements of the
        positive cone W^+ (e.g. contact pressures / Lagrange multipliers).
    epsilon : float
        Relative stopping tolerance δ.
    zero_tol : float
        Numerical tolerance below which a shift/norm is treated as zero.
    upper_bound_tol : float
        Slack added to the elementwise upper bound in the constrained
        projection, to absorb floating-point noise.
    constraint_transform : array-like, shape (d_phys, d), optional
        Map from the coordinates ``snapshots`` is expressed in back to the
        physical dual coordinates in which "stays inside W^+" means
        "elementwise >= 0".

        This exists to support a non-Euclidean dual inner product. To run mCPG
        in ||.||_W one passes W-transformed snapshots (U @ theta, with
        W = U.T @ U); the objective, the residuals and nu_r's normalization are
        then all correctly measured in the W-norm for free. But the cone-shift
        constraint theta_qr - Upsilon in W^+ is elementwise, and U does not
        preserve elementwise-nonnegativity — left alone, the QP would quietly
        enforce U(theta - Upsilon) >= 0, which is a different (and meaningless)
        feasible set. Passing ``constraint_transform=inv(U)`` restores the
        constraint to the physical coordinates.

        ``None`` (the default) means ``snapshots`` is already physical and the
        constraint is applied directly, which is the plain Euclidean mCPG.
    """

    _display_name = "mCPG"

    def __init__(
        self,
        snapshots: np.ndarray,
        epsilon: float = 0.1,
        *,
        zero_tol: float = 1e-12,
        upper_bound_tol: float = 1e-9,
        constraint_transform: np.ndarray | None = None,
    ):
        self.upper_bound_tol = float(upper_bound_tol)
        if constraint_transform is None:
            self.constraint_transform = None
        else:
            self.constraint_transform = np.asarray(constraint_transform, dtype=float)
            if self.constraint_transform.ndim != 2:
                raise ValueError(
                    "constraint_transform must be 2-D, got shape "
                    f"{self.constraint_transform.shape}"
                )
            if self.constraint_transform.shape[1] != np.shape(snapshots)[-1]:
                raise ValueError(
                    f"constraint_transform columns ({self.constraint_transform.shape[1]}) "
                    f"must match snapshot width ({np.shape(snapshots)[-1]})"
                )
        super().__init__(snapshots, epsilon, zero_tol=zero_tol)

    def _init_state(self) -> None:
        # Histories specific to mCPG; the shared ones come from ConeGreedy.
        self.shift_norm_history: list[float] = []   # ||theta_qr - Upsilon_r|| per step
        self.orth_defect_history: list[float] = []  # eq. (41): e_orth(r-1)
        # Angle (radians) between the *raw* candidate theta_qr and its plain
        # cone projection onto K_{r-1}, before the constrained shift is
        # applied. Directly comparable to CPG.candidate_angle_history and
        # AngularDefectGreedy.angle_history: all three use the same
        # vector_projection_angle formula on the same "candidate vs.
        # pre-update cone" pair.
        self.candidate_angle_history: list[float] = []

    # ------------------------------------------------------------------
    # Growth primitives, exposed so fixed-component sweeps (see
    # component_sweep_comparison.fit_mcpg_fixed_components) can drive the
    # algorithm incrementally, the same way they already reach into CPG's
    # and AngularDefectGreedy's internals.
    # ------------------------------------------------------------------

    def _bootstrap(self) -> bool:
        """
        Select q_1 = argmax_q ||theta_q|| and build nu_1 through the exact
        same primitive as every later generator (``_grow_by_one``), instead
        of special-casing it: with K_0 = {0}, Upsilon_1 is trivially 0 (the
        only element of K_0, and the QP's k=0 branch already returns it), so
        this reduces to nu_1 = theta_q1 / ||theta_q1|| as required by
        Algorithm 2, while also populating shift_norm/orth_defect/angle
        history for step 1 instead of leaving it blank.
        """
        if self.snapshots.shape[0] == 0:
            return False
        norms = np.linalg.norm(self.snapshots, axis=1)
        q1 = int(np.argmax(norms))
        if norms[q1] <= self.zero_tol:
            return False
        return self._grow_by_one(q1)

    def _next_candidate(self, residuals: np.ndarray) -> int | None:
        scores = residuals.copy()
        if self.selected_indices:
            scores[np.array(self.selected_indices, dtype=int)] = -np.inf
        candidate = int(np.argmax(scores))
        if not np.isfinite(scores[candidate]):
            return None
        return candidate

    def _clamp_shift(self, shift: np.ndarray) -> np.ndarray:
        """
        Force theta_qr - Upsilon_r back into W^+ before it is normalized.

        ``solve_cone_shift_projection`` imposes the membership
        Upsilon in theta_qr - W^+ as ``cone_system @ c <= cone_vector +
        upper_bound_tol``. That slack is ABSOLUTE, so the shift may dip to
        -upper_bound_tol. Line 10 then divides by ``shift_norm``, which turns
        the dip into a relative violation of

            upper_bound_tol / ||theta_qr - Upsilon_r||

        -- unbounded as the shift norm shrinks, which is exactly what happens as
        the cone saturates and residuals get small. Measured on the Hertz
        contact dataset the generators reached -6.4e-08 at epsilon=1e-2, and on
        tightly clustered snapshots (shift norms ~4e-4) they reached -2.4e-06,
        past this package's own >= -1e-6 assertion.

        That breaks the invariant the algorithm exists to maintain: the paper's
        Section 4 rejects Gram-Schmidt precisely because it "would lead to a
        departure from the positive cone W^+", and nu_r <0 is such a departure.

        Clamping costs an O(upper_bound_tol) perturbation of Upsilon_r, which
        may push it marginally outside K_{r-1}. That is the right trade:
        Upsilon_r is discarded immediately, whereas nu_r is appended to the cone
        and every downstream projection, non-negativity check and reduced
        multiplier inherits its sign.
        """
        if self.constraint_transform is None:
            return np.maximum(shift, 0.0)
        # With a transform T, "inside W^+" is elementwise in PHYSICAL
        # coordinates, and physical shift = T @ working shift (both theta and
        # A c are mapped by the same T). Clamp there, then map back; T is
        # inv(U) from a Cholesky factor, hence square and invertible.
        physical = self.constraint_transform @ shift
        clamped = np.maximum(physical, 0.0)
        if np.array_equal(clamped, physical):
            return shift
        return np.linalg.solve(self.constraint_transform, clamped)

    def _grow_by_one(self, candidate: int) -> bool:
        """
        Add the generator built from ``candidate`` (lines 9-11 of Algo. 2):
        solve for Upsilon_r, shift theta_qr by it, and normalize.
        """
        theta = self.snapshots[candidate]
        A = self._as_basis_matrix()

        # Angle of the raw candidate to K_{r-1} (plain cone projection),
        # before the constrained shift — the same "how novel is this
        # direction" diagnostic CPG and AngularDefectGreedy report.
        raw_projection = self._project_onto_cone(theta)
        candidate_angle = vector_projection_angle(theta, raw_projection)

        # With a constraint_transform, the objective stays in the (possibly
        # W-transformed) working coordinates while the theta - Upsilon in W^+
        # bound is imposed back in physical ones. See the class docstring.
        if self.constraint_transform is None:
            cone_system = None
            cone_vector = None
        else:
            cone_system = self.constraint_transform @ A
            cone_vector = self.constraint_transform @ theta

        coeffs = solve_cone_shift_projection(
            A,
            theta,
            upper_tol=self.upper_bound_tol,
            cone_system=cone_system,
            cone_vector=cone_vector,
        )
        upsilon = A @ coeffs
        shift = self._clamp_shift(theta - upsilon)
        shift_norm = float(np.linalg.norm(shift))
        if shift_norm <= self.zero_tol:
            # See Remark 4.2: this can only happen if theta_qr was already in
            # K_{r-1}, in which case the residual gate should already have
            # stopped the loop. Guard against it defensively.
            return False

        nu = shift / shift_norm

        # Orthogonality defect of nu_r w.r.t. K_{r-1}, eq. (41): computed
        # against the *plain* cone projection, before nu_r is appended.
        plain_projection = self._project_onto_cone(nu)
        orth_defect = float(np.linalg.norm(nu - plain_projection))

        self._basis.append(nu)
        self.selected_indices.append(candidate)
        self.shift_norm_history.append(shift_norm)
        self.orth_defect_history.append(orth_defect)
        self.candidate_angle_history.append(candidate_angle)
        return True

    # ------------------------------------------------------------------
    # Main algorithm (Algorithm 2)
    # ------------------------------------------------------------------

    def compute_phases(self) -> None:
        """
        Run Algorithm 2 — modified Cone-Projected Greedy (mCPG).

        After completion:
          - self.basis_matrix          : ndarray (R, d) — the R generators nu_1..nu_R
          - self.selected_indices      : list[int]       — snapshot rows q_1..q_R
          - self.residual_history      : list[float]     — r_n values (diagnostic)
          - self.orth_defect_history   : list[float]     — eq. (41) e_orth(r)
        """
        self._reset_common_history()
        self.shift_norm_history = []
        self.orth_defect_history = []
        self.candidate_angle_history = []

        if not self._bootstrap():
            self._finalize_basis()
            return

        first_norm = self.stopping_scale
        residuals = self._compute_residuals()
        r_n = float(np.max(residuals))
        self.residual_history.append(r_n)
        self.residual_basis_sizes.append(len(self._basis))
        self.relative_residual_history.append(r_n / first_norm if first_norm > 0.0 else 0.0)

        max_iterations = self.snapshots.shape[0]
        while r_n > self.stopping_tolerance and len(self._basis) < max_iterations:
            candidate = self._next_candidate(residuals)
            if candidate is None:
                break
            if not self._grow_by_one(candidate):
                break

            residuals = self._compute_residuals()
            r_n = float(np.max(residuals))
            self.residual_history.append(r_n)
            self.residual_basis_sizes.append(len(self._basis))
            self.relative_residual_history.append(
                r_n / first_norm if first_norm > 0.0 else 0.0
            )

        self._finalize_basis()
