"""
Contact-force (dual Lagrange multiplier) datasets for the Niakh-Drouet-
Ehrlacher-Ern (ESAIM:M2AN 2022) test cases.

Every error, projection and angle for these datasets is measured in ||.||_W
rather than the Euclidean norm (see docs/contact_force_datasets.md), so each
dataset carries a Gram matrix alongside its snapshots. The greedy algorithms in
this package take snapshots row-wise; ``load_contact_force_dataset`` normalizes
whatever layout the .npz uses.

Two .npz schemas are supported, described per case in ``CASES``:

* the original solver output (``membrane``, ``hertz``) — snapshots column-wise
  as ``Lambda`` (n_c x P), with the Gram stored explicitly as ``W_gram``;
* the reconstructed Hertz cases (``hertz_a``, ``hertz_b``) — snapshots row-wise
  as ``snapshots`` (P x n_c), with no stored Gram. Their dual space is the same
  1-D contact arc, so W is rebuilt as the P1 arc mass matrix over ``abscissas``
  (``gram_source="arc_mass"``) — the same construction the ``hertz`` solver used
  for its own ``W_gram``.

See ``greedy.synthetic_data.contact_forces`` for the high-fidelity solvers that
generate the .npz files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la


DEFAULT_DATA_DIR = Path("data/contact_forces")

# Both cases keep a fixed dual space across mu, which is what allows the
# lambda(mu_p) to be stacked into one matrix at all. What varies is the
# parameter: the membrane sweeps a 3-D grid (radius, cx, cy) while Hertz sweeps
# the scalar R2. ``scalar_parameter_column`` picks the component used as the
# x-axis for plots and for the ordered train/test split; for the membrane the
# obstacle radius mu_1 is the one that actually resizes the contact support.
CASES: dict[str, dict[str, object]] = {
    "membrane": {
        "filename": "membrane_contact_forces.npz",
        "parameter_names": ("radius", "cx", "cy"),
        "scalar_parameter_column": 0,
        "scalar_parameter_name": "obstacle radius mu_1",
        "coordinate_key": "node_coords",
        "description": "Membrane obstacle problem (Sec 5.1): lambda is the obstacle reaction.",
    },
    "hertz": {
        "filename": "hertz_contact_forces.npz",
        "parameter_names": ("R2",),
        "scalar_parameter_column": None,
        "scalar_parameter_name": "body-2 radius R2",
        "coordinate_key": "contact_abscissa",
        "description": "Hertz contact problem (Sec 5.2): lambda is the contact pressure on body 1.",
    },
    "hertz_a": {
        "filename": "hertz_case_a_imposed_displacement.npz",
        "parameter_names": ("d",),
        "scalar_parameter_column": None,
        "scalar_parameter_name": "imposed displacement d",
        "coordinate_key": "abscissas",
        "snapshot_key": "snapshots",
        "snapshot_layout": "rows",
        "parameter_key": "params",
        "gram_source": "arc_mass",
        "description": (
            "Hertz contact, case (a): parametric imposed displacement d; "
            "lambda is the contact pressure on the reference arc."
        ),
    },
    "hertz_b": {
        "filename": "hertz_case_b_parametric_geometry.npz",
        "parameter_names": ("R2",),
        "scalar_parameter_column": None,
        "scalar_parameter_name": "body-2 radius R2",
        "coordinate_key": "abscissas",
        "snapshot_key": "snapshots",
        "snapshot_layout": "rows",
        "parameter_key": "params",
        "gram_source": "arc_mass",
        "description": (
            "Hertz contact, case (b): parametric geometry R2 = mu; "
            "lambda is the contact pressure on the reference arc."
        ),
    },
}


def arc_mass_matrix(abscissas: np.ndarray) -> np.ndarray:
    """
    P1 mass matrix of a 1-D contact arc with nodes at ``abscissas``.

    This is the L2 Gram of the dual space when lambda is a P1 pressure field on
    the arc, so ||lambda||_W is its L2 norm rather than a mesh-dependent nodal
    one. Mirrors ``greedy.synthetic_data.contact_forces.hertz_hf.arc_mass_1d``,
    duplicated here only because that module imports cvxopt at import time and
    loading a dataset must not require the HF solver stack.
    """
    x = np.asarray(abscissas, dtype=float).reshape(-1)
    if x.size < 2:
        raise ValueError(f"need at least 2 arc nodes to build a mass matrix, got {x.size}")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("abscissas must be strictly increasing along the arc")

    mass = np.zeros((x.size, x.size), dtype=float)
    widths = np.diff(x)
    for element, h in enumerate(widths):
        mass[element, element] += h / 3.0
        mass[element + 1, element + 1] += h / 3.0
        mass[element, element + 1] += h / 6.0
        mass[element + 1, element] += h / 6.0
    return mass


@dataclass(frozen=True)
class ContactForceDataset:
    """Dual snapshots plus everything needed to work in the W-inner product."""

    name: str
    snapshots: np.ndarray          # (P, n_c) — row p is lambda(mu_p), rowwise for the greedies
    gram: np.ndarray               # (n_c, n_c) — <a,b>_W = a.T @ gram @ b
    mu_samples: np.ndarray         # (P,) or (P, n_params)
    parameters: np.ndarray         # (P,) the scalar parameter used for plots/splits
    parameter_name: str
    coordinates: np.ndarray | None  # node coords / contact abscissa, for plotting
    source: str
    description: str

    @property
    def snapshot_count(self) -> int:
        return int(self.snapshots.shape[0])

    @property
    def dual_dimension(self) -> int:
        return int(self.snapshots.shape[1])


def load_contact_force_dataset(
    case: str,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> ContactForceDataset:
    """Load the ``membrane`` or ``hertz`` contact-force dataset."""
    if case not in CASES:
        raise ValueError(f"case must be one of {sorted(CASES)}, got {case!r}")

    spec = CASES[case]
    path = Path(data_dir) / str(spec["filename"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Regenerate it with "
            "python -m greedy.synthetic_data.contact_forces.generate_datasets"
        )

    data = np.load(path, allow_pickle=True)
    snapshot_key = str(spec.get("snapshot_key", "Lambda"))
    parameter_key = str(spec.get("parameter_key", "mu_samples"))
    lam = np.asarray(data[snapshot_key], dtype=float)
    mu_samples = np.asarray(data[parameter_key], dtype=float)

    if lam.ndim != 2:
        raise ValueError(f"{snapshot_key} must be 2-D, got shape {lam.shape}")
    # Normalize to the row-wise (P, n_c) layout the greedies want, whichever way
    # the file stored it.
    if str(spec.get("snapshot_layout", "columns")) == "columns":
        snapshots = np.ascontiguousarray(lam.T)
    else:
        snapshots = np.ascontiguousarray(lam)

    if mu_samples.shape[0] != snapshots.shape[0]:
        raise ValueError(
            f"{parameter_key} has {mu_samples.shape[0]} entries but {snapshot_key} "
            f"holds {snapshots.shape[0]} snapshots"
        )
    if not np.all(np.isfinite(snapshots)):
        raise ValueError(f"{snapshot_key} contains non-finite values")

    # The greedies assume snapshots live in the cone W+ (elementwise >= 0); mCPG's
    # cone-shift constraint is meaningless otherwise, so fail loudly rather than
    # silently producing an infeasible QP. A small negative tolerance absorbs
    # solver noise from the HF dual QP.
    most_negative = float(np.min(snapshots))
    if most_negative < -1e-6 * max(1.0, float(np.max(snapshots))):
        raise ValueError(
            f"{snapshot_key} must be nonnegative to lie in the cone W+, but its "
            f"minimum is {most_negative:.6e}"
        )

    coordinate_key = str(spec["coordinate_key"])
    coordinates = (
        np.asarray(data[coordinate_key], dtype=float)
        if coordinate_key in data.files
        else None
    )

    n_c = snapshots.shape[1]
    if str(spec.get("gram_source", "file")) == "arc_mass":
        if coordinates is None:
            raise ValueError(
                f"case {case!r} rebuilds its Gram from the arc geometry but "
                f"{path} has no {coordinate_key!r} entry"
            )
        gram = arc_mass_matrix(coordinates)
    else:
        gram = np.asarray(data["W_gram"], dtype=float)
    if gram.shape != (n_c, n_c):
        raise ValueError(
            f"Gram shape {gram.shape} must be (n_c, n_c) = ({n_c}, {n_c})"
        )

    column = spec["scalar_parameter_column"]
    if column is None:
        parameters = mu_samples.reshape(-1)
    else:
        parameters = np.asarray(mu_samples[:, int(column)], dtype=float)

    description = (
        str(data["description"]) if "description" in data.files else str(spec["description"])
    )

    return ContactForceDataset(
        name=case,
        snapshots=snapshots,
        gram=gram,
        mu_samples=mu_samples,
        parameters=parameters,
        parameter_name=str(spec["scalar_parameter_name"]),
        coordinates=coordinates,
        source=str(path),
        description=description,
    )


def wpod_error(
    dataset: ContactForceDataset,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Relative W-norm POD projection error e(R) for R = 0..rank.

    This is the linear-subspace baseline the cone greedies are judged against:
    POD is free to use negative coefficients, so it lower-bounds what any
    cone-constrained method can reach at the same R. Where a greedy comes close
    to this curve, the cone constraint is costing little.

    ``indices`` restricts the snapshots the POD is fitted and scored on. Pass
    the training rows to compare against a greedy's training convergence: a POD
    fitted on all snapshots has seen the held-out ones and is not a bound on
    anything the greedy could have achieved.
    """
    snapshots = dataset.snapshots if indices is None else dataset.snapshots[indices]
    upper = la.cholesky(dataset.gram, lower=False)
    singular_values = la.svdvals(upper @ snapshots.T)
    total = float((singular_values ** 2).sum())
    if total <= 0.0:
        return np.zeros(singular_values.size + 1, dtype=float)
    return np.array(
        [
            np.sqrt(max(total - float((singular_values[:R] ** 2).sum()), 0.0) / total)
            for R in range(singular_values.size + 1)
        ],
        dtype=float,
    )


def pod_max_relative_error(
    dataset: ContactForceDataset,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    """
    POD error e(R) under the *greedy's* error measure, for R = 0..rank.

    ``wpod_error`` normalizes by total energy (a Frobenius-style average), but
    CPG / ADG / mCPG all stop on max_x ||x - proj(x)||_W / max_x ||x||_W. Those
    two numbers are not comparable, so plotting one against the other is
    meaningless. This returns the POD curve under the greedy's own measure,
    which *is* a valid lower bound on what any of them can reach at a given R:
    the greedies are restricted to nonnegative combinations of a subspace POD
    optimizes over freely.
    """
    snapshots = dataset.snapshots if indices is None else dataset.snapshots[indices]
    upper = la.cholesky(dataset.gram, lower=False)
    transformed = snapshots @ upper.T  # ||.||_W becomes the Euclidean norm here

    scale = float(np.max(np.linalg.norm(transformed, axis=1)))
    if scale <= 0.0:
        return np.zeros(1, dtype=float)

    _, _, right = np.linalg.svd(transformed, full_matrices=False)
    coordinates = transformed @ right.T  # energy of each snapshot per POD mode
    # Residual of snapshot i against the rank-R subspace is the tail energy
    # beyond mode R, so sum the squared coordinates from the tail inward.
    tail = np.concatenate(
        [
            np.cumsum((coordinates ** 2)[:, ::-1], axis=1)[:, ::-1],
            np.zeros((coordinates.shape[0], 1)),
        ],
        axis=1,
    )
    return np.max(np.sqrt(np.maximum(tail, 0.0)), axis=0) / scale
