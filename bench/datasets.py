"""The dataset registry: every source of non-negative dual snapshots in the merge.

The eight sources differ in exactly the ways that matter for this benchmark, and the
differences are the reason for running all of them rather than picking one:

=================== ======================== ======== ================================
key (``--datasets``) reported name             dim x n  why it is here
=================== ======================== ======== ================================
toy_bee20           toy_bee20                60 x 40  [BEE20]'s own test problem, B = I
obstacle_ndee22     obstacle_ndee22          40 x 25  the only parameter-dependent B(mu)
hertz_pressure      hertz_pressure           128 x 4k validated physics, 4000 samples
gaussian_synth      gaussian_synth           200 x n  controlled rank, cheap sweeps
fem_lambda          Half-disks of Hertz       57 x 50 [BEE20] §6.2, real FEM multipliers
fem_lambda_pressure Half-disks of Hertz (p.)  57 x 50 same, half-support corrected
physics             3D Pellet-Cladding       7676x96  76x101 quarter sector, dimension
membrane_2d         membrane_2d              varies   2-D, [NDEE22] §5.1 analogue
hertz_2d            hertz_2d                 varies   2-D elasticity, §5.2 analogue
=================== ======================== ======== ================================

**Keys versus names.** The registry key is the CLI handle and never changes; the
``Dataset.name`` is what appears in ``grid.csv``, figure titles and directory names.
Where a source *is* a recognised physical problem, the name says so -- ``fem_lambda`` is
[BEE20] §6.2's Hertz half-disks and ``physics`` is the 3-D pellet-cladding contact --
because a reader of a figure should see the problem, not the file it came from.

**Tiers.** ``fast`` sources need only numpy/scipy and are the default grid. ``heavy``
ones are the two 2-D FEM models, and they are opt-in for a dependency reason rather than
a runtime one: both import ``cvxopt`` at module scope, which is ``greedy_algos``'
optional ``[qp]`` extra, so the default grid stays runnable on a bare install. Their
build cost is modest (roughly 30s and 7s respectively). Install with::

    pip install cvxopt        # or: pip install -e repos/greedy_algos[qp]

Nothing about the metrics differs between tiers.

**Provenance, not citation.** ``Dataset.paper`` records which repository a source
came from. It is *not* a claim that the source reproduces a number in that paper --
none of them do, and every REPRODUCTION_NOTES in this monorepo says so explicitly.
The 1-D toys are substitutes for §6 / §5.1, chosen so the algorithms can be exercised
on CPU without a FEM stack.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import _paths, geometry
from .types import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    build: Callable[[], Dataset]
    tier: str            # "fast" | "heavy"
    description: str = ""


# ---------------------------------------------------------------------------
# [BEE20] sources
# ---------------------------------------------------------------------------

def _toy_bee20() -> Dataset:
    """[BEE20]'s 1-D obstacle toy. ``B = I``: the constraint is ``u <= gap``.

    Carries primal snapshots and the stiffness ``A``, so it supports the full
    inf-sup family. Its ``B`` is parameter-*independent*, which is precisely why
    [NDEE22]'s PGA has no online cost to remove here -- that vacuity is a documented
    finding, not a defect of the dataset.
    """
    from toy_problem import generate_snapshots, obstacle_gap

    S_pri, S_du, params, A, F = generate_snapshots(N=60, n_train=40, seed=0)
    n = S_du.shape[1]
    idx = np.arange(n)
    # Deterministic interleaved split: no ordering structure in the sampled mu, so a
    # stride keeps train and test over the same parameter range.
    test_idx = idx[::4]
    train_idx = np.setdiff1d(idx, test_idx)
    B = np.eye(S_pri.shape[0])
    return Dataset(
        name="toy_bee20",
        snapshots=S_du,
        description="1-D obstacle problem, B = I (parameter-independent constraint)",
        paper="[BEE20] rb_contact_cpg",
        params=params,
        train_idx=train_idx,
        test_idx=test_idx,
        primal_snapshots=S_pri,
        A=A,
        B_of_mu=lambda i, B=B: B,
        # The load and the obstacle the parameter actually moves, so the reduced
        # saddle-point problem can be solved rather than only scored against snapshots.
        rhs_of_mu=lambda i, F=F: F[:, i],
        gap_of_mu=lambda i, p=params: obstacle_gap(60, p[i]),
    )


def _hertz_pressure() -> Dataset:
    """Validated Hertz half-plane contact pressures, from the cached dataset.

    4000 samples of a boundary-integral contact model that reproduces the analytic
    semi-elliptical profile to ~0.05 median relative shape error. Multipliers are
    non-negative by KKT construction.

    No ``A``/``B`` are exposed: the model supplies only the Schur complement
    ``B A^-1 B^T = C``, with the primal field eliminated analytically, so the
    consistent factorization is ``B = I``. That makes it unsuitable for the inf-sup
    experiments -- established at length in ``rb_contact_cpg``'s hertz_infsup_probe --
    but it is the best *dual snapshot* source in the merge, which is what the
    precision and performance axes need.
    """
    npz = _paths.REPOS / "rb_contact_cpg" / "data" / "contact_pressures.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"{npz} not found; regenerate with rb_contact_cpg/src/make_dataset.py"
        )
    d = np.load(npz, allow_pickle=False)
    P = np.asarray(d["P"], float).T            # (n, dim) -> (dim, n)
    return Dataset(
        name="hertz_pressure",
        snapshots=P,
        description="Hertz half-plane contact pressures, 4000 samples, 128 nodes",
        paper="[BEE20] rb_contact_cpg",
        params=np.asarray(d["params"], float),
        train_idx=np.asarray(d["train_idx"], int),
        test_idx=np.asarray(d["test_idx"], int),
    )


# ---------------------------------------------------------------------------
# [NDEE22] source
# ---------------------------------------------------------------------------

def _obstacle_ndee22() -> Dataset:
    """[NDEE22]'s 1-D obstacle with a **parameter-dependent** constraint ``B(mu)``.

    The one structural feature the paper turns on: ``mu = (r, c)`` moves the
    collocation points ``s_i(mu) = c + r*s_hat``, so ``B(mu)`` genuinely varies while
    ``g`` does not. This is the only source in the merge with that property, which
    makes it the only one where the inf-sup / PGA metrics are testing what [NDEE22] is
    actually about.

    Uses the repository's own ``training_set`` / ``validation_set``, including the
    retuned ``offset``/``curv`` its notes explain are necessary -- with §5.1's own
    coefficient the 1-D contact set collapses to a single point and the multiplier
    carries no spatial structure for a cone algorithm to compress.
    """
    from hf_model import ObstacleHF, training_set, validation_set

    hf = ObstacleHF(n_elem=200, n_dual=40)
    train_params = training_set(n_r=5, n_c=5)
    valid_params = validation_set(n=15, seed=1)
    all_params = np.vstack([train_params, valid_params])

    U, L = hf.snapshots(all_params)
    n_train = len(train_params)
    return Dataset(
        name="obstacle_ndee22",
        snapshots=L,
        description="1-D obstacle, parameter-dependent constraint B(mu)",
        paper="[NDEE22] stable_model_reduction_vi",
        params=all_params,
        train_idx=np.arange(n_train),
        test_idx=np.arange(n_train, len(all_params)),
        primal_snapshots=U,
        A=hf.K,
        B_of_mu=lambda i, hf=hf, p=all_params: hf.B(p[i]),
        # f and g are parameter-INDEPENDENT here; mu enters only through B(mu), which
        # is the one structural feature [NDEE22] §5.1 turns on. Both are still exposed
        # per index so the metric needs no special case.
        rhs_of_mu=lambda i, hf=hf: hf.f,
        gap_of_mu=lambda i, hf=hf: hf.g,
    )


# ---------------------------------------------------------------------------
# greedy_algos sources
# ---------------------------------------------------------------------------

def _gaussian_synth() -> Dataset:
    """Gaussian-bump synthetic fields: known generating rank, fully controlled.

    The only source where the ideal basis size is known in advance, which makes it the
    right place to ask whether a method's ``R`` at a given tolerance is close to the
    truth rather than merely self-consistent. Seeded explicitly -- ``create_data``
    defaults to the unseeded global RNG, which would make every run irreproducible.
    """
    from greedy.synthetic_data.gaussian_data import create_data, gaussian_basis

    _x, basis = gaussian_basis(dim_basis=12, width=0.08, discretization_count=200)
    fields = create_data(basis, num_fields=120, noise_level=0.0, min_value=0.0, rng=0)
    S = np.ascontiguousarray(np.asarray(fields, float).T)     # (n, dim) -> (dim, n)
    idx = np.arange(S.shape[1])
    test_idx = idx[::4]
    return Dataset(
        name="gaussian_synth",
        snapshots=S,
        description="Gaussian-bump fields, 12 generating modes, 120 samples",
        paper="greedy_algos synthetic",
        train_idx=np.setdiff1d(idx, test_idx),
        test_idx=test_idx,
    )


def _fem_lambda() -> Dataset:
    """Real FEM contact multipliers, split on the paper's own parameter grid.

    Uses ``fem_sols_train_test_split``, which matches the [BEE20] grid
    ``{0.15 + 0.01 i}`` against the local ``rad`` folder names via the documented
    ``rad - 0.65`` shift. That protocol is the repository's, not this benchmark's --
    reusing it keeps the train/test numbers comparable with ``greedy_algos``' existing
    lambda pipelines.
    """
    from greedy.core.reduction_common import fem_sols_train_test_split, load_lambda_dataset

    npz = _paths.GREEDY_ROOT / "results" / "lambda" / "dataset" / "lambda_dataset.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"{npz} not found; build it with `python -m greedy.datasets.lambda_snapshots` "
            f"from {_paths.GREEDY_ROOT} (requires data/FEM_SOLS.zip)"
        )
    snapshots, radii, _src = load_lambda_dataset(Path(npz))
    train_idx, test_idx = fem_sols_train_test_split(radii)
    return Dataset(
        name="Half-disks of Hertz",
        snapshots=np.ascontiguousarray(snapshots.T),          # (n, dim) -> (dim, n)
        description="[BEE20] §6.2 half-disk contact, 57 nodes from the symmetry axis out",
        paper="greedy_algos / FEM_SOLS",
        params=radii.reshape(-1, 1),
        train_idx=np.asarray(train_idx, int),
        test_idx=np.asarray(test_idx, int),
        # Already ordered along the contact line: index 0 is the symmetry axis and index
        # increases outward. Established rather than assumed -- see FEM_LAMBDA_ORDERING.
        geometry=geometry.mirrored_line_geometry(snapshots.shape[1]),
    )


def _fem_lambda_pressure() -> Dataset:
    """``fem_lambda`` converted from nodal forces to a **pressure-like** field.

    Same snapshots, with the symmetry-axis node's half-support weighting undone. Under
    ``lambda_i = integral p phi_i ~ p(x_i) h_i`` with uniform spacing ``h``, the tributary
    length is ``h`` at every interior node and ``h/2`` at node 0, whose hat is truncated
    by the symmetry plane. So ``p_i = lambda_i / h_i`` is ``lambda`` with node 0 doubled,
    up to the global constant ``h`` -- and that constant is irrelevant, since ``span_+``
    is invariant to a positive rescaling of the whole dataset.

    **This is a genuinely different reduction problem, not a cosmetic rescaling.** The
    cone algorithms are invariant to scaling each *snapshot*, but not to scaling a
    *coordinate*: doubling one row moves every snapshot in a direction the cone geometry
    can see, so the selected parameters and the achieved errors may differ. Which is the
    point of carrying both -- it measures whether the FEM nodal-force weighting distorts
    the reduction relative to the physical field.

    Two assumptions, both established in ``FEM_LAMBDA_ORDERING`` rather than posited:
    uniform spacing (Hertz semi-ellipse fits the node index at R^2 = 0.989), and node 0
    as the only half-support node (``lambda[0]/lambda[1] = 0.5035 +/- 0.019``). The
    residual +/-8% mesh grading is *not* corrected -- that would need tributary lengths,
    which the archive does not carry.
    """
    base = _fem_lambda()
    S = base.snapshots.copy()
    S[0, :] *= 2.0
    return Dataset(
        name="Half-disks of Hertz (pressure)",
        snapshots=S,
        description="[BEE20] §6.2 half-disk, nodal forces converted to pressure",
        paper="greedy_algos / FEM_SOLS",
        params=base.params,
        train_idx=base.train_idx,
        test_idx=base.test_idx,
        geometry=geometry.mirrored_line_geometry(base.dim),
    )


#: What is known about the FEM_SOLS node ordering, and how.
#:
#: FEM_SOLS ships no coordinates and no mesh connectivity -- only 57 multiplier values per
#: parameter -- so whether the nodes are spatially ordered has to be established from the
#: values themselves. Four checks, all on the 50 snapshots:
#:
#: 1. **They are already ordered.** Total variation is 7.81 in the given order against
#:    84.0 for a random permutation and 5.08 for the (unattainable) sort-by-magnitude
#:    bound. In the active block 91% of consecutive steps strictly decrease.
#:
#: 2. **Node 0 is the symmetry-axis node**, carrying a half-support shape function:
#:    ``lambda[0]/lambda[1] = 0.5035 +/- 0.019`` across all 50 snapshots. So these are
#:    nodal *forces* (``integral p phi_i``), not pressures, and node 0's hat function is
#:    truncated by the symmetry plane.
#:
#: 3. **Spacing is essentially uniform.** Doubling node 0 and fitting the semi-elliptical
#:    Hertz profile ``p0 sqrt(1-(x/a)^2)`` against the node *index* gives R^2 = 0.989,
#:    with contact half-width a = 13.3 nodes -- matching the 15-17 active nodes measured.
#:    Index is therefore a faithful abscissa.
#:
#: 4. **Seriation finds nothing better.** Spectral (Fiedler) ordering on the node-node
#:    correlation scored TV = 20.7, worse than the given order by 2.7x -- the 43
#:    near-zero tail nodes dominate the correlation structure.
#:
#: Two caveats. Residuals to the Hertz fit reach +/-8% with alternating sign, which is
#: mild mesh grading rather than noise; recovering exact tributary lengths would need the
#: mesh, which the archive does not carry. And the snapshots are left **uncorrected**
#: here -- node 0 is not doubled -- because scaling them would silently change every
#: benchmark number. The correction belongs in plotting, not in the data.
FEM_LAMBDA_ORDERING = """index 0 = symmetry axis, increasing outward; uniform spacing;
lambda are nodal forces, and node 0 carries half weight (x2 to compare against pressure)."""


def _physics() -> Dataset:
    """High-dimensional physics contact forces: 7676 dofs, 96 snapshots.

    The only source where ``dim >> n``, so it is what exposes how each method's cost
    scales with the ambient dimension rather than with the training-set size. No
    train/test split: ``greedy_algos``' physics pipeline deliberately builds and
    evaluates on the whole dataset, and its reports summarize full-dataset residuals.
    """
    npz = _paths.GREEDY_ROOT / "results" / "physics" / "dataset" / "physics_dataset.npz"
    if not npz.is_file():
        raise FileNotFoundError(
            f"{npz} not found; build it with `python -m greedy.datasets.physics_dataset` "
            f"from {_paths.GREEDY_ROOT} (requires data/physics_data.txt)"
        )
    d = np.load(npz, allow_pickle=False)
    S = np.ascontiguousarray(np.asarray(d["snapshots"], float).T)
    return Dataset(
        name="3D Pellet-Cladding",
        snapshots=S,
        description="pellet-cladding contact, 76x101 quarter sector (no split, by design)",
        paper="greedy_algos / physics_data",
        params=np.asarray(d["radii"], float).reshape(-1, 1),
        # The 7676 entries are a structured theta x z grid, not a sequence; see
        # bench.geometry. Without this they would be drawn against a component index,
        # which slices the cladding surface into 76 strips laid end to end.
        geometry=geometry.physics_geometry(),
    )


def _membrane_2d() -> Dataset:
    """2-D membrane obstacle problem -- the [NDEE22] §5.1 analogue, in 2-D."""
    from greedy.synthetic_data.contact_forces.membrane_hf import MembraneHF, training_grid

    # n=36 rather than the module default 48: 497 contact dofs against 885, for a third
    # of the build time. The dual dimension is what matters here and 497 is ample.
    hf = MembraneHF(n=36)
    params = training_grid()
    U, L = [], []
    for mu in params:
        u, lam = hf.solve(mu)
        U.append(u)
        L.append(lam)
    S = np.asarray(L, float).T
    idx = np.arange(S.shape[1])
    test_idx = idx[::5]
    return Dataset(
        name="membrane_2d",
        snapshots=S,
        description="2-D membrane obstacle, 5x5x5 parameter grid",
        paper="greedy_algos contact_forces",
        params=params,
        train_idx=np.setdiff1d(idx, test_idx),
        test_idx=test_idx,
        primal_snapshots=np.asarray(U, float).T,
        # Contact nodes are scattered inside the obstacle disc, not on a tensor grid.
        geometry=geometry.scatter_geometry(hf.cnode_coords),
    )


def _hertz_2d() -> Dataset:
    """2-D elastic Hertz contact between quarter-disks -- the [NDEE22] §5.2 analogue.

    The contact patch moves with the radius parameter, so the active set genuinely
    changes across snapshots. The most expensive source in the merge: one 2-D
    elasticity solve per parameter.
    """
    from greedy.synthetic_data.contact_forces.hertz_hf import HertzHF, training_grid

    # The module's own 18x60 mesh, matching [NDEE22] §5.2's discretization, at 26 dual
    # dofs. A coarser mesh saves no meaningful time and drops to 14 dofs, too few to
    # discriminate between cones.
    hf = HertzHF(nr=18, na=60)
    params = training_grid()[::2]         # 41 of the 81 paper parameters
    L = []
    arc_x = None
    for mu in params:
        lam, x, _gap = hf.solve(mu)
        arc_x = x
        L.append(lam)
    S = np.asarray(L, float).T
    idx = np.arange(S.shape[1])
    test_idx = idx[::4]
    return Dataset(
        name="hertz_2d",
        snapshots=S,
        description="2-D elastic Hertz contact, moving contact patch",
        paper="greedy_algos contact_forces",
        params=params.reshape(-1, 1),
        train_idx=np.setdiff1d(idx, test_idx),
        test_idx=test_idx,
        # Genuinely 1-D contact, but along an arc: plot against the physical abscissa,
        # not the node index.
        geometry=geometry.line_geometry(arc_x, xlabel="contact abscissa $x$"),
    )


DATASETS: dict[str, DatasetSpec] = {
    s.key: s
    for s in (
        DatasetSpec("toy_bee20", "1-D obstacle (B = I)", _toy_bee20, "fast",
                    "[BEE20] toy; supports inf-sup, parameter-independent B"),
        DatasetSpec("obstacle_ndee22", "1-D obstacle (B(mu))", _obstacle_ndee22, "fast",
                    "[NDEE22] toy; the only parameter-dependent constraint here"),
        DatasetSpec("hertz_pressure", "Hertz pressures (1-D)", _hertz_pressure, "fast",
                    "validated physics, 4000 samples"),
        DatasetSpec("gaussian_synth", "Gaussian bumps", _gaussian_synth, "fast",
                    "known generating rank"),
        DatasetSpec("fem_lambda", "Half-disks of Hertz", _fem_lambda, "fast",
                    "[BEE20] §6.2 half-disk, as stored"),
        DatasetSpec("fem_lambda_pressure", "Half-disks of Hertz (pressure)", _fem_lambda_pressure,
                    "fast", "same, symmetry-node half-support corrected"),
        DatasetSpec("physics", "3D Pellet-Cladding", _physics, "fast",
                    "7676 dofs; dimension scaling"),
        DatasetSpec("membrane_2d", "2-D membrane", _membrane_2d, "heavy",
                    "497 dual dofs; needs cvxopt"),
        DatasetSpec("hertz_2d", "2-D Hertz contact", _hertz_2d, "heavy",
                    "moving contact patch; needs cvxopt"),
    )
}

FAST = tuple(k for k, s in DATASETS.items() if s.tier == "fast")
HEAVY = tuple(k for k, s in DATASETS.items() if s.tier == "heavy")


@functools.lru_cache(maxsize=None)
def load(key: str) -> Dataset:
    """Build a dataset, caching it: several sources cost real FEM solves."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; known: {sorted(DATASETS)}")
    return DATASETS[key].build()


__all__ = ["DATASETS", "DatasetSpec", "FAST", "HEAVY", "load"]
