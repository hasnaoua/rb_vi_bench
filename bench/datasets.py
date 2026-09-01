"""The dataset registry: every source of non-negative dual snapshots in the merge.

Three sources, all contact problems on real geometry:

=================== ======================== ========= ===============================
key (``--datasets``) reported name             dim x n   why it is here
=================== ======================== ========= ===============================
fem_lambda          Half-disks of Hertz        57 x 50 [BEE20] §6.2, real FEM multipliers
physics             3D Pellet-Cladding       7676 x 94 76x101 quarter sector; dimension
membrane_2d         membrane_2d               497 x125 2-D, [NDEE22] §5.1 analogue
=================== ======================== ========= ===============================

The registry was **narrowed to these three deliberately**; it previously carried six
more -- two 1-D obstacle toys, two synthetic sweeps and two further contact sets. One
consequence is load-bearing and is not obvious from the table: the two obstacle toys
were the only sources that shipped a stiffness matrix ``A`` and a constraint operator
``B(mu)``, so with them gone **no dataset here can drive a reduced solve**. The inf-sup
family (``metrics.stability``) and the solved-error metric (``metrics.online``) are
still implemented and still generic -- any ``Dataset`` given ``A``, ``B_of_mu``,
``rhs_of_mu``, ``gap_of_mu`` and primal snapshots activates them -- but nothing in this
registry does, and their tests build such a problem themselves rather than depend on it.

**Keys versus names.** The registry key is the CLI handle and never changes; the
``Dataset.name`` is what appears in ``grid.csv``, figure titles and directory names.
Where a source *is* a recognised physical problem, the name says so -- ``fem_lambda`` is
[BEE20] §6.2's Hertz half-disks and ``physics`` is the 3-D pellet-cladding contact --
because a reader of a figure should see the problem, not the file it came from.

**Tiers.** ``fast`` sources need only numpy/scipy and are the default grid. ``heavy``
is ``membrane_2d`` alone, opt-in for a dependency reason rather than a runtime one: it
imports ``cvxopt`` at module scope, which is ``greedy_algos``' optional ``[qp]`` extra,
so the default grid stays runnable on a bare install. Its build cost is modest (roughly
30s). Install with::

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

# ---------------------------------------------------------------------------
# [NDEE22] source
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# greedy_algos sources
# ---------------------------------------------------------------------------

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


def _load_cladding_split() -> dict[str, np.ndarray]:
    """Read the pellet-cladding archive and check it is internally consistent.

    Four of the eight files are redundant: ``Contact_forces_train-data.txt`` and its test
    counterpart are column selections of the full matrix, and the two ``*_params_set``
    files are the same selections of ``Params_set.txt``. They are read and verified rather
    than ignored, because a split that disagrees with the matrix it indexes is the one
    failure mode that would silently produce a *plausible* test error -- fitted on some
    columns, scored on others, with no symptom anywhere downstream.
    """
    root = _paths.CLADDING_SPLIT
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found; unpack data/3D_cladding_split.zip beside it, or point "
            f"RB_VI_CLADDING_SPLIT at a copy of it -- see data/README.md"
        )

    def _read(name: str) -> np.ndarray:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing from the pellet-cladding archive")
        return np.loadtxt(path)

    raw = _read("Contact_forces_data.txt")                 # (dim, n), nodes x snapshots
    params = np.atleast_1d(_read("Params_set.txt"))
    train_idx = np.atleast_1d(_read("Training_indices_set.txt")).astype(int)
    test_idx = np.atleast_1d(_read("Ptest_indices_set.txt")).astype(int)

    if raw.ndim != 2 or raw.shape[1] != params.size:
        raise ValueError(
            f"pellet-cladding matrix {raw.shape} does not match {params.size} parameters"
        )
    both = np.concatenate([train_idx, test_idx])
    if not np.array_equal(np.sort(both), np.arange(params.size)):
        raise ValueError(
            "the shipped train/test indices are not a partition of the "
            f"{params.size} columns"
        )
    for name, idx, expected in (
        ("Contact_forces_train-data.txt", train_idx, "Training_params_set.txt"),
        ("Contact_forces_test-data.txt", test_idx, "Ptest_params_set.txt"),
    ):
        if not np.array_equal(_read(name), raw[:, idx]):
            raise ValueError(f"{name} is not the indexed columns of the full matrix")
        if not np.allclose(np.atleast_1d(_read(expected)), params[idx]):
            raise ValueError(f"{expected} does not match the indexed parameters")

    return {"snapshots": raw, "params": params,
            "train_idx": train_idx, "test_idx": test_idx}


def _physics() -> Dataset:
    """High-dimensional physics contact forces: 7676 dofs, 99 parameters, split 50/49.

    The only source where ``dim >> n``, so it is what exposes how each method's cost
    scales with the ambient dimension rather than with the training-set size.

    Read from ``data/3D_cladding_split``, which carries the parameter values and the
    train/test partition that ``greedy_algos/data/physics_data.txt`` does not -- see
    ``PHYSICS_PARAMETER_GRID`` for what that corrects. The split is the archive's own:
    alternating columns, training on the even ones. That is a deliberately *dense*
    interleave rather than a held-out tail, and it is the right one here: the parameter
    sweeps monotonically from no contact to full contact, so a contiguous test block would
    be asking every method to extrapolate past its training range instead of to
    interpolate within it.

    Superseding the npz path is intentional. The old builder produced the same snapshots
    under a shifted parameter axis and with no split at all, so keeping it as a fallback
    would have made the reported test error depend on which files happened to be on disk.
    """
    archive = _load_cladding_split()
    S = np.ascontiguousarray(np.asarray(archive["snapshots"], float))
    return Dataset(
        name="3D Pellet-Cladding",
        snapshots=S,
        description="pellet-cladding contact, 76x101 quarter sector, 50/49 train/test split",
        paper="3D_cladding_split archive",
        params=np.asarray(archive["params"], float).reshape(-1, 1),
        # The five no-contact columns at the low end of the sweep are dropped by
        # ``Dataset``, which remaps both index arrays onto the surviving columns -- so
        # these are the archive's indices, into the archive's numbering, as shipped.
        train_idx=np.asarray(archive["train_idx"], int),
        test_idx=np.asarray(archive["test_idx"], int),
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
    S: np.ndarray = np.asarray(L, float).T
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


DATASETS: dict[str, DatasetSpec] = {
    s.key: s
    for s in (
        DatasetSpec("fem_lambda", "Half-disks of Hertz", _fem_lambda, "fast",
                    "[BEE20] §6.2 half-disk, as stored"),
        DatasetSpec("physics", "3D Pellet-Cladding", _physics, "fast",
                    "7676 dofs; dimension scaling; shipped 50/49 split"),
        DatasetSpec("membrane_2d", "2-D membrane", _membrane_2d, "heavy",
                    "497 dual dofs; needs cvxopt"),
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


__all__ = ["DATASETS", "DatasetSpec", "FAST", "HEAVY", "PHYSICS_PARAMETER_GRID",
           "load"]
