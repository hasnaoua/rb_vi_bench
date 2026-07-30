"""The two data contracts the benchmark is built on.

Both exist to make the two implementation families comparable *without* editing
either of them, which is the whole design constraint of this benchmark: the
duplicate CPG/mCPG transcriptions are deliberately retained (see the top-level
README), so the harness has to absorb their convention differences instead of
removing them.

Two conventions are normalized here, and getting either wrong silently corrupts
every number downstream:

**Orientation.** ``rb_vi_common`` and the NMF baseline take snapshots as
*columns*, shape ``(dim, n_snapshots)``. ``greedy.core`` takes them as *rows*,
shape ``(n_snapshots, dim)``. The benchmark's canonical form is **columns**, and
the ``greedy.core`` adapters transpose on the way in and back on the way out.

**Tolerance.** Three different meanings are in play:

* ``greedy.core`` -- ``epsilon`` is relative, applied as ``epsilon * max_q ||theta_q||``.
* ``rb_vi_common.cpg`` / ``mcpg`` -- ``delta`` is relative, [NDEE22] Eq. (13),
  normalized by ``max_p ||lambda(mu_p)||``.
* ``rb_vi_common.cone_projected_greedy`` -- ``eps_du`` is **absolute**,
  [BEE20] Eq. (58).

The first two use the same scale (the largest snapshot norm), so they are directly
comparable. The benchmark's canonical knob is that relative ``delta``, and the
[BEE20] adapter converts it to an absolute tolerance by multiplying through by the
snapshot scale. This is the conversion [BEE20]'s own REPRODUCTION_NOTES flags as
necessary when reading its tolerance "against the snapshot scale".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class BasisResult:
    """One fitted dual cone, in canonical (column) orientation.

    ``generators`` is what every precision and stability metric consumes, so the
    adapters must return it on the *physical* snapshot scale for the two [BEE20]
    conventions and normalized for the [NDEE22] ones -- the difference is real and
    is itself a measured quantity (see ``metrics.agreement``), not something to
    paper over.
    """

    method: str
    family: str                      # "rb_vi_common" | "greedy.core" | "baseline"
    paper_tag: str                   # "[BEE20]" | "[NDEE22]" | "" for ADG / baselines
    generators: np.ndarray           # (dim, R)
    R: int
    selected_indices: list[int] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)   # per-iteration history
    fit_seconds: float = float("nan")
    solver_calls: dict[str, int] = field(default_factory=dict)
    normalized_generators: bool = False
    notes: str = ""

    @property
    def dim(self) -> int:
        return int(self.generators.shape[0])


@dataclass
class Dataset:
    """Dual snapshots plus whatever else a given source can honestly supply.

    ``snapshots`` is mandatory and must be non-negative: every method benchmarked
    here builds a cone meant to represent Lagrange multipliers of an inequality
    constraint, and ``cone_projected_greedy`` rejects negative input outright.

    The remaining fields are optional because the sources genuinely differ in what
    they expose, and the runner skips metrics rather than fabricating inputs:

    * ``A`` / ``B_of_mu`` / ``params`` -- needed by the inf-sup family
      ([NDEE22] Eq. 14/18/30). Only the HF models that expose a stiffness matrix
      and a constraint operator can support them.
    * ``primal_snapshots`` -- needed for the POD primal basis ``V_N``, hence for
      ``beta^dec``.
    * ``train_idx`` / ``test_idx`` -- when absent, precision metrics report
      full-set residuals only, which is what ``greedy_algos``' physics pipeline
      already does deliberately.
    """

    name: str
    snapshots: np.ndarray                        # (dim, n) non-negative, columns
    description: str = ""
    paper: str = ""                              # provenance, not a citation tag
    params: np.ndarray | None = None             # (n, n_params) parameter samples
    train_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    primal_snapshots: np.ndarray | None = None   # (dim_primal, n)
    A: np.ndarray | None = None                  # primal stiffness / energy matrix
    B_of_mu: Callable[[int], np.ndarray] | None = None   # index -> B(mu_i)
    mass: np.ndarray | None = None               # Gram matrix of ||.||_Lambda

    def __post_init__(self) -> None:
        S = np.asarray(self.snapshots, dtype=float)
        if S.ndim != 2:
            raise ValueError(f"{self.name}: snapshots must be 2-D (dim, n), got {S.shape}")
        if S.shape[1] == 0:
            raise ValueError(f"{self.name}: no snapshots")
        # Clip round-off negatives rather than hiding a real sign error: anything
        # beyond solver noise means the HF solve is wrong, and a cone built from it
        # cannot represent its own snapshots.
        smallest = float(S.min())
        if smallest < 0.0:
            tol = -1e-9 * max(1.0, float(np.abs(S).max()))
            if smallest < tol:
                raise ValueError(
                    f"{self.name}: snapshots contain negative entries (min={smallest:.3e}); "
                    "dual multipliers of an inequality constraint must be >= 0"
                )
            S = np.clip(S, 0.0, None)
        self.snapshots = S

    @property
    def dim(self) -> int:
        return int(self.snapshots.shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self.snapshots.shape[1])

    @property
    def scale(self) -> float:
        """``max_q ||theta_q||`` over the **training** set.

        Training, not all snapshots, and the distinction is not cosmetic. Both relative
        tolerances normalize by the largest norm among the snapshots they are *fitted
        to*: ``rb_vi_common.cpg`` computes ``denom = norms.max()`` over the matrix it is
        handed, and ``greedy.core.ConeGreedy._compute_stopping_scale`` does the same.
        The [BEE20] adapter has to reproduce that exact denominator when it converts the
        canonical relative ``delta`` into Eq. (58)'s absolute ``eps_du``.

        Using the full-set maximum instead silently inflates ``eps_du`` whenever the
        largest-norm snapshot lands in the test split -- by 4.7% on the bump fixture --
        which is enough to stop [BEE20]'s CPG one generator early and make two
        transcriptions of the same algorithm disagree for a reason that has nothing to
        do with either paper.

        Error columns are normalized by this same value, so a reported
        ``test_max_rel_err`` can be read directly against the ``delta`` that produced it.
        """
        s = float(np.max(np.linalg.norm(self.train(), axis=0)))
        return s if s > 0.0 else 1.0

    def train(self) -> np.ndarray:
        if self.train_idx is None:
            return self.snapshots
        return self.snapshots[:, self.train_idx]

    def test(self) -> np.ndarray | None:
        if self.test_idx is None or len(self.test_idx) == 0:
            return None
        return self.snapshots[:, self.test_idx]

    @property
    def has_split(self) -> bool:
        return self.test() is not None

    @property
    def supports_infsup(self) -> bool:
        return self.A is not None and self.B_of_mu is not None
