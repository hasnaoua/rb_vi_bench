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


#: Snapshots whose norm falls below this fraction of the largest are dropped at
#: ``Dataset`` construction, before any algorithm sees them.
#:
#: They are not data, they are absence of data: a parameter value at which no contact
#: occurred, so ``lambda = 0`` everywhere. ``physics`` carries five -- the low end of its
#: displacement sweep, before the imposed displacement has closed the initial gap -- at
#: norm ~7e-67 against a typical 2e9. Keeping them corrupts every angle-based method,
#: because normalizing a zero vector is undefined and any per-snapshot relative criterion
#: sees an arbitrarily large relative error on a vector that carries no information. It
#: also violates the ADG spec's own precondition ``S subset R_+^m \ {0}``.
#:
#: The threshold sits at the numerical-zero scale rather than at a "physically small" one:
#: dropping genuinely small but non-zero contact states would be a modelling decision, not
#: numerical hygiene.
#:
#: Module-level on purpose. Annotated inside the ``@dataclass`` body it became a *field* --
#: a constructor parameter that let any caller silently change the filtering threshold for
#: one dataset, and an entry in ``asdict`` that had to be round-tripped.
ZERO_NORM_RTOL: float = 1e-8


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
      full-set residuals only. Every source in the merge now carries a split: the
      sources that ship one of their own (``fem_lambda``, ``physics``, ``hertz_pressure``)
      use it, and the rest get a deterministic stride. A dataset with no split is still
      a supported shape -- it is what the tests exercise the fallback with -- but it is
      no longer a state any registered source is in.
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
    #: The remaining data of the high-fidelity problem, needed to actually SOLVE the
    #: reduced system rather than only to score a cone against snapshots. Both are
    #: indexed by snapshot, because the load and the obstacle are what the parameter
    #: moves. Only sources that expose their HF assembly can supply them; without them
    #: ``metrics.online`` reports nothing rather than inventing a right-hand side.
    rhs_of_mu: Callable[[int], np.ndarray] | None = None   # index -> f(mu_i)
    gap_of_mu: Callable[[int], np.ndarray] | None = None   # index -> g(mu_i)
    mass: np.ndarray | None = None               # Gram matrix of ||.||_Lambda
    #: How a snapshot lays out in space (see ``bench.geometry``). Only affects how
    #: snapshots are *drawn*; every metric is basis-independent and ignores it. Absent
    #: means a 1-D contact set plotted against its component index -- which is wrong for
    #: any dataset whose contact nodes tile a surface, so it must be set wherever the
    #: source knows its own geometry.
    geometry: object | None = None
    #: How many numerically-zero snapshots were discarded at construction.
    n_dropped_zero: int = 0

    def __post_init__(self) -> None:
        S = np.asarray(self.snapshots, dtype=float)
        if S.ndim != 2:
            raise ValueError(f"{self.name}: snapshots must be 2-D (dim, n), got {S.shape}")
        if S.shape[1] == 0:
            raise ValueError(f"{self.name}: no snapshots")

        norms = np.linalg.norm(S, axis=0)
        scale = float(norms.max()) if norms.size else 0.0
        keep = norms > ZERO_NORM_RTOL * scale if scale > 0 else np.ones(S.shape[1], bool)
        if not keep.all():
            S = S[:, keep]
            if S.shape[1] == 0:
                raise ValueError(f"{self.name}: every snapshot is numerically zero")
            # Index-valued fields must be remapped onto the surviving columns, not merely
            # filtered: a stale index would silently point at a different snapshot.
            remap = {old: new for new, old in enumerate(np.flatnonzero(keep))}
            for attr in ("train_idx", "test_idx"):
                sel = getattr(self, attr)
                if sel is not None:
                    kept = [remap[int(i)] for i in sel if int(i) in remap]
                    object.__setattr__(self, attr,
                                       np.asarray(kept, int) if kept else None)
            if self.params is not None:
                object.__setattr__(self, "params", np.asarray(self.params)[keep])
            if self.primal_snapshots is not None:
                object.__setattr__(self, "primal_snapshots",
                                   np.asarray(self.primal_snapshots)[:, keep])
            object.__setattr__(self, "n_dropped_zero", int((~keep).sum()))
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

    @property
    def supports_online(self) -> bool:
        """Can the reduced saddle-point problem actually be solved for this dataset?

        Every other metric family scores a cone against snapshots, which needs only the
        snapshots. Solving [BEE20] Eq. (53) needs the problem itself: the operator, the
        load, the obstacle, and primal snapshots to build ``V_N`` from. Sources that
        ship only a snapshot matrix cannot support it, and must report nothing rather
        than a number derived from a fabricated right-hand side.
        """
        return (self.A is not None and self.B_of_mu is not None
                and self.rhs_of_mu is not None and self.gap_of_mu is not None
                and self.primal_snapshots is not None)
