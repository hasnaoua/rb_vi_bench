"""The method registry.

``supports_tolerance`` is not bookkeeping -- it encodes [BEE20] §5's central argument
against NMF ("the user does not specify an error tolerance but only the cardinality"),
so the runner can report the tolerance grid as *not applicable* to the baselines
instead of inventing a substitute. ``cone_method`` marks the methods that actually
build a cone in ``W^+``; POD does not, and is present only as a negative control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..types import BasisResult
from . import baselines, family_a, family_b


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    fit: Callable[..., BasisResult]
    family: str
    paper_tag: str
    supports_tolerance: bool = True
    cone_method: bool = True
    deterministic: bool = True
    description: str = ""
    #: Why this method cannot run in tolerance mode. Method-specific, because the reasons
    #: genuinely differ and one shared message would misattribute them.
    tolerance_note: str = "method is cardinality-only ([BEE20] §5)"


METHODS: dict[str, Method] = {
    m.key: m
    for m in (
        Method(
            key="cpg_bee20",
            label="CPG [BEE20] Alg. 2",
            fit=family_a.fit_bee20_cpg,
            family="rb_vi_common",
            paper_tag="[BEE20]",
            description="raw generators, absolute tolerance Eq. (58)",
        ),
        Method(
            key="cpg_ndee22",
            label="CPG [NDEE22] Rmk 4.3",
            fit=family_a.fit_ndee22_cpg,
            family="rb_vi_common",
            paper_tag="[NDEE22]",
            description="normalized generators, relative tolerance Eq. (13)",
        ),
        Method(
            key="mcpg_ndee22",
            label="mCPG [NDEE22] Alg. 2",
            fit=family_a.fit_ndee22_mcpg,
            family="rb_vi_common",
            paper_tag="[NDEE22]",
            description="cone-constrained residual generators; line 9 via SLSQP",
        ),
        Method(
            key="cpg_greedy",
            label="CPG (greedy.core)",
            fit=family_b.fit_greedy_cpg,
            family="greedy.core",
            paper_tag="[BEE20]",
            description="independent CPG implementation",
        ),
        Method(
            key="mcpg_greedy",
            label="mCPG (greedy.core)",
            fit=family_b.fit_greedy_mcpg,
            family="greedy.core",
            paper_tag="[NDEE22]",
            description="independent mCPG implementation",
        ),
        Method(
            key="adg",
            label="ADG (batch normalized)",
            fit=family_b.fit_greedy_adg,
            family="greedy.core",
            paper_tag="",
            description="Batch Normalized Angular-Defect Greedy on S_norm; the standard form",
        ),
        Method(
            key="adg_raw",
            label="ADG (un-normalized, non-standard)",
            fit=family_b.fit_greedy_adg_raw,
            family="greedy.core",
            paper_tag="",
            description="ADG with one shared absolute threshold; kept to show its cost",
        ),
        Method(
            key="adg_momentum",
            label="ADG (momentum stop)",
            fit=family_b.fit_greedy_adg_momentum,
            family="greedy.core",
            paper_tag="",
            description=("ADG stopped when |e(p)-e(p-1)|/e(p-1) <= eps, i.e. when a "
                         "round stops buying anything, rather than when the error "
                         "reaches a target"),
        ),
        Method(
            key="adg_k0",
            label="ADG (from $K_0=\\{0\\}$)",
            fit=family_b.fit_greedy_adg_k0,
            family="greedy.core",
            paper_tag="",
            description=("ABLATION of ADG's initialization: identical angular-defect "
                         "rule, batch admission, normalization and stopping, but seeded "
                         "the way CPG seeds -- K_0 = {0}, which collapses the first "
                         "selection to [BEE20] Eq. (56), argmax ||lambda|| on the "
                         "UNNORMALIZED snapshots, normalized and handed to the same loop "
                         "-- rather than the largest-mutual-angle pair. Any difference "
                         "from `adg` is attributable to the first step alone"),
        ),
        Method(
            key="nmf_s0",
            label="NMF (seed 0)",
            fit=baselines.fit_nmf,
            family="baseline",
            paper_tag="[BEE20]",
            supports_tolerance=False,
            deterministic=False,
            description="[BEE20] §6.4 Eq. (66)-(69); the comparison method",
        ),
        Method(
            key="nmf_s1",
            label="NMF (seed 1)",
            fit=baselines.fit_nmf_seed1,
            family="baseline",
            paper_tag="[BEE20]",
            supports_tolerance=False,
            deterministic=False,
            description="second seed, to expose NMF's non-determinism",
        ),
        Method(
            key="nmf_s2",
            label="NMF (seed 2)",
            fit=baselines.fit_nmf_seed2,
            family="baseline",
            paper_tag="[BEE20]",
            supports_tolerance=False,
            deterministic=False,
            description="third seed, to expose NMF's non-determinism",
        ),
        Method(
            key="orthant",
            label="orthant $W^+$ (naive baseline)",
            fit=baselines.fit_orthant,
            family="baseline",
            paper_tag="",
            supports_tolerance=False,
            description="span_+ of the R most active coordinates; = W^+ at R = dim",
            tolerance_note=(
                "orthant is reported at matched cardinality only: meeting a tolerance "
                "needs nearly every coordinate (R = 5001 at delta=0.5, 7351 at "
                "delta=0.01 on physics, dim 7676) -- itself the finding, but it makes "
                "every O(R) metric intractable. Those two R were measured before physics "
                "carried a split, on all 94 columns rather than the 47 it now trains on; "
                "re-measuring them means running mCPG to R = 5001, which is the very "
                "cost this skip exists to avoid"),
        ),
        Method(
            key="pod_control",
            label="POD (negative control)",
            fit=baselines.fit_pod,
            family="baseline",
            paper_tag="[BEE20]",
            supports_tolerance=False,
            cone_method=False,
            description="NEGATIVE CONTROL: error floor + expected sign violation",
        ),
    )
}

#: Methods implemented in both families, keyed by the algorithm they transcribe.
#: This is what ``metrics.agreement`` cross-checks; ADG is absent by construction.
CROSS_FAMILY_PAIRS: tuple[tuple[str, str], ...] = (
    ("cpg_bee20", "cpg_greedy"),
    ("cpg_ndee22", "cpg_greedy"),
    ("mcpg_ndee22", "mcpg_greedy"),
)

CONE_METHODS = tuple(k for k, m in METHODS.items() if m.cone_method)
TOLERANCE_METHODS = tuple(k for k, m in METHODS.items() if m.supports_tolerance)

#: One canonical implementation per algorithm -- the default for every reported output.
#:
#: Each is the transcription from the paper that *introduced* the algorithm: CPG from
#: [BEE20] Algorithm 2, mCPG from [NDEE22] Algorithm 2, ADG in its normalized (standard)
#: form on ``S_norm``, one NMF seed, and the POD negative control.
#:
#: The duplicates stay registered and reachable through ``--methods``; they are not
#: redundant, they are the input to ``metrics.agreement``, which is what makes the
#: merge's retained transcriptions checkable rather than merely asserted. Having
#: established what they show -- the two CPG families are bit-identical, the two mCPGs
#: build non-unique but accuracy-equivalent cones -- carrying all of them through every
#: table and figure only obscures the algorithm comparison. ``agreement.csv`` is
#: generated from ``CROSS_FAMILY_PAIRS`` regardless of this set, so the check keeps
#: running.
#:
#: **POD is deliberately absent.** It was carried as a negative control -- the
#: unattainable least-squares floor -- but it is not a dual basis at all: [BEE20] §5 rules
#: it out because its mixed-sign modes cannot build ``W_R^+``. Scoring it beside methods
#: that must respect ``lambda >= 0`` compares different problems, and its presence in
#: every figure cost an axis range without informing the comparison. It stays registered
#: and reachable via ``--methods`` for the sign-violation checks in the test suite.
#:
#: Its place is taken by ``orthant``, which is the reference that *is* pertinent: the
#: naive admissible basis, and at ``R = dim`` the whole positive orthant -- the largest
#: cone any of these methods is allowed to build.
#:
#: NMF's other seeds likewise remain available: its across-seed spread is the
#: non-determinism [BEE20] §5 raises against it, measured by ``stability.determinism``.
DEFAULT_METHODS: tuple[str, ...] = (
    "cpg_bee20",
    "mcpg_ndee22",
    "adg",
    "adg_momentum",
    "adg_k0",
    "nmf_s0",
    "orthant",
)

__all__ = ["METHODS", "Method", "CROSS_FAMILY_PAIRS", "CONE_METHODS",
           "TOLERANCE_METHODS", "DEFAULT_METHODS"]
