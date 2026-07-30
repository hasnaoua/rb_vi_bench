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
            label="ADG",
            fit=family_b.fit_greedy_adg,
            family="greedy.core",
            paper_tag="",
            description="Angular Defect Greedy; angle-based selection, in neither paper",
        ),
        Method(
            key="adg_norm",
            label="ADG (normalized)",
            fit=family_b.fit_greedy_adg_normalized,
            family="greedy.core",
            paper_tag="",
            description="ADG with per-snapshot relative stopping criterion",
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

__all__ = ["METHODS", "Method", "CROSS_FAMILY_PAIRS", "CONE_METHODS", "TOLERANCE_METHODS"]
