"""Shared machinery for two papers on reduced-basis methods for variational
inequalities with contact constraints.

THE TWO PAPERS, AND WHY THE TAGS MATTER
---------------------------------------
This package holds code that was originally transcribed separately in
``~/rb_contact_cpg`` and ``~/stable_model_reduction_vi``. Both papers number
their own equations, sections and algorithms, and the numbers COLLIDE:

  * [BEE20] Eq. (57) is the CPG selection step;
    [NDEE22] Eq. (57) is something else entirely.
  * Both papers have an "Algorithm 1" and an "Algorithm 2", and they are
    different algorithms:
        [BEE20] Algorithm 1 = the online stage;  Algorithm 2 = CPG.
        [NDEE22] Algorithm 1 = PGA;              Algorithm 2 = mCPG.

Before consolidation each repository had exactly one paper, so a bare "Eq. (57)"
was unambiguous. It no longer is. **Every citation in this package therefore
carries a paper tag**, and a bare equation number anywhere in ``rb_vi_common``
should be treated as a bug.

    [BEE20]   A. Benaceur, A. Ern, V. Ehrlacher, "A reduced basis method for
              parametrized variational inequalities applied to contact
              mechanics". Published as IJNME 121(6):1170-1197, 2020.

              *** THE TRANSCRIPTION HERE WAS MADE AGAINST THE HAL PREPRINT
              hal-02081485v2 (submitted 27 Mar 2019), NOT against the published
              IJNME version and NOT against v3 (24 Sep 2019). Equation and
              algorithm numbering may differ between versions. Every [BEE20]
              number below means "v2 numbering". ***

              <https://hal.science/hal-02081485v2/file/contact.pdf>

    [NDEE22]  I. Niakh, G. Drouet, V. Ehrlacher, A. Ern, "Stable model reduction
              for linear variational inequalities with parameter-dependent
              constraints". ESAIM: M2AN (2022), doi 10.1051/m2an/2022077.
              Read from the rendered PDF; equations came through cleanly.

              [NDEE22] cites [BEE20] as its reference [9].

[BEE20] additionally carries a PDF-extraction hazard that survives into any code
derived from it: the source PDF uses a shifted Type-1 font encoding in which
delimiters map onto letters and two collisions are not mechanically resolvable
(``u`` is both a closing brace and the primal displacement; ``P`` is both the
membership symbol and the parameter set). Equations quoted in [BEE20]-tagged
docstrings were read by eye against surrounding prose. Verify any [BEE20]
equation you rely on against the PDF. [NDEE22] has no such hazard.

WHAT IS SHARED, AND WHAT IS NOT
-------------------------------
Shared here: the cone algorithms, the cone projection, the reduced-space and
inf-sup machinery, and PGA. These are paper contributions and are identical
wherever they are used.

NOT shared, and deliberately so: the test problems (``toy_problem.py``,
``hf_model.py``, ``contact_dataset.py``) and the drivers. Those are per-repository
substitutes for the papers' own FEM test cases, they reproduce no number in
either paper, and merging them would blur which paper a given result speaks to.

No official code exists for either paper. Both state that the authors used
FreeFem++ plus Python with ``cvxopt``; neither links a repository. Nothing here
derives from the authors' code.

Dependencies: numpy and scipy only.
"""

from __future__ import annotations

# Cone projection and the ||.||_Lambda norm -- [BEE20] §5
from .cone_projection import norm_Lambda, project_onto_cone

# The cone-building greedy algorithms
from .cone_greedy import (
    ConeResult,
    CPGResult,
    cone_projected_greedy,   # [BEE20] Algorithm 2
    cpg,                     # [NDEE22] Remark 4.3 baseline (= [BEE20] Alg. 2)
    e_orth,                  # [NDEE22] Eq. (41)
    mcpg,                    # [NDEE22] Algorithm 2  -- the primary algorithm
)

# Reduced spaces, supremizers, inf-sup constants -- [NDEE22] §2-§3
from .reduction import (
    Whitener,
    boundedness_c_S,
    inf_sup,
    inf_sup_hf,
    orth_union,
    pod,
    sigma_S,
    supremizer_space,
)

# PGA -- [NDEE22] Algorithm 1
from .pga import PGAResult, S_R_full, pga

__all__ = [
    "norm_Lambda", "project_onto_cone",
    "ConeResult", "CPGResult", "cone_projected_greedy", "cpg", "mcpg", "e_orth",
    "Whitener", "pod", "supremizer_space", "sigma_S", "orth_union",
    "inf_sup", "inf_sup_hf", "boundedness_c_S",
    "pga", "PGAResult", "S_R_full",
]
