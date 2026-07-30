"""Locate the shared ``rb_vi_common`` package.

The cone algorithms, the cone projection, PGA, and the reduction machinery live
in ``~/rb_vi_shared`` and are shared with ``~/rb_contact_cpg``, which implements
this paper's reference [9]. See that package's README for the [BEE20] / [NDEE22]
tag key and, in particular, for the reason every citation there carries a paper
tag: the two papers' equation and algorithm numbers collide.

Nothing needs installing. Importing this module puts the shared package on
``sys.path``, so ``cd src && python3 run_experiments.py`` keeps working as before.

Default layout::

    ~/rb_vi_shared/              the shared package
    ~/stable_model_reduction_vi/ this repository
    ~/rb_contact_cpg/            the repository for reference [9]

Set ``RB_VI_SHARED`` to override the location.
"""

from __future__ import annotations

import os
import pathlib
import sys

_env = os.environ.get("RB_VI_SHARED")
_candidates = [pathlib.Path(_env)] if _env else []
# ../../rb_vi_shared relative to this file (src/ -> repo/ -> parent/)
_candidates.append(pathlib.Path(__file__).resolve().parents[2] / "rb_vi_shared")

for _p in _candidates:
    if (_p / "rb_vi_common" / "__init__.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
else:
    raise ImportError(
        "cannot find the shared package rb_vi_common. Looked in:\n  "
        + "\n  ".join(str(p) for p in _candidates)
        + "\nSet RB_VI_SHARED to the directory CONTAINING rb_vi_common/."
    )
