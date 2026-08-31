"""Put the four merged repositories on ``sys.path``.

The monorepo keeps each source repository intact under ``repos/``, exactly as it
was before the merge, so that every citation, tolerance convention and
[UNSPECIFIED] marker stays attached to the paper it came from. Only ``greedy_algos``
is an installable package; the other three are run-from-``src`` trees. This module
makes them importable without touching a single line of their source.

Layout relied upon (and the reason for it)::

    repos/rb_vi_shared/                 the shared algorithm library
    repos/rb_contact_cpg/               [BEE20]
    repos/stable_model_reduction_vi/    [NDEE22]
    repos/greedy_algos/                 CPG/mCPG/ADG, pip-installed as ``greedy``

``rb_contact_cpg`` and ``stable_model_reduction_vi`` locate the shared library
through their own ``_shared_path.py``, which resolves ``parents[2] / "rb_vi_shared"``
-- i.e. ``src/`` -> repo -> the repo's parent directory. Because all four repos sit
under a common ``repos/`` parent here, that resolution keeps working unmodified and
lands on the monorepo's copy. Do not flatten ``repos/`` without setting
``RB_VI_SHARED``.

Both ``src`` trees ship a ``_shared_path.py``. They are functionally identical
(each locates ``rb_vi_common`` and inserts it on the path), so whichever one wins
the shadowing race is harmless -- but a bare ``import _shared_path`` is therefore
ambiguous and this package never relies on which one it gets.
"""

from __future__ import annotations

import os
import pathlib
import sys

# bench/ -> repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOS = ROOT / "repos"

RB_VI_SHARED = REPOS / "rb_vi_shared"
BEE20_SRC = REPOS / "rb_contact_cpg" / "src"
NDEE22_SRC = REPOS / "stable_model_reduction_vi" / "src"
GREEDY_ROOT = REPOS / "greedy_algos"

#: The 3-D pellet-cladding archive, shipped *with* its parameter values and its
#: train/test split -- unlike ``greedy_algos/data/physics_data.txt``, which is the same
#: 7676 x 99 matrix (byte-identical, and now a symlink to this one) with no parameters
#: attached at all.
#:
#: It lives under ``data/`` rather than inside a source repository because no source
#: repository reads it: ``bench.datasets`` does, and nothing else. That is the rule the
#: two data locations follow -- a dataset a vendored pipeline addresses by a relative
#: path stays inside that repo, everything else is here. Like every raw input in this
#: monorepo it is not versioned; see ``data/README.md``.
#:
#: Override with ``RB_VI_CLADDING_SPLIT`` to point at a copy held elsewhere.
CLADDING_SPLIT = pathlib.Path(
    os.environ.get("RB_VI_CLADDING_SPLIT", str(ROOT / "data" / "3D_cladding_split"))
)

# Pin the shared library explicitly rather than relying on the relative walk, so
# that the monorepo copy wins even if a stray ~/rb_vi_shared is still on disk.
os.environ.setdefault("RB_VI_SHARED", str(RB_VI_SHARED))

_MISSING = [p for p in (RB_VI_SHARED, BEE20_SRC, NDEE22_SRC, GREEDY_ROOT) if not p.is_dir()]
if _MISSING:
    raise ImportError(
        "the merged repositories are not where bench expects them:\n  "
        + "\n  ".join(str(p) for p in _MISSING)
        + "\nbench must be run from the rb_vi_bench monorepo root."
    )

for _p in (RB_VI_SHARED, BEE20_SRC, NDEE22_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


#: Where the benchmark's artefacts live. Defined once here rather than as
#: ``RESULTS = _paths.ROOT / "results"`` in each of the five entry points, which is how
#: it was: five identical lines that had to be kept in step by hand, in modules that
#: otherwise share nothing.
RESULTS = ROOT / "results"

#: The dense-cardinality sweep ``decrement`` needs for its ``R`` axis. It is a separate
#: run because the axis needs CONSECUTIVE cardinalities and the main grid does not carry
#: them; naming it here keeps the two commands' idea of where it lives in one place.
SWEEP_DENSE = RESULTS / "sweep_dense"


def use_headless_matplotlib() -> None:
    """Select a non-interactive backend.

    Several modules in ``greedy.viz`` and ``greedy.core.reduction_common`` import
    ``matplotlib.pyplot`` at module scope and apply a publication style. The
    benchmark runs unattended, so the backend is fixed before any of that happens.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)


use_headless_matplotlib()
