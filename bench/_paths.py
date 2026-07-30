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


def use_headless_matplotlib() -> None:
    """Select a non-interactive backend.

    Several modules in ``greedy.viz`` and ``greedy.core.reduction_common`` import
    ``matplotlib.pyplot`` at module scope and apply a publication style. The
    benchmark runs unattended, so the backend is fixed before any of that happens.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)


use_headless_matplotlib()
