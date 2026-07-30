"""Greedy model-order-reduction methods (CPG, mCPG, ADG) and experiment pipelines."""
from __future__ import annotations

import os

# Matplotlib setup for the whole package, in the one place guaranteed to run
# first: importing any greedy.* submodule initializes this package before the
# submodule itself, so this lands before matplotlib is ever imported.
#
# Both lines must precede `import matplotlib.pyplot`, which is why this is not
# deferred into a function. MPLCONFIGDIR is only read when matplotlib is first
# imported, and without it an absent or read-only HOME makes matplotlib stall
# building a font cache. Every entry point here writes files rather than opening
# windows, so the non-interactive Agg backend is always the right one.
#
# This block used to be repeated verbatim at the top of 16 modules.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
