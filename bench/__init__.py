"""rb_vi_bench -- a benchmark over the merged reduced-basis / cone-greedy repositories.

See the top-level README for what is merged, why the duplicate algorithm
transcriptions are deliberately retained, and what each metric means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _paths  # noqa: F401  -- sys.path + headless matplotlib, must be first

if TYPE_CHECKING:  # `from bench import *` imports these submodules at runtime; naming
    # them here as well lets a type checker see what ``__all__`` refers to, without
    # eagerly importing the matplotlib-heavy modules on every ``import bench``.
    from . import adapters, datasets, metrics, runner, types

__all__ = ["adapters", "datasets", "metrics", "runner", "types"]
