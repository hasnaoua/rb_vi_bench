"""rb_vi_bench -- a benchmark over the merged reduced-basis / cone-greedy repositories.

See the top-level README for what is merged, why the duplicate algorithm
transcriptions are deliberately retained, and what each metric means.
"""

from __future__ import annotations

from . import _paths  # noqa: F401  -- sys.path + headless matplotlib, must be first

__all__ = ["adapters", "datasets", "metrics", "runner", "types"]
