"""The four metric families.

``precision``   -- cone-projection error, at matched tolerance and matched cardinality.
``stability``   -- Gram conditioning, e_orth, determinism, and the inf-sup constants.
``performance`` -- solver-call counts, offline time, R at tolerance, online cost.
``agreement``   -- do the independent transcriptions build the same cone?
``cone_geometry`` -- how much of span_+{all snapshots} does K_R capture, and how
                  wide does it open?

Every function takes ``(dataset, result)`` and returns a flat ``dict[str, float]``, so
the runner can concatenate them into one tidy row per grid cell without knowing which
family produced which column.
"""

from __future__ import annotations

from . import agreement, cone_geometry, performance, precision, stability

__all__ = ["agreement", "cone_geometry", "performance", "precision", "stability"]
