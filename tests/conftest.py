"""Fixtures shared by every test module.

``bumps`` is session-scoped rather than module-scoped. It was module-scoped when the
whole suite lived in one file, where the two are the same thing; now that the tests are
split across five modules, module scope would rebuild it five times. The dataset is
immutable in practice -- tests read it and fit against it -- so one instance is correct
as well as cheaper.
"""

from __future__ import annotations

import pytest

from _fixtures import make_bumps
from bench.types import Dataset


@pytest.fixture(scope="session")
def bumps() -> Dataset:
    return make_bumps()
