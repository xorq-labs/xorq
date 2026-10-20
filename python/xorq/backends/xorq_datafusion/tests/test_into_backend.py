"""into_backend fan-out edge cases on xorq_datafusion.

The shapes live in ``xorq.backends.tests.into_backend_fanout``.
"""

from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.backends.tests.into_backend_fanout import *  # noqa: F403


@pytest.fixture
def target() -> xo.Backend:
    return xo.connect()
