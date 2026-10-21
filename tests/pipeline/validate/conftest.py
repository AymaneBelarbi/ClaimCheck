from collections.abc import Callable

import pytest

from pipeline.validate.base import ClaimContext
from tests.pipeline.validate.factories import make_ctx as _make_ctx


@pytest.fixture
def make_ctx() -> Callable[..., ClaimContext]:
    return _make_ctx
