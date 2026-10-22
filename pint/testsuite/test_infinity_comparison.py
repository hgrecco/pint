from __future__ import annotations

import pytest

from pint import Quantity
from pint.testing import assert_allclose


@pytest.mark.parametrize("quantified", [False, True])
@pytest.mark.parametrize(
    "first, second",
    [
        (float("inf"), float("-inf")),
        (float("-inf"), float("inf")),
        (float("inf"), 1),
        (1, float("-inf")),
    ],
)
def test_assert_allclose_rejects_unequal_infinities(first, second, quantified):
    if quantified:
        first, second = Quantity(first, "m"), Quantity(second, "m")
    with pytest.raises(AssertionError, match="different infinity"):
        assert_allclose(first, second, rtol=1, atol=1, msg="different infinity")


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_assert_allclose_accepts_matching_infinities(value):
    assert_allclose(value, value)
    assert_allclose(Quantity(value, "m"), Quantity(value, "cm"))
