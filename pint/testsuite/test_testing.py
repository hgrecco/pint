from __future__ import annotations

from typing import Any

import pytest

from .. import testing

np = pytest.importorskip("numpy")


class QuantityToBe(tuple[Any]):
    def from_many(*args):
        return QuantityToBe(args)


@pytest.mark.parametrize(
    ["first", "second", "error", "message"],
    (
        pytest.param(
            np.array([0, 1]), np.array([0, 1]), False, "", id="ndarray-None-None-equal"
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            1,
            True,
            "The first is not dimensionless",
            id="mixed1-int-not equal-equal",
        ),
        pytest.param(
            1,
            QuantityToBe.from_many(1, "m"),
            True,
            "The second is not dimensionless",
            id="mixed2-int-not equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            QuantityToBe.from_many(1, "m"),
            False,
            "",
            id="QuantityToBe.from_many-int-equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            QuantityToBe.from_many(1, "s"),
            True,
            "Units are not equal",
            id="QuantityToBe.from_many-int-equal-not equal",
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            QuantityToBe.from_many(2, "m"),
            True,
            "Magnitudes are not equal",
            id="QuantityToBe.from_many-int-not equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            QuantityToBe.from_many(2, "s"),
            True,
            "Units are not equal",
            id="QuantityToBe.from_many-int-not equal-not equal",
        ),
        pytest.param(
            QuantityToBe.from_many(1, "m"),
            QuantityToBe.from_many(float("nan"), "m"),
            True,
            "Magnitudes are not equal",
            id="QuantityToBe.from_many-float-not equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many([1, 2], "m"),
            QuantityToBe.from_many([1, 2], "m"),
            False,
            "",
            id="QuantityToBe.from_many-ndarray-equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many([1, 2], "m"),
            QuantityToBe.from_many([1, 2], "s"),
            True,
            "Units are not equal",
            id="QuantityToBe.from_many-ndarray-equal-not equal",
        ),
        pytest.param(
            QuantityToBe.from_many([1, 2], "m"),
            QuantityToBe.from_many([2, 2], "m"),
            True,
            "Magnitudes are not equal",
            id="QuantityToBe.from_many-ndarray-not equal-equal",
        ),
        pytest.param(
            QuantityToBe.from_many([1, 2], "m"),
            QuantityToBe.from_many([2, 2], "s"),
            True,
            "Units are not equal",
            id="QuantityToBe.from_many-ndarray-not equal-not equal",
        ),
    ),
)
def test_assert_equal(sess_registry, first, second, error, message):
    if isinstance(first, QuantityToBe):
        first = sess_registry.Quantity(*first)
    if isinstance(second, QuantityToBe):
        second = sess_registry.Quantity(*second)
    if error:
        with pytest.raises(AssertionError, match=message):
            testing.assert_equal(first, second)
    else:
        testing.assert_equal(first, second)
