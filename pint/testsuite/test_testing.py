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


@pytest.mark.parametrize(
    ["first", "second", "error"],
    (
        pytest.param(float("inf"), float("-inf"), True, id="scalar-opposite-inf"),
        pytest.param(float("inf"), float("inf"), False, id="scalar-positive-inf"),
        pytest.param(float("-inf"), float("-inf"), False, id="scalar-negative-inf"),
        pytest.param(float("inf"), 1.0, True, id="scalar-inf-finite"),
        pytest.param(1.0, float("-inf"), True, id="scalar-finite-inf"),
        pytest.param(float("nan"), float("nan"), False, id="scalar-nan"),
        pytest.param(
            QuantityToBe.from_many(float("inf"), "m"),
            QuantityToBe.from_many(float("-inf"), "m"),
            True,
            id="quantity-opposite-inf",
        ),
        pytest.param(
            QuantityToBe.from_many(float("inf"), "m"),
            QuantityToBe.from_many(float("inf"), "km"),
            False,
            id="quantity-same-inf-converted",
        ),
        pytest.param(
            QuantityToBe.from_many(float("inf"), "m"),
            QuantityToBe.from_many(float("-inf"), "km"),
            True,
            id="quantity-opposite-inf-converted",
        ),
        pytest.param(
            np.array([float("inf"), 1.0]),
            np.array([float("-inf"), 1.0]),
            True,
            id="array-opposite-inf",
        ),
    ),
)
def test_assert_allclose_infinities(sess_registry, first, second, error):
    if isinstance(first, QuantityToBe):
        first = sess_registry.Quantity(*first)
    if isinstance(second, QuantityToBe):
        second = sess_registry.Quantity(*second)
    if error:
        with pytest.raises(AssertionError):
            testing.assert_allclose(first, second)
    else:
        testing.assert_allclose(first, second)
