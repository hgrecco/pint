import pytest

from pint import Quantity

from .. import testing

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    ["first", "second", "error", "message"],
    (
        pytest.param(
            np.array([0, 1]), np.array([0, 1]), False, "", id="ndarray-None-None-equal"
        ),
        pytest.param(
            Quantity(1, "m"),
            1,
            True,
            "The first is not dimensionless",
            id="mixed1-int-not equal-equal",
        ),
        pytest.param(
            1,
            Quantity(1, "m"),
            True,
            "The second is not dimensionless",
            id="mixed2-int-not equal-equal",
        ),
        pytest.param(
            Quantity(1, "m"), Quantity(1, "m"), False, "", id="Quantity-int-equal-equal"
        ),
        pytest.param(
            Quantity(1, "m"),
            Quantity(1, "s"),
            True,
            "Units are not equal",
            id="Quantity-int-equal-not equal",
        ),
        pytest.param(
            Quantity(1, "m"),
            Quantity(2, "m"),
            True,
            "Magnitudes are not equal",
            id="Quantity-int-not equal-equal",
        ),
        pytest.param(
            Quantity(1, "m"),
            Quantity(2, "s"),
            True,
            "Units are not equal",
            id="Quantity-int-not equal-not equal",
        ),
        pytest.param(
            Quantity(1, "m"),
            Quantity(float("nan"), "m"),
            True,
            "Magnitudes are not equal",
            id="Quantity-float-not equal-equal",
        ),
        pytest.param(
            Quantity([1, 2], "m"),
            Quantity([1, 2], "m"),
            False,
            "",
            id="Quantity-ndarray-equal-equal",
        ),
        pytest.param(
            Quantity([1, 2], "m"),
            Quantity([1, 2], "s"),
            True,
            "Units are not equal",
            id="Quantity-ndarray-equal-not equal",
        ),
        pytest.param(
            Quantity([1, 2], "m"),
            Quantity([2, 2], "m"),
            True,
            "Magnitudes are not equal",
            id="Quantity-ndarray-not equal-equal",
        ),
        pytest.param(
            Quantity([1, 2], "m"),
            Quantity([2, 2], "s"),
            True,
            "Units are not equal",
            id="Quantity-ndarray-not equal-not equal",
        ),
    ),
)
def test_assert_equal(first, second, error, message):
    if error:
        with pytest.raises(AssertionError, match=message):
            testing.assert_equal(first, second)
    else:
        testing.assert_equal(first, second)
