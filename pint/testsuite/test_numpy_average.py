"""Regression tests for dimensionless Quantity weights in numpy.average."""

import pytest

from pint.compat import np
from pint.testsuite import helpers

pytestmark = pytest.mark.skipif(np is None, reason="NumPy is not available")


@pytest.mark.parametrize("unit", ["dimensionless", "percent", "meter / centimeter"])
@pytest.mark.parametrize("data_unit", [None, "dimensionless", "meter"])
@pytest.mark.parametrize("positional", [False, True])
def test_average_dimensionless_weights(sess_registry, unit, data_unit, positional):
    values = np.arange(6.0).reshape(2, 3)
    data = values if data_unit is None else sess_registry.Quantity(values, data_unit)
    weights = sess_registry.Quantity([1.0, 2.0, 3.0], unit)
    original = weights.copy()
    expected = np.average(values, axis=1, weights=weights.m_as(""), keepdims=True)
    if positional:
        result = np.average(data, 1, weights, False, keepdims=True)
    else:
        result = np.average(a=data, axis=1, weights=weights, keepdims=True)
    if data_unit is None:
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected)
    else:
        assert result.units == data.units
        helpers.assert_quantity_almost_equal(
            result, sess_registry.Quantity(expected, data_unit)
        )
    assert result.shape == (2, 1)
    assert weights.units == original.units
    np.testing.assert_array_equal(weights.magnitude, original.magnitude)


@pytest.mark.parametrize("weights", [None, [1.0, 2.0, 3.0]])
def test_average_plain_weights(sess_registry, weights):
    data = sess_registry.Quantity([1.0, 2.0, 4.0], "meter")
    result = np.average(data, weights=weights)
    helpers.assert_quantity_almost_equal(
        result,
        sess_registry.Quantity(np.average(data.magnitude, weights=weights), "meter"),
    )


@pytest.mark.parametrize("axis", [None, (0, 1)])
def test_average_full_shape_weights(sess_registry, axis):
    values = np.arange(6.0).reshape(2, 3)
    weights = sess_registry.Quantity([[1, 2, 3], [4, 5, 6]], "percent")
    result = np.average(values * sess_registry.meter, axis=axis, weights=weights)
    expected = np.average(values, axis=axis, weights=weights.m_as(""))
    helpers.assert_quantity_almost_equal(result, expected * sess_registry.meter)


@pytest.mark.parametrize(
    "weights, error", [([0, 0, 0], ZeroDivisionError), ([1, 2], ValueError)]
)
def test_average_weight_validation(sess_registry, weights, error):
    data = sess_registry.Quantity(np.arange(6).reshape(2, 3), "meter")
    with pytest.raises(error):
        np.average(data, axis=1, weights=sess_registry.Quantity(weights, "percent"))
