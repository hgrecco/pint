import pytest

from pint.compat import np
from pint.testsuite import helpers


@pytest.mark.parametrize("data_unit", [None, "dimensionless", "meter"])
@pytest.mark.parametrize(
    "weight_unit", [None, "dimensionless", "percent", "meter / centimeter"]
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@helpers.requires_numpy
def test_average_returned_outputs(sess_registry, data_unit, weight_unit, axis):
    values = np.arange(6.0).reshape(2, 3)
    original_values = values.copy()
    data = values if data_unit is None else sess_registry.Quantity(values, data_unit)
    weights = np.arange(1.0, 7.0).reshape(2, 3)
    if weight_unit is not None:
        weights = sess_registry.Quantity(weights, weight_unit)
        reference_weights = weights.m_as("")
    else:
        reference_weights = weights
    expected_mean, expected_weights = np.average(
        values, weights=reference_weights, axis=axis, returned=True, keepdims=True
    )
    result = np.average(data, weights=weights, axis=axis, returned=True, keepdims=True)
    assert isinstance(result, tuple)
    actual_mean, actual_weights = result
    if data_unit is None:
        np.testing.assert_allclose(actual_mean, expected_mean)
    else:
        helpers.assert_quantity_almost_equal(
            actual_mean, sess_registry.Quantity(expected_mean, data_unit)
        )
    assert not hasattr(actual_weights, "units")
    np.testing.assert_allclose(actual_weights, expected_weights)
    np.testing.assert_array_equal(
        data.magnitude if data_unit else data, original_values
    )


@helpers.requires_numpy
def test_average_returned_default_weights_can_combine_units(sess_registry):
    first = sess_registry.Quantity([1.0, 2.0], "meter")
    second = sess_registry.Quantity([300.0, 400.0], "centimeter")
    first_mean, first_count = np.average(first, returned=True)
    second_mean, second_count = np.average(second, returned=True)
    assert not hasattr(first_count, "units")
    assert not hasattr(second_count, "units")
    combined = (first_mean * first_count + second_mean * second_count) / (
        first_count + second_count
    )
    helpers.assert_quantity_almost_equal(combined, sess_registry.Quantity(2.5, "meter"))
