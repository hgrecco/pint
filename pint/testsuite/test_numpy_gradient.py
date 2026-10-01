"""Regression tests for the units of individual gradient components."""

import pytest

from pint.compat import np
from pint.testsuite import helpers

pytestmark = helpers.requires_numpy


@pytest.mark.parametrize("spacing_units", [("m", "m"), ("m", "s"), ("cm", "m")])
@pytest.mark.parametrize("coordinates", [False, True])
@pytest.mark.parametrize("field_unit, delta_unit", [("K", "K"), ("degC", "delta_degC")])
def test_gradient_per_axis_units(
    sess_registry, spacing_units, coordinates, field_unit, delta_unit
):
    q = sess_registry.Quantity
    values = np.arange(12).reshape(4, 3)
    spacing = ([0, 1, 3, 6], [0, 2, 5]) if coordinates else (2, 3)
    actual = np.gradient(
        q(values, field_unit),
        *(q(s, u) for s, u in zip(spacing, spacing_units)),
        edge_order=2,
    )
    expected = np.gradient(values, *spacing, edge_order=2)
    for component, magnitude, unit in zip(actual, expected, spacing_units):
        helpers.assert_quantity_equal(component, q(magnitude, f"{delta_unit}/{unit}"))
    if spacing_units[0] == spacing_units[1]:
        assert isinstance(actual, q)
        assert actual.shape == (2, 4, 3)
    else:
        assert type(actual) is type(expected)


@pytest.mark.parametrize("axis", [(1, 0), (-1, 0), (1,), -1])
def test_gradient_selected_axes(sess_registry, axis):
    q = sess_registry.Quantity
    values = np.arange(12).reshape(4, 3)
    axes = (axis,) if isinstance(axis, int) else axis
    coordinates = (q([0, 1, 3, 6], "m"), q([0, 2, 5], "s"))
    spacing = tuple(coordinates[index] for index in axes)
    actual = np.gradient(q(values, "K"), *spacing, axis=axis)
    expected = np.gradient(values, *(s.magnitude for s in spacing), axis=axis)
    if len(axes) == 1:
        helpers.assert_quantity_equal(actual, q(expected, "K") / spacing[0].units)
    else:
        for component, magnitude, coordinate in zip(actual, expected, spacing):
            helpers.assert_quantity_equal(
                component, q(magnitude, "K") / coordinate.units
            )
        assert type(actual) is type(expected)


@pytest.mark.parametrize("field_unit, delta_unit", [("m", "m"), ("degC", "delta_degC")])
@pytest.mark.parametrize("spacing_kind", ["default", "bare", "quantity"])
def test_gradient_default_and_shared_spacing(
    sess_registry, field_unit, delta_unit, spacing_kind
):
    q = sess_registry.Quantity
    values = np.arange(12).reshape(4, 3)
    spacing = () if spacing_kind == "default" else (2,)
    unit = delta_unit
    if spacing_kind == "quantity":
        spacing = (q(2, "s"),)
        unit += "/s"
    actual = np.gradient(q(values, field_unit), *spacing)
    expected = np.gradient(values, *(getattr(s, "magnitude", s) for s in spacing))
    for component, magnitude in zip(actual, expected):
        helpers.assert_quantity_equal(component, q(magnitude, unit))
    assert isinstance(actual, q)
    assert actual.shape == (2, 4, 3)


def test_gradient_bare_field(sess_registry):
    q = sess_registry.Quantity
    values = np.arange(12).reshape(4, 3)
    actual = np.gradient(values, q(2, "m"), q(3, "s"))
    expected = np.gradient(values, 2, 3)
    for component, magnitude, unit in zip(actual, expected, ("1/m", "1/s")):
        helpers.assert_quantity_equal(component, q(magnitude, unit))


@pytest.mark.parametrize("container", [list, tuple])
def test_gradient_quantity_sequence(sess_registry, container):
    q = sess_registry.Quantity
    values = [1, 3, 7]
    field = container(q(value, "m") for value in values)
    actual = np.gradient(field, q(2, "s"))
    helpers.assert_quantity_equal(actual, q(np.gradient(values, 2), "m/s"))


def test_gradient_one_dimensional(sess_registry):
    q = sess_registry.Quantity
    values = np.arange(4) ** 2
    actual = np.gradient(q(values, "K"), q([0, 1, 3, 6], "m"))
    helpers.assert_quantity_equal(actual, q(np.gradient(values, [0, 1, 3, 6]), "K/m"))


def test_gradient_empty_axes(sess_registry):
    values = np.ones((4, 3))
    actual = np.gradient(sess_registry.Quantity(values, "m"), axis=())
    assert actual.units == sess_registry.m
    np.testing.assert_array_equal(actual.magnitude, np.gradient(values, axis=()))


def test_gradient_invalid_coordinates(sess_registry):
    q = sess_registry.Quantity
    values = np.ones((4, 3))
    with pytest.raises(ValueError, match="distances must match"):
        np.gradient(q(values, "K"), q([0, 1], "m"), axis=0)
