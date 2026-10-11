from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, assert_type

import pytest

from pint import Quantity, UnitRegistry
from pint.facets.plain import (
    GenericPlainRegistry,
    PlainQuantity,
    PlainRegistry,
    PlainUnit,
)
from pint.registry_helpers import wraps
from pint.testsuite import helpers

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    from pint.compat import np


@helpers.requires_numpy
@pytest.mark.parametrize("use_unit", [False, True])
def test_wraps_quantity_indexing(sess_registry: UnitRegistry, use_unit: bool) -> None:
    ret = sess_registry.meter if use_unit else "meter"

    @sess_registry.wraps(ret, "meter")
    def duplicate(value: float) -> NDArray[np.float64]:
        return np.array([value, value], dtype=np.float64)

    result = duplicate(sess_registry.Quantity(100, "centimeter"))
    assert_type(result, Quantity)
    assert result[0] == sess_registry.Quantity(1, "meter")
    np.testing.assert_array_equal(result.magnitude, [1, 1])


@helpers.requires_numpy
def test_wraps_typed_registry(
    sess_registry: UnitRegistry[NDArray[np.float64]],
) -> None:
    @sess_registry.wraps("meter", "meter")
    def double(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * value

    result = double(sess_registry.Quantity(np.array([1.0, 2.0]), "meter"))
    assert_type(result, "Quantity[NDArray[np.float64]]")
    assert result[0] == sess_registry.Quantity(2, "meter")


def test_wraps_plain_registry(tiny_definition_file: Path) -> None:
    ureg = PlainRegistry(str(tiny_definition_file))

    @wraps(ureg, "meter", "meter")
    def double(value: float) -> float:
        return 2 * value

    result = double(ureg.Quantity(1, "meter"))
    assert_type(result, PlainQuantity)
    assert not hasattr(result, "__getitem__")
    assert result == ureg.Quantity(2, "meter")


class CustomQuantity(PlainQuantity):
    def label(self) -> str:
        return "custom"


class CustomRegistry(GenericPlainRegistry[CustomQuantity, PlainUnit]):
    Quantity = CustomQuantity
    Unit = PlainUnit


def test_wraps_custom_registry(tiny_definition_file: Path) -> None:
    ureg = CustomRegistry(str(tiny_definition_file))

    @wraps(ureg, ureg.meter, "meter")
    def double(value: float) -> float:
        return 2 * value

    result = double(ureg.Quantity(1, "meter"))
    assert_type(result, CustomQuantity)
    assert result.label() == "custom"
    assert result == ureg.Quantity(2, "meter")


def test_wraps_without_return_conversion(sess_registry: UnitRegistry) -> None:
    sentinel = object()

    @sess_registry.wraps(None, "meter")
    def unconverted(value: float) -> object:
        return sentinel

    result = unconverted(sess_registry.Quantity(1, "meter"))
    assert_type(result, object)
    assert result is sentinel


@pytest.mark.parametrize("use_tuple", [False, True])
def test_wraps_multiple_returns(sess_registry: UnitRegistry, use_tuple: bool) -> None:
    ret = ("meter", None) if use_tuple else ["meter", None]

    @sess_registry.wraps(ret, "meter")
    def pair(value: float) -> tuple[float, str]:
        return value, "unconverted"

    result = pair(sess_registry.Quantity(100, "centimeter"))
    assert_type(result, object)
    expected = [sess_registry.Quantity(1, "meter"), "unconverted"]
    if use_tuple:
        assert isinstance(result, tuple)
        assert result == tuple(expected)
    else:
        assert isinstance(result, list)
        assert result == expected
