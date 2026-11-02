from __future__ import annotations

from typing import assert_type

from pint import Quantity, UnitRegistry


def test_reflected_add_sub_return_quantity(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(2.0, "dimensionless")

    assert_type(1.0 + x, Quantity[float])
    assert_type(1.0 - x, Quantity[float])