from __future__ import annotations

from typing import Any, assert_type

from pint import Quantity, UnitRegistry


def test_round(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(1.1, "kg")
    round1 = assert_type(round(x), Quantity[int])
    round2 = assert_type(round(x, 0), Quantity[float])
    assert isinstance(round1.magnitude, int)
    assert isinstance(round2.magnitude, float)


def test_unary(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(-4.2, "meter")
    c: Quantity[complex] = sess_registry.Quantity(3 + 2j, "meter")

    assert_type(abs(x), Quantity[float])
    assert_type(abs(c), Quantity[float])
    assert_type(round(x), Quantity[int])
    assert_type(+x, Quantity[float])
    assert_type(+c, Quantity[complex])
    assert_type(-x, Quantity[float])
    assert_type(-c, Quantity[complex])


def test_binary(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(1.0)
    y: Quantity[float] = sess_registry.Quantity(1.0)

    assert_type(x + y, Quantity[float])
    assert_type(x - y, Quantity[float])
    assert_type(x * y, Quantity[float])
    assert_type(x / y, Quantity[float])
    assert_type(x // y, Quantity[float])
    assert_type(x % y, Quantity[float])
    assert_type(x**y, Quantity[Any])  # NOTE: This is a known problem in typeshed
    assert_type(divmod(x, y), tuple[Quantity[float], Quantity[float]])


def test_reflected(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(2.0)

    assert_type(1.0 + x, Quantity[float])
    assert_type(1.0 - x, Quantity[float])


def test_inplace(sess_registry: UnitRegistry) -> None:
    x: Quantity[float] = sess_registry.Quantity(1.0)
    y: Quantity[float] = sess_registry.Quantity(1.0)

    x += y
    x -= y
    x *= y
    x /= y
    x //= y
    x %= y
    x **= y
