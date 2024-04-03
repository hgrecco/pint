from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from pint import UnitRegistry
from pint.testsuite import helpers
from pint.util import infer_base_unit


def test_infer_base_unit(sess_registry):
    test_units = sess_registry.Quantity(1, "meter**2").units
    registry = sess_registry

    assert (
        infer_base_unit(sess_registry.Quantity(1, "millimeter * nanometer"))
        == test_units
    )

    assert infer_base_unit("millimeter * nanometer", registry) == test_units

    assert (
        infer_base_unit(
            sess_registry.Quantity(1, "millimeter * nanometer").units, registry
        )
        == test_units
    )

    with pytest.raises(ValueError, match=r"No registry provided."):
        infer_base_unit("millimeter")


def test_infer_base_unit_decimal(sess_registry):
    ureg = UnitRegistry(non_int_type=Decimal)
    QD = ureg.Quantity

    ibu_d = infer_base_unit(QD(Decimal(1), "millimeter * nanometer"))

    assert ibu_d == QD(Decimal(1), "meter**2").units

    assert all(isinstance(v, Decimal) for v in ibu_d.values())


def test_infer_base_unit_fraction(sess_registry):
    ureg = UnitRegistry(non_int_type=Fraction)
    QD = ureg.Quantity

    ibu_d = infer_base_unit(QD(Fraction("1"), "millimeter * nanometer"))

    assert ibu_d == QD(Fraction("1"), "meter**2").units

    assert all(isinstance(v, Fraction) for v in ibu_d.values())


def test_units_adding_to_zero(sess_registry):
    assert (
        infer_base_unit(sess_registry.Quantity(1, "m * mm / m / um * s"))
        == sess_registry.Quantity(1, "s").units
    )


def test_to_compact(sess_registry):
    r = (
        sess_registry.Quantity(1000000000, "m")
        * sess_registry.Quantity(1, "mm")
        / sess_registry.Quantity(1, "s")
        / sess_registry.Quantity(1, "ms")
    )
    compact_r = r.to_compact()
    expected = sess_registry.Quantity(1000.0, "kilometer**2 / second**2")
    helpers.assert_quantity_almost_equal(compact_r, expected)

    r = (
        sess_registry.Quantity(1, "m")
        * sess_registry.Quantity(1, "mm")
        / sess_registry.Quantity(1, "m")
        / sess_registry.Quantity(2, "um")
        * sess_registry.Quantity(2, "s")
    ).to_compact()
    helpers.assert_quantity_almost_equal(r, sess_registry.Quantity(1000, "s"))


def test_to_compact_decimal(sess_registry):
    ureg = UnitRegistry(non_int_type=Decimal)
    Q = ureg.Quantity
    r = (
        Q(Decimal("1000000000.0"), "m")
        * Q(Decimal(1), "mm")
        / Q(Decimal(1), "s")
        / Q(Decimal(1), "ms")
    )
    compact_r = r.to_compact()
    expected = Q(Decimal("1000.0"), "kilometer**2 / second**2")
    assert compact_r == expected

    r = (
        Q(Decimal(1), "m") * Q(1, "mm") / Q(1, "m**2") / Q(2, "um") * Q(2, "s")
    ).to_compact()
    assert r == Q(1000, "s/m")


def test_to_compact_fraction(sess_registry):
    ureg = UnitRegistry(non_int_type=Fraction)
    Q = ureg.Quantity
    r = (
        Q(Fraction("10000000000/10"), "m")
        * Q(Fraction("1"), "mm")
        / Q(Fraction("1"), "s")
        / Q(Fraction("1"), "ms")
    )
    compact_r = r.to_compact()
    expected = Q(Fraction("1000.0"), "kilometer**2 / second**2")
    assert compact_r == expected

    r = (
        sess_registry.Quantity(Fraction(1), "m")
        * sess_registry.Quantity(1, "mm")
        / sess_registry.Quantity(1, "m**2")
        / sess_registry.Quantity(2, "um")
        * sess_registry.Quantity(2, "s")
    ).to_compact()
    assert r == Q(1000, "s/m")


def test_volts(sess_registry):
    r = (
        sess_registry.Quantity(1, "V")
        * sess_registry.Quantity(1, "mV")
        / sess_registry.Quantity(1, "kV")
    )
    b = infer_base_unit(r)
    assert b == sess_registry.Quantity(1, "V").units
    helpers.assert_quantity_almost_equal(r, sess_registry.Quantity(1, "uV"))
