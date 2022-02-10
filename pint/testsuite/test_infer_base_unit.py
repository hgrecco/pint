from decimal import Decimal
from fractions import Fraction

import pytest

from pint import Quantity as Q
from pint import UnitRegistry
from pint.testsuite import helpers
from pint.util import infer_base_unit


class TestInferBaseUnit:
    def test_infer_base_unit(self):
        from pint.util import infer_base_unit

        test_units = Q(1, "meter**2").units
        registry = Q(1, "meter**2")._REGISTRY

        assert infer_base_unit(Q(1, "millimeter * nanometer")) == test_units

        assert infer_base_unit("millimeter * nanometer", registry) == test_units

        assert (
            infer_base_unit(Q(1, "millimeter * nanometer").units, registry)
            == test_units
        )

        with pytest.raises(ValueError, match=r"No registry provided."):
            infer_base_unit("millimeter")

    def test_infer_base_unit_decimal(self):
        from pint.util import infer_base_unit

        ureg = UnitRegistry(non_int_type=Decimal)
        QD = ureg.Quantity

        ibu_d = infer_base_unit(QD(Decimal("1"), "millimeter * nanometer"))

        assert ibu_d == QD(Decimal("1"), "meter**2").units

        assert all(isinstance(v, Decimal) for v in ibu_d.values())

    def test_infer_base_unit_fraction(self):
        from pint.util import infer_base_unit

        ureg = UnitRegistry(non_int_type=Fraction)
        QD = ureg.Quantity

        ibu_d = infer_base_unit(QD(Fraction("1"), "millimeter * nanometer"))

        assert ibu_d == QD(Fraction("1"), "meter**2").units

        assert all(isinstance(v, Fraction) for v in ibu_d.values())

    def test_units_adding_to_zero(self):
        assert infer_base_unit(Q(1, "m * mm / m / um * s")) == Q(1, "s").units

    def test_to_compact(self):
        r = Q(1000000000, "m") * Q(1, "mm") / Q(1, "s") / Q(1, "ms")
        compact_r = r.to_compact()
        expected = Q(1000.0, "kilometer**2 / second**2")
        helpers.assert_quantity_almost_equal(compact_r, expected)

        r = (Q(1, "m") * Q(1, "mm") / Q(1, "m") / Q(2, "um") * Q(2, "s")).to_compact()
        helpers.assert_quantity_almost_equal(r, Q(1000, "s"))

    def test_to_compact_decimal(self):
        ureg = UnitRegistry(non_int_type=Decimal)
        Q = ureg.Quantity
        r = (
            Q(Decimal("1000000000.0"), "m")
            * Q(Decimal("1"), "mm")
            / Q(Decimal("1"), "s")
            / Q(Decimal("1"), "ms")
        )
        compact_r = r.to_compact()
        expected = Q(Decimal("1000.0"), "kilometer**2 / second**2")
        assert compact_r == expected

        r = (
            Q(Decimal(1), "m") * Q(1, "mm") / Q(1, "m**2") / Q(2, "um") * Q(2, "s")
        ).to_compact()
        assert r == Q(1000, "s/m")

    def test_to_compact_fraction(self):
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
            Q(Fraction(1), "m") * Q(1, "mm") / Q(1, "m**2") / Q(2, "um") * Q(2, "s")
        ).to_compact()
        assert r == Q(1000, "s/m")

    def test_volts(self):
        from pint.util import infer_base_unit

        r = Q(1, "V") * Q(1, "mV") / Q(1, "kV")
        b = infer_base_unit(r)
        assert b == Q(1, "V").units
        helpers.assert_quantity_almost_equal(r, Q(1, "uV"))
