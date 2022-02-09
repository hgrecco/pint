from decimal import Decimal

from pint import Quantity as Q
from pint import UnitRegistry
from pint.testsuite import helpers
from pint.util import infer_base_unit


class TestInferBaseUnit:
    def test_infer_base_unit(self):
        from pint.util import infer_base_unit

        assert infer_base_unit(Q(1, "millimeter * nanometer")) == Q(1, "meter**2").units

    def test_infer_base_unit_decimal(self):
        from pint.util import infer_base_unit

        ureg = UnitRegistry(non_int_type=Decimal)
        Q = ureg.Quantity

        assert (
            infer_base_unit(Q(Decimal("1"), "millimeter * nanometer"))
            == Q(Decimal("1"), "meter**2").units
        )

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
            Q(Decimal(1), "m") * Q(1, "mm") / Q(1, "m") / Q(2, "um") * Q(2, "s")
        ).to_compact()
        assert r == Q(1000, "s")

    def test_volts(self):
        from pint.util import infer_base_unit

        r = Q(1, "V") * Q(1, "mV") / Q(1, "kV")
        b = infer_base_unit(r)
        assert b == Q(1, "V").units
        helpers.assert_quantity_almost_equal(r, Q(1, "uV"))
