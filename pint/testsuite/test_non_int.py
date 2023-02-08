import copy
import math
import operator as op
import pickle
from decimal import Decimal
from fractions import Fraction

import pytest

from pint import DimensionalityError, OffsetUnitCalculusError, UnitRegistry
from pint.facets.plain.unit import UnitsContainer
from pint.testsuite import QuantityTestCase, helpers


# TODO: do not subclass from QuantityTestCase
class NonIntTypeTestCase(QuantityTestCase):
    def assert_quantity_almost_equal(
        self, first, second, rtol="1e-07", atol="0", msg=None
    ):

        if isinstance(first, self.Q_):
            assert isinstance(first.m, (self.kwargs["non_int_type"], int))
        else:
            assert isinstance(first, (self.kwargs["non_int_type"], int))

        if isinstance(second, self.Q_):
            assert isinstance(second.m, (self.kwargs["non_int_type"], int))
        else:
            assert isinstance(second, (self.kwargs["non_int_type"], int))
        helpers.assert_quantity_almost_equal(
            first,
            second,
            self.kwargs["non_int_type"](rtol),
            self.kwargs["non_int_type"](atol),
            msg,
        )

    def QP_(self, value, units):
        assert isinstance(value, str)
        return self.Q_(self.kwargs["non_int_type"](value), units)


class _TestBasic(NonIntTypeTestCase):
    def test_quantity_creation(self, caplog):

        value = self.kwargs["non_int_type"]("4.2")

        for args in (
            (value, "meter"),
            (value, UnitsContainer(meter=1)),
            (value, self.ureg.meter),
            ("4.2*meter",),
            ("4.2/meter**(-1)",),
            (self.Q_(value, "meter"),),
        ):
            x = self.Q_(*args)
            assert x.magnitude == value
            assert x.units == self.ureg.UnitsContainer(meter=1)

        x = self.Q_(value, UnitsContainer(length=1))
        y = self.Q_(x)
        assert x.magnitude == y.magnitude
        assert x.units == y.units
        assert x is not y

        x = self.Q_(value, None)
        assert x.magnitude == value
        assert x.units == UnitsContainer()

        caplog.clear()
        assert value * self.ureg.meter == self.Q_(
            value, self.kwargs["non_int_type"]("2") * self.ureg.meter
        )
        assert len(caplog.records) == 1
        assert (
            caplog.records[0].message
            == "Creating new PlainQuantity using a non unity PlainQuantity as units."
        )

    def test_nan_creation(self):
        if self.SUPPORTS_NAN:
            value = self.kwargs["non_int_type"]("nan")

            for args in (
                (value, "meter"),
                (value, UnitsContainer(meter=1)),
                (value, self.ureg.meter),
                ("NaN*meter",),
                ("nan/meter**(-1)",),
                (self.Q_(value, "meter"),),
            ):
                x = self.Q_(*args)
                assert math.isnan(x.magnitude)
                assert type(x.magnitude) == self.kwargs["non_int_type"]
                assert x.units == self.ureg.UnitsContainer(meter=1)

        else:
            with pytest.raises(ValueError):
                self.Q_("NaN meters")

    def test_quantity_comparison(self):
        x = self.QP_("4.2", "meter")
        y = self.QP_("4.2", "meter")
        z = self.QP_("5", "meter")
        j = self.QP_("5", "meter*meter")

        # identity for single object
        assert x == x
        assert not (x != x)

        # identity for multiple objects with same value
        assert x == y
        assert not (x != y)

        assert x <= y
        assert x >= y
        assert not (x < y)
        assert not (x > y)

        assert not (x == z)
        assert x != z
        assert x < z

        assert z != j

        assert z != j
        assert self.QP_("0", "meter") == self.QP_("0", "centimeter")
        assert self.QP_("0", "meter") != self.QP_("0", "second")

        assert self.QP_("10", "meter") < self.QP_("5", "kilometer")

    def test_quantity_comparison_convert(self):
        assert self.QP_("1000", "millimeter") == self.QP_("1", "meter")
        assert self.QP_("1000", "millimeter/min") == self.Q_(
            self.kwargs["non_int_type"]("1000") / self.kwargs["non_int_type"]("60"),
            "millimeter/s",
        )

    def test_quantity_hash(self):
        x = self.QP_("4.2", "meter")
        x2 = self.QP_("4200", "millimeter")
        y = self.QP_("2", "second")
        z = self.QP_("0.5", "hertz")
        assert hash(x) == hash(x2)

        # Dimensionless equality
        assert hash(y * z) == hash(1.0)

        # Dimensionless equality from a different unit registry
        ureg2 = UnitRegistry()
        y2 = ureg2.Quantity(self.kwargs["non_int_type"]("2"), "second")
        z2 = ureg2.Quantity(self.kwargs["non_int_type"]("0.5"), "hertz")
        assert hash(y * z) == hash(y2 * z2)

    def test_to_base_units(self):
        x = self.Q_("1*inch")
        self.assert_quantity_almost_equal(
            x.to_base_units(), self.QP_("0.0254", "meter")
        )
        x = self.Q_("1*inch*inch")
        self.assert_quantity_almost_equal(
            x.to_base_units(),
            self.Q_(
                self.kwargs["non_int_type"]("0.0254")
                ** self.kwargs["non_int_type"]("2.0"),
                "meter*meter",
            ),
        )
        x = self.Q_("1*inch/minute")
        self.assert_quantity_almost_equal(
            x.to_base_units(),
            self.Q_(
                self.kwargs["non_int_type"]("0.0254")
                / self.kwargs["non_int_type"]("60"),
                "meter/second",
            ),
        )

    def test_convert(self):
        self.assert_quantity_almost_equal(
            self.Q_("2 inch").to("meter"),
            self.Q_(
                self.kwargs["non_int_type"]("2")
                * self.kwargs["non_int_type"]("0.0254"),
                "meter",
            ),
        )
        self.assert_quantity_almost_equal(
            self.Q_("2 meter").to("inch"),
            self.Q_(
                self.kwargs["non_int_type"]("2")
                / self.kwargs["non_int_type"]("0.0254"),
                "inch",
            ),
        )
        self.assert_quantity_almost_equal(
            self.Q_("2 sidereal_year").to("second"), self.QP_("63116297.5325", "second")
        )
        self.assert_quantity_almost_equal(
            self.Q_("2.54 centimeter/second").to("inch/second"),
            self.Q_("1 inch/second"),
        )
        assert round(abs(self.Q_("2.54 centimeter").to("inch").magnitude - 1), 7) == 0
        assert (
            round(abs(self.Q_("2 second").to("millisecond").magnitude - 2000), 7) == 0
        )

    def test_convert_from(self):
        x = self.Q_("2*inch")
        meter = self.ureg.meter

        # from quantity
        self.assert_quantity_almost_equal(
            meter.from_(x),
            self.Q_(
                self.kwargs["non_int_type"]("2")
                * self.kwargs["non_int_type"]("0.0254"),
                "meter",
            ),
        )
        self.assert_quantity_almost_equal(
            meter.m_from(x),
            self.kwargs["non_int_type"]("2") * self.kwargs["non_int_type"]("0.0254"),
        )

        # from unit
        self.assert_quantity_almost_equal(
            meter.from_(self.ureg.inch), self.QP_("0.0254", "meter")
        )
        self.assert_quantity_almost_equal(
            meter.m_from(self.ureg.inch), self.kwargs["non_int_type"]("0.0254")
        )

        # from number
        self.assert_quantity_almost_equal(
            meter.from_(2, strict=False), self.QP_("2", "meter")
        )
        self.assert_quantity_almost_equal(
            meter.m_from(self.kwargs["non_int_type"]("2"), strict=False),
            self.kwargs["non_int_type"]("2"),
        )

        # from number (strict mode)
        with pytest.raises(ValueError):
            meter.from_(self.kwargs["non_int_type"]("2"))
        with pytest.raises(ValueError):
            meter.m_from(self.kwargs["non_int_type"]("2"))

    def test_context_attr(self):
        assert self.ureg.meter == self.QP_("1", "meter")

    def test_both_symbol(self):
        assert self.QP_("2", "ms") == self.QP_("2", "millisecond")
        assert self.QP_("2", "cm") == self.QP_("2", "centimeter")

    def test_dimensionless_units(self):
        twopi = self.kwargs["non_int_type"]("2") * self.ureg.pi
        assert (
            round(abs(self.QP_("360", "degree").to("radian").magnitude - twopi), 7) == 0
        )
        assert round(abs(self.Q_(twopi, "radian") - self.QP_("360", "degree")), 7) == 0
        assert self.QP_("1", "radian").dimensionality == UnitsContainer()
        assert self.QP_("1", "radian").dimensionless
        assert not self.QP_("1", "radian").unitless

        assert self.QP_("1", "meter") / self.QP_("1", "meter") == 1
        assert (self.QP_("1", "meter") / self.QP_("1", "mm")).to("") == 1000

        assert self.Q_(10) // self.QP_("360", "degree") == 1
        assert self.QP_("400", "degree") // self.Q_(twopi) == 1
        assert self.QP_("400", "degree") // twopi == 1
        assert 7 // self.QP_("360", "degree") == 1

    def test_offset(self):
        self.assert_quantity_almost_equal(
            self.QP_("0", "kelvin").to("kelvin"), self.QP_("0", "kelvin")
        )
        self.assert_quantity_almost_equal(
            self.QP_("0", "degC").to("kelvin"), self.QP_("273.15", "kelvin")
        )
        self.assert_quantity_almost_equal(
            self.QP_("0", "degF").to("kelvin"),
            self.QP_("255.372222", "kelvin"),
            rtol=0.01,
        )

        self.assert_quantity_almost_equal(
            self.QP_("100", "kelvin").to("kelvin"), self.QP_("100", "kelvin")
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "degC").to("kelvin"), self.QP_("373.15", "kelvin")
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "degF").to("kelvin"),
            self.QP_("310.92777777", "kelvin"),
            rtol=0.01,
        )

        self.assert_quantity_almost_equal(
            self.QP_("0", "kelvin").to("degC"), self.QP_("-273.15", "degC")
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "kelvin").to("degC"), self.QP_("-173.15", "degC")
        )
        self.assert_quantity_almost_equal(
            self.QP_("0", "kelvin").to("degF"), self.QP_("-459.67", "degF"), rtol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "kelvin").to("degF"), self.QP_("-279.67", "degF"), rtol=0.01
        )

        self.assert_quantity_almost_equal(
            self.QP_("32", "degF").to("degC"), self.QP_("0", "degC"), atol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "degC").to("degF"), self.QP_("212", "degF"), atol=0.01
        )

        self.assert_quantity_almost_equal(
            self.QP_("54", "degF").to("degC"), self.QP_("12.2222", "degC"), atol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("12", "degC").to("degF"), self.QP_("53.6", "degF"), atol=0.01
        )

        self.assert_quantity_almost_equal(
            self.QP_("12", "kelvin").to("degC"), self.QP_("-261.15", "degC"), atol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("12", "degC").to("kelvin"), self.QP_("285.15", "kelvin"), atol=0.01
        )

        self.assert_quantity_almost_equal(
            self.QP_("12", "kelvin").to("degR"), self.QP_("21.6", "degR"), atol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("12", "degR").to("kelvin"),
            self.QP_("6.66666667", "kelvin"),
            atol=0.01,
        )

        self.assert_quantity_almost_equal(
            self.QP_("12", "degC").to("degR"), self.QP_("513.27", "degR"), atol=0.01
        )
        self.assert_quantity_almost_equal(
            self.QP_("12", "degR").to("degC"),
            self.QP_("-266.483333", "degC"),
            atol=0.01,
        )

    def test_offset_delta(self):
        self.assert_quantity_almost_equal(
            self.QP_("0", "delta_degC").to("kelvin"), self.QP_("0", "kelvin")
        )
        self.assert_quantity_almost_equal(
            self.QP_("0", "delta_degF").to("kelvin"), self.QP_("0", "kelvin"), rtol=0.01
        )

        self.assert_quantity_almost_equal(
            self.QP_("100", "kelvin").to("delta_degC"), self.QP_("100", "delta_degC")
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "kelvin").to("delta_degF"),
            self.QP_("180", "delta_degF"),
            rtol=0.01,
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "delta_degF").to("kelvin"),
            self.QP_("55.55555556", "kelvin"),
            rtol=0.01,
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "delta_degC").to("delta_degF"),
            self.QP_("180", "delta_degF"),
            rtol=0.01,
        )
        self.assert_quantity_almost_equal(
            self.QP_("100", "delta_degF").to("delta_degC"),
            self.QP_("55.55555556", "delta_degC"),
            rtol=0.01,
        )

        self.assert_quantity_almost_equal(
            self.QP_("12.3", "delta_degC").to("delta_degF"),
            self.QP_("22.14", "delta_degF"),
            rtol=0.01,
        )

    def test_pickle(self, subtests):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for magnitude, unit in (
                ("32", ""),
                ("2.4", ""),
                ("32", "m/s"),
                ("2.4", "m/s"),
            ):
                with subtests.test(protocol=protocol, magnitude=magnitude, unit=unit):
                    q1 = self.QP_(magnitude, unit)
                    q2 = pickle.loads(pickle.dumps(q1, protocol))
                    assert q1 == q2

    def test_notiter(self):
        # Verify that iter() crashes immediately, without needing to draw any
        # element from it, if the magnitude isn't iterable
        x = self.QP_("1", "m")
        with pytest.raises(TypeError):
            iter(x)


class _TestQuantityBasicMath(NonIntTypeTestCase):
    def _test_inplace(self, operator, value1, value2, expected_result, unit=None):
        if isinstance(value1, str):
            value1 = self.Q_(value1)
        if isinstance(value2, str):
            value2 = self.Q_(value2)
        if isinstance(expected_result, str):
            expected_result = self.Q_(expected_result)

        if unit is not None:
            value1 = value1 * unit
            value2 = value2 * unit
            expected_result = expected_result * unit

        value1 = copy.copy(value1)
        value2 = copy.copy(value2)
        id1 = id(value1)
        id2 = id(value2)
        value1 = operator(value1, value2)
        value2_cpy = copy.copy(value2)
        self.assert_quantity_almost_equal(value1, expected_result)
        assert id1 == id(value1)
        self.assert_quantity_almost_equal(value2, value2_cpy)
        assert id2 == id(value2)

    def _test_not_inplace(self, operator, value1, value2, expected_result, unit=None):
        if isinstance(value1, str):
            value1 = self.Q_(value1)
        if isinstance(value2, str):
            value2 = self.Q_(value2)
        if isinstance(expected_result, str):
            expected_result = self.Q_(expected_result)

        if unit is not None:
            value1 = value1 * unit
            value2 = value2 * unit
            expected_result = expected_result * unit

        id1 = id(value1)
        id2 = id(value2)

        value1_cpy = copy.copy(value1)
        value2_cpy = copy.copy(value2)

        result = operator(value1, value2)

        self.assert_quantity_almost_equal(expected_result, result)
        self.assert_quantity_almost_equal(value1, value1_cpy)
        self.assert_quantity_almost_equal(value2, value2_cpy)
        assert id(result) != id1
        assert id(result) != id2

    def _test_quantity_add_sub(self, unit, func):
        x = self.Q_(unit, "centimeter")
        y = self.Q_(unit, "inch")
        z = self.Q_(unit, "second")
        a = self.Q_(unit, None)

        func(op.add, x, x, self.Q_(unit + unit, "centimeter"))
        func(
            op.add,
            x,
            y,
            self.Q_(unit + self.kwargs["non_int_type"]("2.54") * unit, "centimeter"),
        )
        func(
            op.add,
            y,
            x,
            self.Q_(unit + unit / (self.kwargs["non_int_type"]("2.54") * unit), "inch"),
        )
        func(op.add, a, unit, self.Q_(unit + unit, None))
        with pytest.raises(DimensionalityError):
            op.add(self.kwargs["non_int_type"]("10"), x)
        with pytest.raises(DimensionalityError):
            op.add(x, self.kwargs["non_int_type"]("10"))
        with pytest.raises(DimensionalityError):
            op.add(x, z)

        func(op.sub, x, x, self.Q_(unit - unit, "centimeter"))
        func(
            op.sub,
            x,
            y,
            self.Q_(unit - self.kwargs["non_int_type"]("2.54") * unit, "centimeter"),
        )
        func(
            op.sub,
            y,
            x,
            self.Q_(unit - unit / (self.kwargs["non_int_type"]("2.54") * unit), "inch"),
        )
        func(op.sub, a, unit, self.Q_(unit - unit, None))
        with pytest.raises(DimensionalityError):
            op.sub(self.kwargs["non_int_type"]("10"), x)
        with pytest.raises(DimensionalityError):
            op.sub(x, self.kwargs["non_int_type"]("10"))
        with pytest.raises(DimensionalityError):
            op.sub(x, z)

    def _test_quantity_iadd_isub(self, unit, func):
        x = self.Q_(unit, "centimeter")
        y = self.Q_(unit, "inch")
        z = self.Q_(unit, "second")
        a = self.Q_(unit, None)

        func(op.iadd, x, x, self.Q_(unit + unit, "centimeter"))
        func(
            op.iadd,
            x,
            y,
            self.Q_(unit + self.kwargs["non_int_type"]("2.54") * unit, "centimeter"),
        )
        func(
            op.iadd,
            y,
            x,
            self.Q_(unit + unit / self.kwargs["non_int_type"]("2.54"), "inch"),
        )
        func(op.iadd, a, unit, self.Q_(unit + unit, None))
        with pytest.raises(DimensionalityError):
            op.iadd(self.kwargs["non_int_type"]("10"), x)
        with pytest.raises(DimensionalityError):
            op.iadd(x, self.kwargs["non_int_type"]("10"))
        with pytest.raises(DimensionalityError):
            op.iadd(x, z)

        func(op.isub, x, x, self.Q_(unit - unit, "centimeter"))
        func(
            op.isub,
            x,
            y,
            self.Q_(unit - self.kwargs["non_int_type"]("2.54"), "centimeter"),
        )
        func(
            op.isub,
            y,
            x,
            self.Q_(unit - unit / self.kwargs["non_int_type"]("2.54"), "inch"),
        )
        func(op.isub, a, unit, self.Q_(unit - unit, None))
        with pytest.raises(DimensionalityError):
            op.sub(self.kwargs["non_int_type"]("10"), x)
        with pytest.raises(DimensionalityError):
            op.sub(x, self.kwargs["non_int_type"]("10"))
        with pytest.raises(DimensionalityError):
            op.sub(x, z)

    def _test_quantity_mul_div(self, unit, func):
        func(
            op.mul,
            unit * self.kwargs["non_int_type"]("10"),
            "4.2*meter",
            "42*meter",
            unit,
        )
        func(
            op.mul,
            "4.2*meter",
            unit * self.kwargs["non_int_type"]("10"),
            "42*meter",
            unit,
        )
        func(op.mul, "4.2*meter", "10*inch", "42*meter*inch", unit)
        func(
            op.truediv,
            unit * self.kwargs["non_int_type"]("42"),
            "4.2*meter",
            "10/meter",
            unit,
        )
        func(
            op.truediv,
            "4.2*meter",
            unit * self.kwargs["non_int_type"]("10"),
            "0.42*meter",
            unit,
        )
        func(op.truediv, "4.2*meter", "10*inch", "0.42*meter/inch", unit)

    def _test_quantity_imul_idiv(self, unit, func):
        # func(op.imul, 10.0, '4.2*meter', '42*meter')
        func(op.imul, "4.2*meter", self.kwargs["non_int_type"]("10"), "42*meter", unit)
        func(op.imul, "4.2*meter", "10*inch", "42*meter*inch", unit)
        # func(op.truediv, 42, '4.2*meter', '10/meter')
        func(
            op.itruediv,
            "4.2*meter",
            unit * self.kwargs["non_int_type"]("10"),
            "0.42*meter",
            unit,
        )
        func(op.itruediv, "4.2*meter", "10*inch", "0.42*meter/inch", unit)

    def _test_quantity_floordiv(self, unit, func):
        a = self.Q_("10*meter")
        b = self.Q_("3*second")
        with pytest.raises(DimensionalityError):
            op.floordiv(a, b)
        with pytest.raises(DimensionalityError):
            op.floordiv(self.kwargs["non_int_type"]("3"), b)
        with pytest.raises(DimensionalityError):
            op.floordiv(a, self.kwargs["non_int_type"]("3"))
        with pytest.raises(DimensionalityError):
            op.ifloordiv(a, b)
        with pytest.raises(DimensionalityError):
            op.ifloordiv(self.kwargs["non_int_type"]("3"), b)
        with pytest.raises(DimensionalityError):
            op.ifloordiv(a, self.kwargs["non_int_type"]("3"))
        func(
            op.floordiv,
            unit * self.kwargs["non_int_type"]("10"),
            "4.2*meter/meter",
            self.kwargs["non_int_type"]("2"),
            unit,
        )
        func(
            op.floordiv, "10*meter", "4.2*inch", self.kwargs["non_int_type"]("93"), unit
        )

    def _test_quantity_mod(self, unit, func):
        a = self.Q_("10*meter")
        b = self.Q_("3*second")
        with pytest.raises(DimensionalityError):
            op.mod(a, b)
        with pytest.raises(DimensionalityError):
            op.mod(3, b)
        with pytest.raises(DimensionalityError):
            op.mod(a, 3)
        with pytest.raises(DimensionalityError):
            op.imod(a, b)
        with pytest.raises(DimensionalityError):
            op.imod(3, b)
        with pytest.raises(DimensionalityError):
            op.imod(a, 3)
        func(
            op.mod,
            unit * self.kwargs["non_int_type"]("10"),
            "4.2*meter/meter",
            self.kwargs["non_int_type"]("1.6"),
            unit,
        )

    def _test_quantity_ifloordiv(self, unit, func):
        func(
            op.ifloordiv,
            self.kwargs["non_int_type"]("10"),
            "4.2*meter/meter",
            self.kwargs["non_int_type"]("2"),
            unit,
        )
        func(
            op.ifloordiv,
            "10*meter",
            "4.2*inch",
            self.kwargs["non_int_type"]("93"),
            unit,
        )

    def _test_quantity_divmod_one(self, a, b):
        if isinstance(a, str):
            a = self.Q_(a)
        if isinstance(b, str):
            b = self.Q_(b)

        q, r = divmod(a, b)
        assert q == a // b
        assert r == a % b
        helpers.assert_quantity_equal(a, (q * b) + r)
        assert q == math.floor(q)
        if b > (0 * b):
            assert (0 * b) <= r < b
        else:
            assert (0 * b) >= r > b
        if isinstance(a, self.Q_):
            assert r.units == a.units
        else:
            assert r.unitless
        assert q.unitless

        copy_a = copy.copy(a)
        a %= b
        assert a == r
        copy_a //= b
        assert copy_a == q

    def _test_quantity_divmod(self):
        self._test_quantity_divmod_one("10*meter", "4.2*inch")

        # Disabling these tests as it yields different results without Quantities
        # >>> from decimal import Decimal as D
        # >>> divmod(-D('100'), D('3'))
        # (Decimal('-33'), Decimal('-1'))
        # >>> divmod(-100, 3)
        # (-34, 2)

        # self._test_quantity_divmod_one("-10*meter", "4.2*inch")
        # self._test_quantity_divmod_one("-10*meter", "-4.2*inch")
        # self._test_quantity_divmod_one("10*meter", "-4.2*inch")

        self._test_quantity_divmod_one("400*degree", "3")
        self._test_quantity_divmod_one("4", "180 degree")
        self._test_quantity_divmod_one(4, "180 degree")
        self._test_quantity_divmod_one("20", 4)
        self._test_quantity_divmod_one("300*degree", "100 degree")

        a = self.Q_("10*meter")
        b = self.Q_("3*second")
        with pytest.raises(DimensionalityError):
            divmod(a, b)
        with pytest.raises(DimensionalityError):
            divmod(3, b)
        with pytest.raises(DimensionalityError):
            divmod(a, 3)

    def _test_numeric(self, unit, ifunc):
        self._test_quantity_add_sub(unit, self._test_not_inplace)
        self._test_quantity_iadd_isub(unit, ifunc)
        self._test_quantity_mul_div(unit, self._test_not_inplace)
        self._test_quantity_imul_idiv(unit, ifunc)
        self._test_quantity_floordiv(unit, self._test_not_inplace)
        self._test_quantity_mod(unit, self._test_not_inplace)
        self._test_quantity_divmod()
        # self._test_quantity_ifloordiv(unit, ifunc)

    def test_quantity_abs_round(self):

        value = self.kwargs["non_int_type"]("4.2")
        x = self.Q_(-value, "meter")
        y = self.Q_(value, "meter")

        for fun in (abs, round, op.pos, op.neg):
            zx = self.Q_(fun(x.magnitude), "meter")
            zy = self.Q_(fun(y.magnitude), "meter")
            rx = fun(x)
            ry = fun(y)
            assert rx == zx, "while testing {0}".format(fun)
            assert ry == zy, "while testing {0}".format(fun)
            assert rx is not zx, "while testing {0}".format(fun)
            assert ry is not zy, "while testing {0}".format(fun)

    def test_quantity_float_complex(self):
        x = self.QP_("-4.2", None)
        y = self.QP_("4.2", None)
        z = self.QP_("1", "meter")
        for fun in (float, complex):
            assert fun(x) == fun(x.magnitude)
            assert fun(y) == fun(y.magnitude)
            with pytest.raises(DimensionalityError):
                fun(z)

    def test_not_inplace(self):
        self._test_numeric(self.kwargs["non_int_type"]("1.0"), self._test_not_inplace)


class _TestOffsetUnitMath(NonIntTypeTestCase):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.ureg.autoconvert_offset_to_baseunit = False
        cls.ureg.default_as_delta = True

    additions = [
        # --- input tuple -------------------- | -- expected result --
        ((("100", "kelvin"), ("10", "kelvin")), ("110", "kelvin")),
        ((("100", "kelvin"), ("10", "degC")), "error"),
        ((("100", "kelvin"), ("10", "degF")), "error"),
        ((("100", "kelvin"), ("10", "degR")), ("105.56", "kelvin")),
        ((("100", "kelvin"), ("10", "delta_degC")), ("110", "kelvin")),
        ((("100", "kelvin"), ("10", "delta_degF")), ("105.56", "kelvin")),
        ((("100", "degC"), ("10", "kelvin")), "error"),
        ((("100", "degC"), ("10", "degC")), "error"),
        ((("100", "degC"), ("10", "degF")), "error"),
        ((("100", "degC"), ("10", "degR")), "error"),
        ((("100", "degC"), ("10", "delta_degC")), ("110", "degC")),
        ((("100", "degC"), ("10", "delta_degF")), ("105.56", "degC")),
        ((("100", "degF"), ("10", "kelvin")), "error"),
        ((("100", "degF"), ("10", "degC")), "error"),
        ((("100", "degF"), ("10", "degF")), "error"),
        ((("100", "degF"), ("10", "degR")), "error"),
        ((("100", "degF"), ("10", "delta_degC")), ("118", "degF")),
        ((("100", "degF"), ("10", "delta_degF")), ("110", "degF")),
        ((("100", "degR"), ("10", "kelvin")), ("118", "degR")),
        ((("100", "degR"), ("10", "degC")), "error"),
        ((("100", "degR"), ("10", "degF")), "error"),
        ((("100", "degR"), ("10", "degR")), ("110", "degR")),
        ((("100", "degR"), ("10", "delta_degC")), ("118", "degR")),
        ((("100", "degR"), ("10", "delta_degF")), ("110", "degR")),
        ((("100", "delta_degC"), ("10", "kelvin")), ("110", "kelvin")),
        ((("100", "delta_degC"), ("10", "degC")), ("110", "degC")),
        ((("100", "delta_degC"), ("10", "degF")), ("190", "degF")),
        ((("100", "delta_degC"), ("10", "degR")), ("190", "degR")),
        ((("100", "delta_degC"), ("10", "delta_degC")), ("110", "delta_degC")),
        ((("100", "delta_degC"), ("10", "delta_degF")), ("105.56", "delta_degC")),
        ((("100", "delta_degF"), ("10", "kelvin")), ("65.56", "kelvin")),
        ((("100", "delta_degF"), ("10", "degC")), ("65.56", "degC")),
        ((("100", "delta_degF"), ("10", "degF")), ("110", "degF")),
        ((("100", "delta_degF"), ("10", "degR")), ("110", "degR")),
        ((("100", "delta_degF"), ("10", "delta_degC")), ("118", "delta_degF")),
        ((("100", "delta_degF"), ("10", "delta_degF")), ("110", "delta_degF")),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), additions)
    def test_addition(self, input_tuple, expected_output):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.QP_(*qin1), self.QP_(*qin2)
        # update input tuple with new values to have correct values on failure
        input_tuple = q1, q2
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.add(q1, q2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.add(q1, q2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.add(q1, q2), expected_output, atol="0.01"
            )

    subtractions = [
        ((("100", "kelvin"), ("10", "kelvin")), ("90", "kelvin")),
        ((("100", "kelvin"), ("10", "degC")), ("-183.15", "kelvin")),
        ((("100", "kelvin"), ("10", "degF")), ("-160.93", "kelvin")),
        ((("100", "kelvin"), ("10", "degR")), ("94.44", "kelvin")),
        ((("100", "kelvin"), ("10", "delta_degC")), ("90", "kelvin")),
        ((("100", "kelvin"), ("10", "delta_degF")), ("94.44", "kelvin")),
        ((("100", "degC"), ("10", "kelvin")), ("363.15", "delta_degC")),
        ((("100", "degC"), ("10", "degC")), ("90", "delta_degC")),
        ((("100", "degC"), ("10", "degF")), ("112.22", "delta_degC")),
        ((("100", "degC"), ("10", "degR")), ("367.59", "delta_degC")),
        ((("100", "degC"), ("10", "delta_degC")), ("90", "degC")),
        ((("100", "degC"), ("10", "delta_degF")), ("94.44", "degC")),
        ((("100", "degF"), ("10", "kelvin")), ("541.67", "delta_degF")),
        ((("100", "degF"), ("10", "degC")), ("50", "delta_degF")),
        ((("100", "degF"), ("10", "degF")), ("90", "delta_degF")),
        ((("100", "degF"), ("10", "degR")), ("549.67", "delta_degF")),
        ((("100", "degF"), ("10", "delta_degC")), ("82", "degF")),
        ((("100", "degF"), ("10", "delta_degF")), ("90", "degF")),
        ((("100", "degR"), ("10", "kelvin")), ("82", "degR")),
        ((("100", "degR"), ("10", "degC")), ("-409.67", "degR")),
        ((("100", "degR"), ("10", "degF")), ("-369.67", "degR")),
        ((("100", "degR"), ("10", "degR")), ("90", "degR")),
        ((("100", "degR"), ("10", "delta_degC")), ("82", "degR")),
        ((("100", "degR"), ("10", "delta_degF")), ("90", "degR")),
        ((("100", "delta_degC"), ("10", "kelvin")), ("90", "kelvin")),
        ((("100", "delta_degC"), ("10", "degC")), ("90", "degC")),
        ((("100", "delta_degC"), ("10", "degF")), ("170", "degF")),
        ((("100", "delta_degC"), ("10", "degR")), ("170", "degR")),
        ((("100", "delta_degC"), ("10", "delta_degC")), ("90", "delta_degC")),
        ((("100", "delta_degC"), ("10", "delta_degF")), ("94.44", "delta_degC")),
        ((("100", "delta_degF"), ("10", "kelvin")), ("45.56", "kelvin")),
        ((("100", "delta_degF"), ("10", "degC")), ("45.56", "degC")),
        ((("100", "delta_degF"), ("10", "degF")), ("90", "degF")),
        ((("100", "delta_degF"), ("10", "degR")), ("90", "degR")),
        ((("100", "delta_degF"), ("10", "delta_degC")), ("82", "delta_degF")),
        ((("100", "delta_degF"), ("10", "delta_degF")), ("90", "delta_degF")),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), subtractions)
    def test_subtraction(self, input_tuple, expected_output):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.QP_(*qin1), self.QP_(*qin2)
        input_tuple = q1, q2
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.sub(q1, q2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.sub(q1, q2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.sub(q1, q2), expected_output, atol=0.01
            )

    multiplications = [
        ((("100", "kelvin"), ("10", "kelvin")), ("1000", "kelvin**2")),
        ((("100", "kelvin"), ("10", "degC")), "error"),
        ((("100", "kelvin"), ("10", "degF")), "error"),
        ((("100", "kelvin"), ("10", "degR")), ("1000", "kelvin*degR")),
        ((("100", "kelvin"), ("10", "delta_degC")), ("1000", "kelvin*delta_degC")),
        ((("100", "kelvin"), ("10", "delta_degF")), ("1000", "kelvin*delta_degF")),
        ((("100", "degC"), ("10", "kelvin")), "error"),
        ((("100", "degC"), ("10", "degC")), "error"),
        ((("100", "degC"), ("10", "degF")), "error"),
        ((("100", "degC"), ("10", "degR")), "error"),
        ((("100", "degC"), ("10", "delta_degC")), "error"),
        ((("100", "degC"), ("10", "delta_degF")), "error"),
        ((("100", "degF"), ("10", "kelvin")), "error"),
        ((("100", "degF"), ("10", "degC")), "error"),
        ((("100", "degF"), ("10", "degF")), "error"),
        ((("100", "degF"), ("10", "degR")), "error"),
        ((("100", "degF"), ("10", "delta_degC")), "error"),
        ((("100", "degF"), ("10", "delta_degF")), "error"),
        ((("100", "degR"), ("10", "kelvin")), ("1000", "degR*kelvin")),
        ((("100", "degR"), ("10", "degC")), "error"),
        ((("100", "degR"), ("10", "degF")), "error"),
        ((("100", "degR"), ("10", "degR")), ("1000", "degR**2")),
        ((("100", "degR"), ("10", "delta_degC")), ("1000", "degR*delta_degC")),
        ((("100", "degR"), ("10", "delta_degF")), ("1000", "degR*delta_degF")),
        ((("100", "delta_degC"), ("10", "kelvin")), ("1000", "delta_degC*kelvin")),
        ((("100", "delta_degC"), ("10", "degC")), "error"),
        ((("100", "delta_degC"), ("10", "degF")), "error"),
        ((("100", "delta_degC"), ("10", "degR")), ("1000", "delta_degC*degR")),
        ((("100", "delta_degC"), ("10", "delta_degC")), ("1000", "delta_degC**2")),
        (
            (("100", "delta_degC"), ("10", "delta_degF")),
            ("1000", "delta_degC*delta_degF"),
        ),
        ((("100", "delta_degF"), ("10", "kelvin")), ("1000", "delta_degF*kelvin")),
        ((("100", "delta_degF"), ("10", "degC")), "error"),
        ((("100", "delta_degF"), ("10", "degF")), "error"),
        ((("100", "delta_degF"), ("10", "degR")), ("1000", "delta_degF*degR")),
        (
            (("100", "delta_degF"), ("10", "delta_degC")),
            ("1000", "delta_degF*delta_degC"),
        ),
        ((("100", "delta_degF"), ("10", "delta_degF")), ("1000", "delta_degF**2")),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), multiplications)
    def test_multiplication(self, input_tuple, expected_output):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.QP_(*qin1), self.QP_(*qin2)
        input_tuple = q1, q2
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.mul(q1, q2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.mul(q1, q2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.mul(q1, q2), expected_output, atol=0.01
            )

    divisions = [
        ((("100", "kelvin"), ("10", "kelvin")), ("10", "")),
        ((("100", "kelvin"), ("10", "degC")), "error"),
        ((("100", "kelvin"), ("10", "degF")), "error"),
        ((("100", "kelvin"), ("10", "degR")), ("10", "kelvin/degR")),
        ((("100", "kelvin"), ("10", "delta_degC")), ("10", "kelvin/delta_degC")),
        ((("100", "kelvin"), ("10", "delta_degF")), ("10", "kelvin/delta_degF")),
        ((("100", "degC"), ("10", "kelvin")), "error"),
        ((("100", "degC"), ("10", "degC")), "error"),
        ((("100", "degC"), ("10", "degF")), "error"),
        ((("100", "degC"), ("10", "degR")), "error"),
        ((("100", "degC"), ("10", "delta_degC")), "error"),
        ((("100", "degC"), ("10", "delta_degF")), "error"),
        ((("100", "degF"), ("10", "kelvin")), "error"),
        ((("100", "degF"), ("10", "degC")), "error"),
        ((("100", "degF"), ("10", "degF")), "error"),
        ((("100", "degF"), ("10", "degR")), "error"),
        ((("100", "degF"), ("10", "delta_degC")), "error"),
        ((("100", "degF"), ("10", "delta_degF")), "error"),
        ((("100", "degR"), ("10", "kelvin")), ("10", "degR/kelvin")),
        ((("100", "degR"), ("10", "degC")), "error"),
        ((("100", "degR"), ("10", "degF")), "error"),
        ((("100", "degR"), ("10", "degR")), ("10", "")),
        ((("100", "degR"), ("10", "delta_degC")), ("10", "degR/delta_degC")),
        ((("100", "degR"), ("10", "delta_degF")), ("10", "degR/delta_degF")),
        ((("100", "delta_degC"), ("10", "kelvin")), ("10", "delta_degC/kelvin")),
        ((("100", "delta_degC"), ("10", "degC")), "error"),
        ((("100", "delta_degC"), ("10", "degF")), "error"),
        ((("100", "delta_degC"), ("10", "degR")), ("10", "delta_degC/degR")),
        ((("100", "delta_degC"), ("10", "delta_degC")), ("10", "")),
        (
            (("100", "delta_degC"), ("10", "delta_degF")),
            ("10", "delta_degC/delta_degF"),
        ),
        ((("100", "delta_degF"), ("10", "kelvin")), ("10", "delta_degF/kelvin")),
        ((("100", "delta_degF"), ("10", "degC")), "error"),
        ((("100", "delta_degF"), ("10", "degF")), "error"),
        ((("100", "delta_degF"), ("10", "degR")), ("10", "delta_degF/degR")),
        (
            (("100", "delta_degF"), ("10", "delta_degC")),
            ("10", "delta_degF/delta_degC"),
        ),
        ((("100", "delta_degF"), ("10", "delta_degF")), ("10", "")),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), divisions)
    def test_truedivision(self, input_tuple, expected_output):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.QP_(*qin1), self.QP_(*qin2)
        input_tuple = q1, q2
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.truediv(q1, q2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.truediv(q1, q2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.truediv(q1, q2), expected_output, atol=0.01
            )

    multiplications_with_autoconvert_to_baseunit = [
        ((("100", "kelvin"), ("10", "degC")), ("28315.0", "kelvin**2")),
        ((("100", "kelvin"), ("10", "degF")), ("26092.78", "kelvin**2")),
        ((("100", "degC"), ("10", "kelvin")), ("3731.5", "kelvin**2")),
        ((("100", "degC"), ("10", "degC")), ("105657.42", "kelvin**2")),
        ((("100", "degC"), ("10", "degF")), ("97365.20", "kelvin**2")),
        ((("100", "degC"), ("10", "degR")), ("3731.5", "kelvin*degR")),
        ((("100", "degC"), ("10", "delta_degC")), ("3731.5", "kelvin*delta_degC")),
        ((("100", "degC"), ("10", "delta_degF")), ("3731.5", "kelvin*delta_degF")),
        ((("100", "degF"), ("10", "kelvin")), ("3109.28", "kelvin**2")),
        ((("100", "degF"), ("10", "degC")), ("88039.20", "kelvin**2")),
        ((("100", "degF"), ("10", "degF")), ("81129.69", "kelvin**2")),
        ((("100", "degF"), ("10", "degR")), ("3109.28", "kelvin*degR")),
        ((("100", "degF"), ("10", "delta_degC")), ("3109.28", "kelvin*delta_degC")),
        ((("100", "degF"), ("10", "delta_degF")), ("3109.28", "kelvin*delta_degF")),
        ((("100", "degR"), ("10", "degC")), ("28315.0", "degR*kelvin")),
        ((("100", "degR"), ("10", "degF")), ("26092.78", "degR*kelvin")),
        ((("100", "delta_degC"), ("10", "degC")), ("28315.0", "delta_degC*kelvin")),
        ((("100", "delta_degC"), ("10", "degF")), ("26092.78", "delta_degC*kelvin")),
        ((("100", "delta_degF"), ("10", "degC")), ("28315.0", "delta_degF*kelvin")),
        ((("100", "delta_degF"), ("10", "degF")), ("26092.78", "delta_degF*kelvin")),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected_output"), multiplications_with_autoconvert_to_baseunit
    )
    def test_multiplication_with_autoconvert(self, input_tuple, expected_output):
        self.ureg.autoconvert_offset_to_baseunit = True
        qin1, qin2 = input_tuple
        q1, q2 = self.QP_(*qin1), self.QP_(*qin2)
        input_tuple = q1, q2
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.mul(q1, q2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.mul(q1, q2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.mul(q1, q2), expected_output, atol=0.01
            )

    multiplications_with_scalar = [
        ((("10", "kelvin"), "2"), ("20.0", "kelvin")),
        ((("10", "kelvin**2"), "2"), ("20.0", "kelvin**2")),
        ((("10", "degC"), "2"), ("20.0", "degC")),
        ((("10", "1/degC"), "2"), "error"),
        ((("10", "degC**0.5"), "2"), "error"),
        ((("10", "degC**2"), "2"), "error"),
        ((("10", "degC**-2"), "2"), "error"),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected_output"), multiplications_with_scalar
    )
    def test_multiplication_with_scalar(self, input_tuple, expected_output):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple:
            in1, in2 = self.QP_(*in1), self.kwargs["non_int_type"](in2)
        else:
            in1, in2 = in1, self.QP_(*in2)
        input_tuple = in1, in2  # update input_tuple for better tracebacks
        if expected_output == "error":
            with pytest.raises(OffsetUnitCalculusError):
                op.mul(in1, in2)
        else:
            expected_output = self.QP_(*expected_output)
            assert op.mul(in1, in2).units == expected_output.units
            self.assert_quantity_almost_equal(
                op.mul(in1, in2), expected_output, atol="0.01"
            )

    divisions_with_scalar = [  # without / with autoconvert to plain unit
        ((("10", "kelvin"), "2"), [("5.0", "kelvin"), ("5.0", "kelvin")]),
        ((("10", "kelvin**2"), "2"), [("5.0", "kelvin**2"), ("5.0", "kelvin**2")]),
        ((("10", "degC"), "2"), ["error", "error"]),
        ((("10", "degC**2"), "2"), ["error", "error"]),
        ((("10", "degC**-2"), "2"), ["error", "error"]),
        (("2", ("10", "kelvin")), [("0.2", "1/kelvin"), ("0.2", "1/kelvin")]),
        # (('2', ('10', "degC")), ["error", (2 / 283.15, "1/kelvin")]),
        (("2", ("10", "degC**2")), ["error", "error"]),
        (("2", ("10", "degC**-2")), ["error", "error"]),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), divisions_with_scalar)
    def test_division_with_scalar(self, input_tuple, expected_output):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple:
            in1, in2 = self.QP_(*in1), self.kwargs["non_int_type"](in2)
        else:
            in1, in2 = self.kwargs["non_int_type"](in1), self.QP_(*in2)
        input_tuple = in1, in2  # update input_tuple for better tracebacks
        expected_copy = expected_output[:]
        for i, mode in enumerate([False, True]):
            self.ureg.autoconvert_offset_to_baseunit = mode
            if expected_copy[i] == "error":
                with pytest.raises(OffsetUnitCalculusError):
                    op.truediv(in1, in2)
            else:
                expected_output = self.QP_(*expected_copy[i])
                assert op.truediv(in1, in2).units == expected_output.units
                self.assert_quantity_almost_equal(op.truediv(in1, in2), expected_output)

    exponentiation = [  # results without / with autoconvert
        ((("10", "degC"), "1"), [("10", "degC"), ("10", "degC")]),
        # ((('10', "degC"), 0.5), ["error", (283.15 ** '0.5', "kelvin**0.5")]),
        ((("10", "degC"), "0"), [("1.0", ""), ("1.0", "")]),
        # ((('10', "degC"), -1), ["error", (1 / (10 + 273.15), "kelvin**-1")]),
        # ((('10', "degC"), -2), ["error", (1 / (10 + 273.15) ** 2.0, "kelvin**-2")]),
        # ((('0', "degC"), -2), ["error", (1 / (273.15) ** 2, "kelvin**-2")]),
        # ((('10', "degC"), ('2', "")), ["error", ((283.15) ** 2, "kelvin**2")]),
        ((("10", "degC"), ("10", "degK")), ["error", "error"]),
        (
            (("10", "kelvin"), ("2", "")),
            [("100.0", "kelvin**2"), ("100.0", "kelvin**2")],
        ),
        (("2", ("2", "kelvin")), ["error", "error"]),
        # (('2', ('500.0', "millikelvin/kelvin")), [2 ** 0.5, 2 ** 0.5]),
        # (('2', ('0.5', "kelvin/kelvin")), [2 ** 0.5, 2 ** 0.5]),
        # (
        #     (('10', "degC"), ('500.0', "millikelvin/kelvin")),
        #     ["error", (283.15 ** '0.5', "kelvin**0.5")],
        # ),
    ]

    @pytest.mark.parametrize(("input_tuple", "expected_output"), exponentiation)
    def test_exponentiation(self, input_tuple, expected_output):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple and type(in2) is tuple:
            in1, in2 = self.QP_(*in1), self.QP_(*in2)
        elif not type(in1) is tuple and type(in2) is tuple:
            in1, in2 = self.kwargs["non_int_type"](in1), self.QP_(*in2)
        else:
            in1, in2 = self.QP_(*in1), self.kwargs["non_int_type"](in2)
        input_tuple = in1, in2
        expected_copy = expected_output[:]
        for i, mode in enumerate([False, True]):
            self.ureg.autoconvert_offset_to_baseunit = mode
            if expected_copy[i] == "error":
                with pytest.raises((OffsetUnitCalculusError, DimensionalityError)):
                    op.pow(in1, in2)
            else:
                if type(expected_copy[i]) is tuple:
                    expected_output = self.QP_(*expected_copy[i])
                    assert op.pow(in1, in2).units == expected_output.units
                else:
                    expected_output = expected_copy[i]
                self.assert_quantity_almost_equal(op.pow(in1, in2), expected_output)


class TestNonIntTypeQuantityFloat(_TestBasic):

    kwargs = dict(non_int_type=float)
    SUPPORTS_NAN = True


class TestNonIntTypeQuantityBasicMathFloat(_TestQuantityBasicMath):

    kwargs = dict(non_int_type=float)


class TestNonIntTypeOffsetUnitMathFloat(_TestOffsetUnitMath):

    kwargs = dict(non_int_type=float)


class TestNonIntTypeQuantityDecimal(_TestBasic):

    kwargs = dict(non_int_type=Decimal)
    SUPPORTS_NAN = True


class TestNonIntTypeQuantityBasicMathDecimal(_TestQuantityBasicMath):

    kwargs = dict(non_int_type=Decimal)


class TestNonIntTypeOffsetUnitMathDecimal(_TestOffsetUnitMath):

    kwargs = dict(non_int_type=Decimal)


class TestNonIntTypeQuantityFraction(_TestBasic):

    kwargs = dict(non_int_type=Fraction)
    SUPPORTS_NAN = False


class TestNonIntTypeQuantityBasicMathFraction(_TestQuantityBasicMath):

    kwargs = dict(non_int_type=Fraction)


class TestNonIntTypeOffsetUnitMathFraction(_TestOffsetUnitMath):

    kwargs = dict(non_int_type=Fraction)
