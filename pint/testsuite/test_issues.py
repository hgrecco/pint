import copy
import math
import pprint

import pytest

from pint import Context, DimensionalityError, UnitRegistry, get_application_registry
from pint.compat import np
from pint.testsuite import QuantityTestCase, helpers
from pint.unit import UnitsContainer
from pint.util import ParserHelper


# TODO: do not subclass from QuantityTestCase
class TestIssues(QuantityTestCase):

    kwargs = dict(autoconvert_offset_to_baseunit=False)

    @pytest.mark.xfail
    def test_issue25(self, module_registry):
        x = ParserHelper.from_string("10 %")
        assert x == ParserHelper(10, {"%": 1})
        x = ParserHelper.from_string("10 ‰")
        assert x == ParserHelper(10, {"‰": 1})
        module_registry.define("percent = [fraction]; offset: 0 = %")
        module_registry.define("permille = percent / 10 = ‰")
        x = module_registry.parse_expression("10 %")
        assert x == module_registry.Quantity(10, {"%": 1})
        y = module_registry.parse_expression("10 ‰")
        assert y == module_registry.Quantity(10, {"‰": 1})
        assert x.to("‰") == module_registry.Quantity(1, {"‰": 1})

    def test_issue29(self, module_registry):
        t = 4 * module_registry("mW")
        assert t.magnitude == 4
        assert t._units == UnitsContainer(milliwatt=1)
        assert t.to("joule / second") == 4e-3 * module_registry("W")

    @pytest.mark.xfail
    @helpers.requires_numpy
    def test_issue37(self, module_registry):
        x = np.ma.masked_array([1, 2, 3], mask=[True, True, False])
        q = module_registry.meter * x
        assert isinstance(q, module_registry.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        assert q.units == module_registry.meter.units
        q = x * module_registry.meter
        assert isinstance(q, module_registry.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        assert q.units == module_registry.meter.units

        m = np.ma.masked_array(2 * np.ones(3, 3))
        qq = q * m
        assert isinstance(qq, module_registry.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        assert qq.units == module_registry.meter.units
        qq = m * q
        assert isinstance(qq, module_registry.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        assert qq.units == module_registry.meter.units

    @pytest.mark.xfail
    @helpers.requires_numpy
    def test_issue39(self, module_registry):
        x = np.matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        q = module_registry.meter * x
        assert isinstance(q, module_registry.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        assert q.units == module_registry.meter.units
        q = x * module_registry.meter
        assert isinstance(q, module_registry.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        assert q.units == module_registry.meter.units

        m = np.matrix(2 * np.ones(3, 3))
        qq = q * m
        assert isinstance(qq, module_registry.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        assert qq.units == module_registry.meter.units
        qq = m * q
        assert isinstance(qq, module_registry.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        assert qq.units == module_registry.meter.units

    @helpers.requires_numpy
    def test_issue44(self, module_registry):
        x = 4.0 * module_registry.dimensionless
        np.sqrt(x)
        helpers.assert_quantity_almost_equal(
            np.sqrt([4.0] * module_registry.dimensionless),
            [2.0] * module_registry.dimensionless,
        )
        helpers.assert_quantity_almost_equal(
            np.sqrt(4.0 * module_registry.dimensionless),
            2.0 * module_registry.dimensionless,
        )

    def test_issue45(self, module_registry):
        import math

        helpers.assert_quantity_almost_equal(
            math.sqrt(4 * module_registry.m / module_registry.cm), math.sqrt(4 * 100)
        )
        helpers.assert_quantity_almost_equal(
            float(module_registry.V / module_registry.mV), 1000.0
        )

    @helpers.requires_numpy
    def test_issue45b(self, module_registry):
        helpers.assert_quantity_almost_equal(
            np.sin([np.pi / 2] * module_registry.m / module_registry.m),
            np.sin([np.pi / 2] * module_registry.dimensionless),
        )
        helpers.assert_quantity_almost_equal(
            np.sin([np.pi / 2] * module_registry.cm / module_registry.m),
            np.sin([np.pi / 2] * module_registry.dimensionless * 0.01),
        )

    def test_issue50(self, module_registry):
        Q_ = module_registry.Quantity
        assert Q_(100) == 100 * module_registry.dimensionless
        assert Q_("100") == 100 * module_registry.dimensionless

    def test_issue52(self):
        u1 = UnitRegistry()
        u2 = UnitRegistry()
        q1 = 1 * u1.meter
        q2 = 1 * u2.meter
        import operator as op

        for fun in (
            op.add,
            op.iadd,
            op.sub,
            op.isub,
            op.mul,
            op.imul,
            op.floordiv,
            op.ifloordiv,
            op.truediv,
            op.itruediv,
        ):
            with pytest.raises(ValueError):
                fun(q1, q2)

    def test_issue54(self, module_registry):
        assert (1 * module_registry.km / module_registry.m + 1).magnitude == 1001

    def test_issue54_related(self, module_registry):
        assert module_registry.km / module_registry.m == 1000
        assert 1000 == module_registry.km / module_registry.m
        assert 900 < module_registry.km / module_registry.m
        assert 1100 > module_registry.km / module_registry.m

    def test_issue61(self, module_registry):
        Q_ = module_registry.Quantity
        for value in ({}, {"a": 3}, None):
            with pytest.raises(TypeError):
                Q_(value)
            with pytest.raises(TypeError):
                Q_(value, "meter")
        with pytest.raises(ValueError):
            Q_("", "meter")
        with pytest.raises(ValueError):
            Q_("")

    @helpers.requires_not_numpy()
    def test_issue61_notNP(self, module_registry):
        Q_ = module_registry.Quantity
        for value in ([1, 2, 3], (1, 2, 3)):
            with pytest.raises(TypeError):
                Q_(value)
            with pytest.raises(TypeError):
                Q_(value, "meter")

    def test_issue62(self, module_registry):
        m = module_registry("m**0.5")
        assert str(m.units) == "meter ** 0.5"

    def test_issue66(self, module_registry):
        assert module_registry.get_dimensionality(
            UnitsContainer({"[temperature]": 1})
        ) == UnitsContainer({"[temperature]": 1})
        assert module_registry.get_dimensionality(
            module_registry.kelvin
        ) == UnitsContainer({"[temperature]": 1})
        assert module_registry.get_dimensionality(
            module_registry.degC
        ) == UnitsContainer({"[temperature]": 1})

    def test_issue66b(self, module_registry):
        assert module_registry.get_base_units(module_registry.kelvin) == (
            1.0,
            module_registry.Unit(UnitsContainer({"kelvin": 1})),
        )
        assert module_registry.get_base_units(module_registry.degC) == (
            1.0,
            module_registry.Unit(UnitsContainer({"kelvin": 1})),
        )

    def test_issue69(self, module_registry):
        q = module_registry("m").to(module_registry("in"))
        assert q == module_registry("m").to("in")

    @helpers.requires_numpy
    def test_issue74(self, module_registry):
        v1 = np.asarray([1.0, 2.0, 3.0])
        v2 = np.asarray([3.0, 2.0, 1.0])
        q1 = v1 * module_registry.ms
        q2 = v2 * module_registry.ms

        np.testing.assert_array_equal(q1 < q2, v1 < v2)
        np.testing.assert_array_equal(q1 > q2, v1 > v2)

        np.testing.assert_array_equal(q1 <= q2, v1 <= v2)
        np.testing.assert_array_equal(q1 >= q2, v1 >= v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * module_registry.s
        v2s = q2s.to("ms").magnitude

        np.testing.assert_array_equal(q1 < q2s, v1 < v2s)
        np.testing.assert_array_equal(q1 > q2s, v1 > v2s)

        np.testing.assert_array_equal(q1 <= q2s, v1 <= v2s)
        np.testing.assert_array_equal(q1 >= q2s, v1 >= v2s)

    @helpers.requires_numpy
    def test_issue75(self, module_registry):
        v1 = np.asarray([1.0, 2.0, 3.0])
        v2 = np.asarray([3.0, 2.0, 1.0])
        q1 = v1 * module_registry.ms
        q2 = v2 * module_registry.ms

        np.testing.assert_array_equal(q1 == q2, v1 == v2)
        np.testing.assert_array_equal(q1 != q2, v1 != v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * module_registry.s
        v2s = q2s.to("ms").magnitude

        np.testing.assert_array_equal(q1 == q2s, v1 == v2s)
        np.testing.assert_array_equal(q1 != q2s, v1 != v2s)

    @helpers.requires_uncertainties()
    def test_issue77(self, module_registry):
        acc = (5.0 * module_registry("m/s/s")).plus_minus(0.25)
        tim = (37.0 * module_registry("s")).plus_minus(0.16)
        dis = acc * tim**2 / 2
        assert dis.value == acc.value * tim.value**2 / 2

    def test_issue85(self, module_registry):

        T = 4.0 * module_registry.kelvin
        m = 1.0 * module_registry.amu
        va = 2.0 * module_registry.k * T / m

        va.to_base_units()

        boltmk = 1.380649e-23 * module_registry.J / module_registry.K
        vb = 2.0 * boltmk * T / m

        helpers.assert_quantity_almost_equal(va.to_base_units(), vb.to_base_units())

    def test_issue86(self, module_registry):

        module_registry.autoconvert_offset_to_baseunit = True

        def parts(q):
            return q.magnitude, q.units

        q1 = 10.0 * module_registry.degC
        q2 = 10.0 * module_registry.kelvin

        k1 = q1.to_base_units()

        q3 = 3.0 * module_registry.meter

        q1m, q1u = parts(q1)
        q2m, q2u = parts(q2)
        q3m, q3u = parts(q3)

        k1m, k1u = parts(k1)

        assert parts(q2 * q3) == (q2m * q3m, q2u * q3u)
        assert parts(q2 / q3) == (q2m / q3m, q2u / q3u)
        assert parts(q3 * q2) == (q3m * q2m, q3u * q2u)
        assert parts(q3 / q2) == (q3m / q2m, q3u / q2u)
        assert parts(q2**1) == (q2m**1, q2u**1)
        assert parts(q2**-1) == (q2m**-1, q2u**-1)
        assert parts(q2**2) == (q2m**2, q2u**2)
        assert parts(q2**-2) == (q2m**-2, q2u**-2)

        assert parts(q1 * q3) == (k1m * q3m, k1u * q3u)
        assert parts(q1 / q3) == (k1m / q3m, k1u / q3u)
        assert parts(q3 * q1) == (q3m * k1m, q3u * k1u)
        assert parts(q3 / q1) == (q3m / k1m, q3u / k1u)
        assert parts(q1**-1) == (k1m**-1, k1u**-1)
        assert parts(q1**2) == (k1m**2, k1u**2)
        assert parts(q1**-2) == (k1m**-2, k1u**-2)

    def test_issues86b(self, module_registry):
        T1 = module_registry.Quantity(200, module_registry.degC)
        # T1 = 200.0 * module_registry.degC
        T2 = T1.to(module_registry.kelvin)
        m = 132.9054519 * module_registry.amu
        v1 = 2 * module_registry.k * T1 / m
        v2 = 2 * module_registry.k * T2 / m

        helpers.assert_quantity_almost_equal(v1, v2)
        helpers.assert_quantity_almost_equal(v1, v2.to_base_units())
        helpers.assert_quantity_almost_equal(v1.to_base_units(), v2)
        helpers.assert_quantity_almost_equal(v1.to_base_units(), v2.to_base_units())

    @pytest.mark.xfail
    def test_issue86c(self, module_registry):
        module_registry.autoconvert_offset_to_baseunit = True
        T = module_registry.degC
        T = 100.0 * T
        helpers.assert_quantity_almost_equal(
            module_registry.k * 2 * T, module_registry.k * (2 * T)
        )

    def test_issue93(self, module_registry):
        x = 5 * module_registry.meter
        assert isinstance(x.magnitude, int)
        y = 0.1 * module_registry.meter
        assert isinstance(y.magnitude, float)
        z = 5 * module_registry.meter
        assert isinstance(z.magnitude, int)
        z += y
        assert isinstance(z.magnitude, float)

        helpers.assert_quantity_almost_equal(x + y, 5.1 * module_registry.meter)
        helpers.assert_quantity_almost_equal(z, 5.1 * module_registry.meter)

    def test_issue104(self, module_registry):

        x = [
            module_registry("1 meter"),
            module_registry("1 meter"),
            module_registry("1 meter"),
        ]
        y = [module_registry("1 meter")] * 3

        def summer(values):
            if not values:
                return 0
            total = values[0]
            for v in values[1:]:
                total += v

            return total

        helpers.assert_quantity_almost_equal(
            summer(x), module_registry.Quantity(3, "meter")
        )
        helpers.assert_quantity_almost_equal(x[0], module_registry.Quantity(1, "meter"))
        helpers.assert_quantity_almost_equal(
            summer(y), module_registry.Quantity(3, "meter")
        )
        helpers.assert_quantity_almost_equal(y[0], module_registry.Quantity(1, "meter"))

    def test_issue105(self, module_registry):

        func = module_registry.parse_unit_name
        val = list(func("meter"))
        assert list(func("METER")) == []
        assert val == list(func("METER", False))

        for func in (module_registry.get_name, module_registry.parse_expression):
            val = func("meter")
            with pytest.raises(AttributeError):
                func("METER")
            assert val == func("METER", False)

    @helpers.requires_numpy
    def test_issue127(self, module_registry):
        q = [1.0, 2.0, 3.0, 4.0] * module_registry.meter
        q[0] = np.nan
        assert q[0] != 1.0
        assert math.isnan(q[0].magnitude)
        q[1] = float("NaN")
        assert q[1] != 2.0
        assert math.isnan(q[1].magnitude)

    def test_issue170(self):
        Q_ = UnitRegistry().Quantity
        q = Q_("1 kHz") / Q_("100 Hz")
        iq = int(q)
        assert iq == 10
        assert isinstance(iq, int)

    def test_angstrom_creation(self, module_registry):
        module_registry.Quantity(2, "Å")

    def test_alternative_angstrom_definition(self, module_registry):
        module_registry.Quantity(2, "\u212B")

    def test_micro_creation(self, module_registry):
        module_registry.Quantity(2, "µm")

    @helpers.requires_numpy
    def test_issue171_real_imag(self, module_registry):
        qr = [1.0, 2.0, 3.0, 4.0] * module_registry.meter
        qi = [4.0, 3.0, 2.0, 1.0] * module_registry.meter
        q = qr + 1j * qi
        helpers.assert_quantity_equal(q.real, qr)
        helpers.assert_quantity_equal(q.imag, qi)

    @helpers.requires_numpy
    def test_issue171_T(self, module_registry):
        a = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        q1 = a * module_registry.meter
        q2 = a.T * module_registry.meter
        helpers.assert_quantity_equal(q1.T, q2)

    @helpers.requires_numpy
    def test_issue250(self, module_registry):
        a = module_registry.V
        b = module_registry.mV
        assert np.float16(a / b) == 1000.0
        assert np.float32(a / b) == 1000.0
        assert np.float64(a / b) == 1000.0
        if "float128" in dir(np):
            assert np.float128(a / b) == 1000.0

    def test_issue252(self):
        ur = UnitRegistry()
        q = ur("3 F")
        t = copy.deepcopy(q)
        u = t.to(ur.mF)
        helpers.assert_quantity_equal(q.to(ur.mF), u)

    def test_issue323(self, module_registry):
        from fractions import Fraction as F

        assert (self.Q_(F(2, 3), "s")).to("ms") == self.Q_(F(2000, 3), "ms")
        assert (self.Q_(F(2, 3), "m")).to("km") == self.Q_(F(1, 1500), "km")

    def test_issue339(self, module_registry):
        q1 = module_registry("")
        assert q1.magnitude == 1
        assert q1.units == module_registry.dimensionless
        q2 = module_registry("1 dimensionless")
        assert q1 == q2

    def test_issue354_356_370(self, module_registry):
        assert (
            "{:~}".format(1 * module_registry.second / module_registry.millisecond)
            == "1.0 s / ms"
        )
        assert "{:~}".format(1 * module_registry.count) == "1 count"
        assert "{:~}".format(1 * module_registry("MiB")) == "1 MiB"

    def test_issue468(self, module_registry):
        @module_registry.wraps("kg", "meter")
        def f(x):
            return x

        x = module_registry.Quantity(1.0, "meter")
        y = f(x)
        z = x * y
        assert z == module_registry.Quantity(1.0, "meter * kilogram")

    @helpers.requires_numpy
    def test_issue482(self, module_registry):
        q = module_registry.Quantity(1, module_registry.dimensionless)
        qe = np.exp(q)
        assert isinstance(qe, module_registry.Quantity)

    @helpers.requires_numpy
    def test_issue483(self, module_registry):

        a = np.asarray([1, 2, 3])
        q = [1, 2, 3] * module_registry.dimensionless
        p = (q**q).m
        np.testing.assert_array_equal(p, a**a)

    def test_issue507(self, module_registry):
        # leading underscore in unit works with numbers
        module_registry.define("_100km = 100 * kilometer")
        battery_ec = 16 * module_registry.kWh / module_registry._100km  # noqa: F841
        # ... but not with text
        module_registry.define("_home = 4700 * kWh / year")
        with pytest.raises(AttributeError):
            home_elec_power = 1 * module_registry._home  # noqa: F841
        # ... or with *only* underscores
        module_registry.define("_ = 45 * km")
        with pytest.raises(AttributeError):
            one_blank = 1 * module_registry._  # noqa: F841

    def test_issue523(self, module_registry):
        src, dst = UnitsContainer({"meter": 1}), UnitsContainer({"degF": 1})
        value = 10.0
        convert = module_registry.convert
        with pytest.raises(DimensionalityError):
            convert(value, src, dst)
        with pytest.raises(DimensionalityError):
            convert(value, dst, src)

    def test_issue532(self, module_registry):
        @module_registry.check(module_registry(""))
        def f(x):
            return 2 * x

        assert f(module_registry.Quantity(1, "")) == 2
        with pytest.raises(DimensionalityError):
            f(module_registry.Quantity(1, "m"))

    def test_issue625a(self, module_registry):
        Q_ = module_registry.Quantity
        from math import sqrt

        @module_registry.wraps(
            module_registry.second,
            (
                module_registry.meters,
                module_registry.meters / module_registry.second**2,
            ),
        )
        def calculate_time_to_fall(height, gravity=Q_(9.8, "m/s^2")):
            """Calculate time to fall from a height h with a default gravity.

            By default, the gravity is assumed to be earth gravity,
            but it can be modified.

            d = .5 * g * t**2
            t = sqrt(2 * d / g)

            Parameters
            ----------
            height :

            gravity :
                 (Default value = Q_(9.8)
            "m/s^2") :


            Returns
            -------

            """
            return sqrt(2 * height / gravity)

        lunar_module_height = Q_(10, "m")
        t1 = calculate_time_to_fall(lunar_module_height)
        # print(t1)
        assert round(abs(t1 - Q_(1.4285714285714286, "s")), 7) == 0

        moon_gravity = Q_(1.625, "m/s^2")
        t2 = calculate_time_to_fall(lunar_module_height, moon_gravity)
        assert round(abs(t2 - Q_(3.508232077228117, "s")), 7) == 0

    def test_issue625b(self, module_registry):
        Q_ = module_registry.Quantity

        @module_registry.wraps("=A*B", ("=A", "=B"))
        def get_displacement(time, rate=Q_(1, "m/s")):
            """Calculates displacement from a duration and default rate.

            Parameters
            ----------
            time :

            rate :
                 (Default value = Q_(1)
            "m/s") :


            Returns
            -------

            """
            return time * rate

        d1 = get_displacement(Q_(2, "s"))
        assert round(abs(d1 - Q_(2, "m")), 7) == 0

        d2 = get_displacement(Q_(2, "s"), Q_(1, "deg/s"))
        assert round(abs(d2 - Q_(2, " deg")), 7) == 0

    def test_issue625c(self):
        u = UnitRegistry()

        @u.wraps("=A*B*C", ("=A", "=B", "=C"))
        def get_product(a=2 * u.m, b=3 * u.m, c=5 * u.m):
            return a * b * c

        assert get_product(a=3 * u.m) == 45 * u.m**3
        assert get_product(b=2 * u.m) == 20 * u.m**3
        assert get_product(c=1 * u.dimensionless) == 6 * u.m**2

    def test_issue655a(self, module_registry):
        distance = 1 * module_registry.m
        time = 1 * module_registry.s
        velocity = distance / time
        assert distance.check("[length]")
        assert not distance.check("[time]")
        assert velocity.check("[length] / [time]")
        assert velocity.check("1 / [time] * [length]")

    def test_issue655b(self, module_registry):
        Q_ = module_registry.Quantity

        @module_registry.check("[length]", "[length]/[time]^2")
        def pendulum_period(length, G=Q_(1, "standard_gravity")):
            # print(length)
            return (2 * math.pi * (length / G) ** 0.5).to("s")

        length = Q_(1, module_registry.m)
        # Assume earth gravity
        t = pendulum_period(length)
        assert round(abs(t - Q_("2.0064092925890407 second")), 7) == 0
        # Use moon gravity
        moon_gravity = Q_(1.625, "m/s^2")
        t = pendulum_period(length, moon_gravity)
        assert round(abs(t - Q_("4.928936075204336 second")), 7) == 0

    def test_issue783(self, module_registry):
        assert not module_registry("g") == []

    def test_issue856(self, module_registry):
        ph1 = ParserHelper(scale=123)
        ph2 = copy.deepcopy(ph1)
        assert ph2.scale == ph1.scale

        module_registry1 = UnitRegistry()
        module_registry2 = copy.deepcopy(module_registry1)
        # Very basic functionality test
        assert module_registry2("1 t").to("kg").magnitude == 1000

    def test_issue856b(self):
        # Test that, after a deepcopy(), the two UnitRegistries are
        # independent from each other
        ureg1 = UnitRegistry()
        ureg2 = copy.deepcopy(ureg1)
        ureg1.define("test123 = 123 kg")
        ureg2.define("test123 = 456 kg")
        assert ureg1("1 test123").to("kg").magnitude == 123
        assert ureg2("1 test123").to("kg").magnitude == 456

    def test_issue876(self):
        # Same hash must not imply equality.

        # As an implementation detail of CPython, hash(-1) == hash(-2).
        # This test is useless in potential alternative Python implementations where
        # hash(-1) != hash(-2); one would need to find hash collisions specific for each
        # implementation

        a = UnitsContainer({"[mass]": -1})
        b = UnitsContainer({"[mass]": -2})
        c = UnitsContainer({"[mass]": -3})

        # Guarantee working on alternative Python implementations
        assert (hash(-1) == hash(-2)) == (hash(a) == hash(b))
        assert (hash(-1) == hash(-3)) == (hash(a) == hash(c))
        assert a != b
        assert a != c

    def test_issue902(self):
        module_registry = UnitRegistry(auto_reduce_dimensions=True)
        velocity = 1 * module_registry.m / module_registry.s
        cross_section = 1 * module_registry.um**2
        result = cross_section / velocity
        assert result == 1e-12 * module_registry.m * module_registry.s

    def test_issue912(self, module_registry):
        """pprint.pformat() invokes sorted() on large sets and frozensets and graciously
        handles TypeError, but not generic Exceptions. This test will fail if
        pint.DimensionalityError stops being a subclass of TypeError.

        Parameters
        ----------

        Returns
        -------

        """
        meter_units = module_registry.get_compatible_units(module_registry.meter)
        hertz_units = module_registry.get_compatible_units(module_registry.hertz)
        pprint.pformat(meter_units | hertz_units)

    def test_issue932(self, module_registry):
        q = module_registry.Quantity("1 kg")
        with pytest.raises(DimensionalityError):
            q.to("joule")
        module_registry.enable_contexts("energy", *(Context() for _ in range(20)))
        q.to("joule")
        module_registry.disable_contexts()
        with pytest.raises(DimensionalityError):
            q.to("joule")

    def test_issue960(self, module_registry):
        q = (1 * module_registry.nanometer).to_compact("micrometer")
        assert q.units == module_registry.nanometer
        assert q.magnitude == 1

    def test_issue1032(self, module_registry):
        class MultiplicativeDictionary(dict):
            def __rmul__(self, other):
                return self.__class__(
                    {key: value * other for key, value in self.items()}
                )

        q = 3 * module_registry.s
        d = MultiplicativeDictionary({4: 5, 6: 7})
        assert q * d == MultiplicativeDictionary(
            {4: 15 * module_registry.s, 6: 21 * module_registry.s}
        )
        with pytest.raises(TypeError):
            d * q

    @helpers.requires_numpy
    def test_issue973(self, module_registry):
        """Verify that an empty array Quantity can be created through multiplication."""
        q0 = np.array([]) * module_registry.m  # by Unit
        q1 = np.array([]) * module_registry("m")  # by Quantity
        assert isinstance(q0, module_registry.Quantity)
        assert isinstance(q1, module_registry.Quantity)
        assert len(q0) == len(q1) == 0

    def test_issue1058(self, module_registry):
        """verify that auto-reducing quantities with three or more units
        of same base type succeeds"""
        q = 1 * module_registry.mg / module_registry.g / module_registry.kg
        q.ito_reduced_units()
        assert isinstance(q, module_registry.Quantity)

    def test_issue1062_issue1097(self):
        # Must not be used by any other tests
        ureg = UnitRegistry()
        assert "nanometer" not in ureg._units
        for i in range(5):
            ctx = Context.from_lines(["@context _", "cal = 4 J"])
            with ureg.context("sp", ctx):
                q = ureg.Quantity(1, "nm")
                q.to("J")

    def test_issue1066(self):
        """Verify calculations for offset units of higher dimension"""
        ureg = UnitRegistry()
        ureg.define("barga = 1e5 * Pa; offset: 1e5")
        ureg.define("bargb = 1 * bar; offset: 1")
        q_4barg_a = ureg.Quantity(4, ureg.barga)
        q_4barg_b = ureg.Quantity(4, ureg.bargb)
        q_5bar = ureg.Quantity(5, ureg.bar)
        helpers.assert_quantity_equal(q_4barg_a, q_5bar)
        helpers.assert_quantity_equal(q_4barg_b, q_5bar)

    def test_issue1086(self, module_registry):
        # units with prefixes should correctly test as 'in' the registry
        assert "bits" in module_registry
        assert "gigabits" in module_registry
        assert "meters" in module_registry
        assert "kilometers" in module_registry
        # unknown or incorrect units should test as 'not in' the registry
        assert "magicbits" not in module_registry
        assert "unknownmeters" not in module_registry
        assert "gigatrees" not in module_registry

    def test_issue1112(self):
        ureg = UnitRegistry(
            """
            m = [length]
            g = [mass]
            s = [time]

            ft = 0.305 m
            lb = 454 g

            @context c1
                [time]->[length] : value * 10 m/s
            @end
            @context c2
                ft = 0.3 m
            @end
            @context c3
                lb = 500 g
            @end
            """.splitlines()
        )
        ureg.enable_contexts("c1")
        ureg.enable_contexts("c2")
        ureg.enable_contexts("c3")

    @helpers.requires_numpy
    def test_issue1144_1102(self, module_registry):
        # Performing operations shouldn't modify the original objects
        # Issue 1144
        ddc = "delta_degree_Celsius"
        q1 = module_registry.Quantity([-287.78, -32.24, -1.94], ddc)
        q2 = module_registry.Quantity(70.0, "degree_Fahrenheit")
        q1 - q2
        assert all(q1 == module_registry.Quantity([-287.78, -32.24, -1.94], ddc))
        assert q2 == module_registry.Quantity(70.0, "degree_Fahrenheit")
        q2 - q1
        assert all(q1 == module_registry.Quantity([-287.78, -32.24, -1.94], ddc))
        assert q2 == module_registry.Quantity(70.0, "degree_Fahrenheit")
        # Issue 1102
        val = [30.0, 45.0, 60.0] * module_registry.degree
        val == 1
        1 == val
        assert all(val == module_registry.Quantity([30.0, 45.0, 60.0], "degree"))
        # Test for another bug identified by searching on "_convert_magnitude"
        q2 = module_registry.Quantity(3, "degree_Kelvin")
        q1 - q2
        assert all(q1 == module_registry.Quantity([-287.78, -32.24, -1.94], ddc))

    @helpers.requires_numpy
    def test_issue_1136(self, module_registry):
        assert (
            2 ** module_registry.Quantity([2, 3], "") == 2 ** np.array([2, 3])
        ).all()

        with pytest.raises(DimensionalityError):
            2 ** module_registry.Quantity([2, 3], "m")

    def test_issue1175(self):
        import pickle

        foo1 = get_application_registry().Quantity(1, "s")
        foo2 = pickle.loads(pickle.dumps(foo1))
        assert isinstance(foo1, foo2.__class__)
        assert isinstance(foo2, foo1.__class__)

    @helpers.requires_numpy
    def test_issue1174(self, module_registry):
        q = [1.0, -2.0, 3.0, -4.0] * module_registry.meter
        assert np.sign(q[0].magnitude)
        assert np.sign(q[1].magnitude)

    @helpers.requires_numpy()
    def test_issue_1185(self, module_registry):
        # Test __pow__
        foo = module_registry.Quantity((3, 3), "mm / cm")
        assert np.allclose(
            foo ** module_registry.Quantity([2, 3], ""), 0.3 ** np.array([2, 3])
        )
        assert np.allclose(foo ** np.array([2, 3]), 0.3 ** np.array([2, 3]))
        assert np.allclose(np.array([2, 3]) ** foo, np.array([2, 3]) ** 0.3)
        # Test __ipow__
        foo **= np.array([2, 3])
        assert np.allclose(foo, 0.3 ** np.array([2, 3]))
        # Test __rpow__
        assert np.allclose(
            np.array((1, 1)).__rpow__(module_registry.Quantity((2, 3), "mm / cm")),
            np.array((0.2, 0.3)),
        )
        assert np.allclose(
            module_registry.Quantity((20, 20), "mm / cm").__rpow__(
                np.array((0.2, 0.3))
            ),
            np.array((0.04, 0.09)),
        )

    @helpers.requires_uncertainties()
    def test_issue_1300(self):
        module_registry = UnitRegistry()
        module_registry.default_format = "~P"
        m = module_registry.Measurement(1, 0.1, "meter")
        assert m.default_format == "~P"


if np is not None:

    @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
    @pytest.mark.parametrize(
        "callable",
        [
            lambda x: np.sin(x / x.units),  # Issue 399
            lambda x: np.cos(x / x.units),  # Issue 399
            np.isfinite,  # Issue 481
            np.shape,  # Issue 509
            np.size,  # Issue 509
            np.sqrt,  # Issue 622
            lambda x: x.mean(),  # Issue 678
            lambda x: x.copy(),  # Issue 678
            np.array,
            lambda x: x.conjugate,
        ],
    )
    @pytest.mark.parametrize(
        "q_params",
        [
            pytest.param((1, "m"), id="python scalar int"),
            pytest.param(([1, 2, 3, 4], "m"), id="array int"),
            pytest.param(([1], "m", 0), id="numpy scalar int"),
            pytest.param((1.0, "m"), id="python scalar float"),
            pytest.param(([1.0, 2.0, 3.0, 4.0], "m"), id="array float"),
            pytest.param(([1.0], "m", 0), id="numpy scalar float"),
        ],
    )
    def test_issue925(module_registry, callable, q_params):
        # Test for immutability of type
        if len(q_params) == 3:
            q_params, el = q_params[:2], q_params[2]
        else:
            el = None
        q = module_registry.Quantity(*q_params)
        if el is not None:
            q = q[el]
        type_before = type(q._magnitude)
        callable(q)
        assert isinstance(q._magnitude, type_before)

    def test_issue1498(tmp_path):
        def0 = tmp_path / "def0.txt"
        def1 = tmp_path / "def1.txt"
        def2 = tmp_path / "def2.txt"

        # A file that defines a new base unit and uses it in a context
        def0.write_text(
            """
        foo = [FOO]

        @context BAR
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end
        """
        )

        # A file that defines a new base unit, then imports another file…
        def1.write_text(
            f"""
        foo = [FOO]

        @import {str(def2)}
        """
        )

        # …that, in turn, uses it in a context
        def2.write_text(
            """
        @context BAR
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end
        """
        )

        # Succeeds with pint 0.18; fails with pint 0.19
        ureg1 = UnitRegistry()
        ureg1.load_definitions(def1)  # ← FAILS

        assert 12.0 == ureg1("1.2 foo").to("kg", "BAR").magnitude

    def test_issue1498b(tmp_path):
        def0 = tmp_path / "def0.txt"
        def1 = tmp_path / "def1.txt"
        def1_1 = tmp_path / "def1_1.txt"
        def1_2 = tmp_path / "def1_2.txt"
        def2 = tmp_path / "def2.txt"

        # A file that defines a new base unit and uses it in a context
        def0.write_text(
            """
        foo = [FOO]

        @context BAR
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end

        @import def1.txt
        @import def2.txt
        """
        )

        # A file that defines a new base unit, then imports another file…
        def1.write_text(
            """
        @import def1_1.txt
        @import def1_2.txt
        """
        )

        def1_1.write_text(
            """
        @context BAR1_1
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end
        """
        )

        def1_2.write_text(
            """
        @context BAR1_2
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end
        """
        )

        # …that, in turn, uses it in a context
        def2.write_text(
            """
        @context BAR2
            [FOO] -> [mass]: value / foo * 10.0 kg
        @end
        """
        )

        # Succeeds with pint 0.18; fails with pint 0.19
        ureg1 = UnitRegistry()
        ureg1.load_definitions(def0)  # ← FAILS

        assert 12.0 == ureg1("1.2 foo").to("kg", "BAR").magnitude
