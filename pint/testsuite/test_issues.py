# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import math
import copy
import unittest

from pint import UnitRegistry
from pint.unit import UnitsContainer
from pint.util import ParserHelper

from pint.compat import np, long_type
from pint.errors import UndefinedUnitError, DimensionalityError
from pint.testsuite import QuantityTestCase, helpers


class TestIssues(QuantityTestCase):

    FORCE_NDARRAY = False

    def setup(self):
        self.ureg.autoconvert_offset_to_baseunit = False

    @unittest.expectedFailure
    def test_issue25(self):
        x = ParserHelper.from_string('10 %')
        self.assertEqual(x, ParserHelper(10, {'%': 1}))
        x = ParserHelper.from_string('10 ‰')
        self.assertEqual(x, ParserHelper(10, {'‰': 1}))
        ureg = UnitRegistry()
        ureg.define('percent = [fraction]; offset: 0 = %')
        ureg.define('permille = percent / 10 = ‰')
        x = ureg.parse_expression('10 %')
        self.assertEqual(x, ureg.Quantity(10, {'%': 1}))
        y = ureg.parse_expression('10 ‰')
        self.assertEqual(y, ureg.Quantity(10, {'‰': 1}))
        self.assertEqual(x.to('‰'), ureg.Quantity(1, {'‰': 1}))

    def test_issue29(self):
        ureg = UnitRegistry()
        t = 4 * ureg('mM')
        self.assertEqual(t.magnitude, 4)
        self.assertEqual(t._units, UnitsContainer(millimolar=1))
        self.assertEqual(t.to('mole / liter'), 4e-3 * ureg('M'))

    def test_issue52(self):
        u1 = UnitRegistry()
        u2 = UnitRegistry()
        q1 = 1*u1.meter
        q2 = 1*u2.meter
        import operator as op
        for fun in (op.add, op.iadd,
                    op.sub, op.isub,
                    op.mul, op.imul,
                    op.floordiv, op.ifloordiv,
                    op.truediv, op.itruediv):
            self.assertRaises(ValueError, fun, q1, q2)

    def test_issue54(self):
        ureg = UnitRegistry()
        self.assertEqual((1*ureg.km/ureg.m + 1).magnitude, 1001)

    def test_issue54_related(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.km/ureg.m, 1000)
        self.assertEqual(1000, ureg.km/ureg.m)
        self.assertLess(900, ureg.km/ureg.m)
        self.assertGreater(1100, ureg.km/ureg.m)

    def test_issue61(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        for value in ({}, {'a': 3}, None):
            self.assertRaises(TypeError, Q_, value)
            self.assertRaises(TypeError, Q_, value, 'meter')
        self.assertRaises(ValueError, Q_, '', 'meter')
        self.assertRaises(ValueError, Q_, '')

    @helpers.requires_not_numpy()
    def test_issue61_notNP(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        for value in ([1, 2, 3], (1, 2, 3)):
            self.assertRaises(TypeError, Q_, value)
            self.assertRaises(TypeError, Q_, value, 'meter')

    def test_issue66(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_dimensionality(UnitsContainer({'[temperature]': 1})),
                         UnitsContainer({'[temperature]': 1}))
        self.assertEqual(ureg.get_dimensionality(ureg.kelvin),
                         UnitsContainer({'[temperature]': 1}))
        self.assertEqual(ureg.get_dimensionality(ureg.degC),
                         UnitsContainer({'[temperature]': 1}))

    def test_issue66b(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_base_units(ureg.kelvin),
                         (1.0, ureg.Unit(UnitsContainer({'kelvin': 1}))))
        self.assertEqual(ureg.get_base_units(ureg.degC),
                         (1.0, ureg.Unit(UnitsContainer({'kelvin': 1}))))

    def test_issue69(self):
        ureg = UnitRegistry()
        q = ureg('m').to(ureg('in'))
        self.assertEqual(q, ureg('m').to('in'))

    @helpers.requires_uncertainties()
    def test_issue77(self):
        ureg = UnitRegistry()
        acc = (5.0 * ureg('m/s/s')).plus_minus(0.25)
        tim = (37.0 * ureg('s')).plus_minus(0.16)
        dis = acc * tim ** 2 / 2
        self.assertEqual(dis.value, acc.value * tim.value ** 2 / 2)

    def test_issue85(self):
        ureg = UnitRegistry()

        T = 4. * ureg.kelvin
        m = 1. * ureg.amu
        va = 2. * ureg.k * T / m

        try:
            va.to_base_units()
        except:
            self.assertTrue(False, 'Error while trying to get base units for {}'.format(va))

        boltmk = 1.3806488e-23*ureg.J/ureg.K
        vb = 2. * boltmk * T / m

        self.assertQuantityAlmostEqual(va.to_base_units(), vb.to_base_units())

    def test_issue86(self):
        ureg = self.ureg
        ureg.autoconvert_offset_to_baseunit = True

        def parts(q):
            return q.magnitude, q.units

        q1 = 10. * ureg.degC
        q2 = 10. * ureg.kelvin

        k1 = q1.to_base_units()

        q3 = 3. * ureg.meter

        q1m, q1u = parts(q1)
        q2m, q2u = parts(q2)
        q3m, q3u = parts(q3)

        k1m, k1u = parts(k1)

        self.assertEqual(parts(q2 * q3), (q2m * q3m, q2u * q3u))
        self.assertEqual(parts(q2 / q3), (q2m / q3m, q2u / q3u))
        self.assertEqual(parts(q3 * q2), (q3m * q2m, q3u * q2u))
        self.assertEqual(parts(q3 / q2), (q3m / q2m, q3u / q2u))
        self.assertEqual(parts(q2 **  1), (q2m **  1, q2u **  1))
        self.assertEqual(parts(q2 ** -1), (q2m ** -1, q2u ** -1))
        self.assertEqual(parts(q2 **  2), (q2m **  2, q2u **  2))
        self.assertEqual(parts(q2 ** -2), (q2m ** -2, q2u ** -2))

        self.assertEqual(parts(q1 * q3), (k1m * q3m, k1u * q3u))
        self.assertEqual(parts(q1 / q3), (k1m / q3m, k1u / q3u))
        self.assertEqual(parts(q3 * q1), (q3m * k1m, q3u * k1u))
        self.assertEqual(parts(q3 / q1), (q3m / k1m, q3u / k1u))
        self.assertEqual(parts(q1 ** -1), (k1m ** -1, k1u ** -1))
        self.assertEqual(parts(q1 **  2), (k1m **  2, k1u **  2))
        self.assertEqual(parts(q1 ** -2), (k1m ** -2, k1u ** -2))

    def test_issues86b(self):
        ureg = self.ureg

        T1 = 200. * ureg.degC
        T2 = T1.to(ureg.kelvin)
        m = 132.9054519 * ureg.amu
        v1 = 2 * ureg.k * T1 / m
        v2 = 2 * ureg.k * T2 / m

        self.assertQuantityAlmostEqual(v1, v2)
        self.assertQuantityAlmostEqual(v1, v2.to_base_units())
        self.assertQuantityAlmostEqual(v1.to_base_units(), v2)
        self.assertQuantityAlmostEqual(v1.to_base_units(), v2.to_base_units())

    @unittest.expectedFailure
    def test_issue86c(self):
        ureg = self.ureg
        ureg.autoconvert_offset_to_baseunit = True
        T = ureg.degC
        T = 100. * T
        self.assertQuantityAlmostEqual(ureg.k*2*T, ureg.k*(2*T))

    def test_issue93(self):
        ureg = UnitRegistry()
        x = 5 * ureg.meter
        self.assertIsInstance(x.magnitude, int)
        y = 0.1 * ureg.meter
        self.assertIsInstance(y.magnitude, float)
        z = 5 * ureg.meter
        self.assertIsInstance(z.magnitude, int)
        z += y
        self.assertIsInstance(z.magnitude, float)

        self.assertQuantityAlmostEqual(x + y, 5.1 * ureg.meter)
        self.assertQuantityAlmostEqual(z, 5.1 * ureg.meter)

    def test_issue523(self):
        ureg = UnitRegistry()
        src, dst = UnitsContainer({'meter': 1}), UnitsContainer({'degF': 1})
        value = 10.
        convert = self.ureg.convert
        self.assertRaises(DimensionalityError, convert, value, src, dst)
        self.assertRaises(DimensionalityError, convert, value, dst, src)

    def _test_issueXX(self):
        ureg = UnitRegistry()
        try:
            ureg.convert(1, ureg.degC, ureg.kelvin * ureg.meter / ureg.nanometer)
        except:
            self.assertTrue(False,
                            'Error while trying to convert {} to {}'.format(ureg.degC, ureg.kelvin * ureg.meter / ureg.nanometer))

    def test_issue121(self):
        sh = (2, 1)
        ureg = UnitRegistry()
        z, v = 0, 2.
        self.assertEqual(z + v * ureg.meter, v * ureg.meter)
        self.assertEqual(z - v * ureg.meter, -v * ureg.meter)
        self.assertEqual(v * ureg.meter + z, v * ureg.meter)
        self.assertEqual(v * ureg.meter - z, v * ureg.meter)

        self.assertEqual(sum([v * ureg.meter, v * ureg.meter]), 2 * v * ureg.meter)

    def test_issue105(self):
        ureg = UnitRegistry()

        func = ureg.parse_unit_name
        val = list(func('meter'))
        self.assertEqual(list(func('METER')), [])
        self.assertEqual(val, list(func('METER', False)))

        for func in (ureg.get_name, ureg.parse_expression):
            val = func('meter')
            self.assertRaises(AttributeError, func, 'METER')
            self.assertEqual(val, func('METER', False))

    def test_issue104(self):
        ureg = UnitRegistry()

        x = [ureg('1 meter'), ureg('1 meter'), ureg('1 meter')]
        y = [ureg('1 meter')] * 3

        def summer(values):
            if not values:
                return 0
            total = values[0]
            for v in values[1:]:
                total += v

            return total

        self.assertQuantityAlmostEqual(summer(x), ureg.Quantity(3, 'meter'))
        self.assertQuantityAlmostEqual(x[0], ureg.Quantity(1, 'meter'))
        self.assertQuantityAlmostEqual(summer(y), ureg.Quantity(3, 'meter'))
        self.assertQuantityAlmostEqual(y[0], ureg.Quantity(1, 'meter'))

    def test_issue170(self):
        Q_ = UnitRegistry().Quantity
        q = Q_('1 kHz')/Q_('100 Hz')
        iq = int(q)
        self.assertEqual(iq, 10)
        self.assertIsInstance(iq, int)

    @helpers.requires_python2()
    def test_issue170b(self):
        Q_ = UnitRegistry().Quantity
        q = Q_('1 kHz')/Q_('100 Hz')
        iq = long(q)
        self.assertEqual(iq, long(10))
        self.assertIsInstance(iq, long)

    def test_angstrom_creation(self):
        ureg = UnitRegistry()
        try:
            ureg.Quantity(2, 'Å')
        except SyntaxError:
            self.fail('Quantity with Å could not be created.')

    def test_alternative_angstrom_definition(self):
        ureg = UnitRegistry()
        try:
            ureg.Quantity(2, '\u212B')
        except UndefinedUnitError:
            self.fail('Quantity with Å could not be created.')

    def test_micro_creation(self):
        ureg = UnitRegistry()
        try:
            ureg.Quantity(2, 'µm')
        except SyntaxError:
            self.fail('Quantity with µ prefix could not be created.')


@helpers.requires_numpy()
class TestIssuesNP(QuantityTestCase):

    FORCE_NDARRAY = False

    @unittest.expectedFailure
    def test_issue37(self):
        x = np.ma.masked_array([1, 2, 3], mask=[True, True, False])
        ureg = UnitRegistry()
        q = ureg.meter * x
        self.assertIsInstance(q, ureg.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        self.assertEqual(q.units, ureg.meter.units)
        q = x * ureg.meter
        self.assertIsInstance(q, ureg.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        self.assertEqual(q.units, ureg.meter.units)

        m = np.ma.masked_array(2 * np.ones(3,3))
        qq = q * m
        self.assertIsInstance(qq, ureg.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        self.assertEqual(qq.units, ureg.meter.units)
        qq = m * q
        self.assertIsInstance(qq, ureg.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        self.assertEqual(qq.units, ureg.meter.units)

    @unittest.expectedFailure
    def test_issue39(self):
        x = np.matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        ureg = UnitRegistry()
        q = ureg.meter * x
        self.assertIsInstance(q, ureg.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        self.assertEqual(q.units, ureg.meter.units)
        q = x * ureg.meter
        self.assertIsInstance(q, ureg.Quantity)
        np.testing.assert_array_equal(q.magnitude, x)
        self.assertEqual(q.units, ureg.meter.units)

        m = np.matrix(2 * np.ones(3,3))
        qq = q * m
        self.assertIsInstance(qq, ureg.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        self.assertEqual(qq.units, ureg.meter.units)
        qq = m * q
        self.assertIsInstance(qq, ureg.Quantity)
        np.testing.assert_array_equal(qq.magnitude, x * m)
        self.assertEqual(qq.units, ureg.meter.units)

    def test_issue44(self):
        ureg = UnitRegistry()
        x = 4. * ureg.dimensionless
        np.sqrt(x)
        self.assertQuantityAlmostEqual(np.sqrt([4.] * ureg.dimensionless), [2.] * ureg.dimensionless)
        self.assertQuantityAlmostEqual(np.sqrt(4. * ureg.dimensionless), 2. * ureg.dimensionless)

    def test_issue45(self):
        import math
        ureg = UnitRegistry()
        self.assertAlmostEqual(math.sqrt(4 * ureg.m/ureg.cm), math.sqrt(4 * 100))
        self.assertAlmostEqual(float(ureg.V / ureg.mV), 1000.)

    def test_issue45b(self):
        ureg = UnitRegistry()
        self.assertAlmostEqual(np.sin([np.pi/2] * ureg.m / ureg.m ), np.sin([np.pi/2] * ureg.dimensionless))
        self.assertAlmostEqual(np.sin([np.pi/2] * ureg.cm / ureg.m ), np.sin([np.pi/2] * ureg.dimensionless * 0.01))

    def test_issue50(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        self.assertEqual(Q_(100), 100 * ureg.dimensionless)
        self.assertEqual(Q_('100'), 100 * ureg.dimensionless)

    def test_issue62(self):
        ureg = UnitRegistry()
        m = ureg('m**0.5')
        self.assertEqual(str(m.units), 'meter ** 0.5')

    def test_issue74(self):
        ureg = UnitRegistry()
        v1 = np.asarray([1., 2., 3.])
        v2 = np.asarray([3., 2., 1.])
        q1 = v1 * ureg.ms
        q2 = v2 * ureg.ms

        np.testing.assert_array_equal(q1 < q2, v1 < v2)
        np.testing.assert_array_equal(q1 > q2, v1 > v2)

        np.testing.assert_array_equal(q1 <= q2, v1 <= v2)
        np.testing.assert_array_equal(q1 >= q2, v1 >= v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * ureg.s
        v2s = q2s.to('ms').magnitude

        np.testing.assert_array_equal(q1 < q2s, v1 < v2s)
        np.testing.assert_array_equal(q1 > q2s, v1 > v2s)

        np.testing.assert_array_equal(q1 <= q2s, v1 <= v2s)
        np.testing.assert_array_equal(q1 >= q2s, v1 >= v2s)

    def test_issue75(self):
        ureg = UnitRegistry()
        v1 = np.asarray([1., 2., 3.])
        v2 = np.asarray([3., 2., 1.])
        q1 = v1 * ureg.ms
        q2 = v2 * ureg.ms

        np.testing.assert_array_equal(q1 == q2, v1 == v2)
        np.testing.assert_array_equal(q1 != q2, v1 != v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * ureg.s
        v2s = q2s.to('ms').magnitude

        np.testing.assert_array_equal(q1 == q2s, v1 == v2s)
        np.testing.assert_array_equal(q1 != q2s, v1 != v2s)

    def test_issue93(self):
        ureg = UnitRegistry()
        x = 5 * ureg.meter
        self.assertIsInstance(x.magnitude, int)
        y = 0.1 * ureg.meter
        self.assertIsInstance(y.magnitude, float)
        z = 5 * ureg.meter
        self.assertIsInstance(z.magnitude, int)
        z += y
        self.assertIsInstance(z.magnitude, float)

        self.assertQuantityAlmostEqual(x + y, 5.1 * ureg.meter)
        self.assertQuantityAlmostEqual(z, 5.1 * ureg.meter)

    @helpers.requires_numpy_previous_than('1.10')
    def test_issue94(self):
        ureg = UnitRegistry()
        v1 = np.array([5, 5]) * ureg.meter
        v2 = 0.1 * ureg.meter
        v3 = np.array([5, 5]) * ureg.meter
        v3 += v2

        np.testing.assert_array_equal((v1 + v2).magnitude, np.array([5.1, 5.1]))
        np.testing.assert_array_equal(v3.magnitude, np.array([5, 5]))

    @helpers.requires_numpy18()
    def test_issue121(self):
        sh = (2, 1)
        ureg = UnitRegistry()

        z, v = 0, 2.
        self.assertEqual(z + v * ureg.meter, v * ureg.meter)
        self.assertEqual(z - v * ureg.meter, -v * ureg.meter)
        self.assertEqual(v * ureg.meter + z, v * ureg.meter)
        self.assertEqual(v * ureg.meter - z, v * ureg.meter)

        self.assertEqual(sum([v * ureg.meter, v * ureg.meter]), 2 * v * ureg.meter)

        z, v = np.zeros(sh), 2. * np.ones(sh)
        self.assertQuantityEqual(z + v * ureg.meter, v * ureg.meter)
        self.assertQuantityEqual(z - v * ureg.meter, -v * ureg.meter)
        self.assertQuantityEqual(v * ureg.meter + z, v * ureg.meter)
        self.assertQuantityEqual(v * ureg.meter - z, v * ureg.meter)

        z, v = np.zeros((3, 1)), 2. * np.ones(sh)
        for x, y in ((z, v),
                     (z, v * ureg.meter),
                     (v * ureg.meter, z)
                     ):
            try:
                w = x + y
                self.assertTrue(False, "ValueError not raised")
            except ValueError:
                pass
            try:
                w = x - y
                self.assertTrue(False, "ValueError not raised")
            except ValueError:
                pass

    def test_issue127(self):
        q = [1., 2., 3., 4.] * self.ureg.meter
        q[0] = np.nan
        self.assertNotEqual(q[0], 1.)
        self.assertTrue(math.isnan(q[0].magnitude))
        q[1] = float('NaN')
        self.assertNotEqual(q[1], 2.)
        self.assertTrue(math.isnan(q[1].magnitude))

    def test_issue171_real_imag(self):
        qr = [1., 2., 3., 4.] * self.ureg.meter
        qi = [4., 3., 2., 1.] * self.ureg.meter
        q = qr + 1j * qi
        self.assertQuantityEqual(q.real, qr)
        self.assertQuantityEqual(q.imag, qi)

    def test_issue171_T(self):
        a = np.asarray([[1., 2., 3., 4.],[4., 3., 2., 1.]])
        q1 = a * self.ureg.meter
        q2 = a.T * self.ureg.meter
        self.assertQuantityEqual(q1.T, q2)

    def test_issue250(self):
        a = self.ureg.V
        b = self.ureg.mV
        self.assertEqual(np.float16(a/b), 1000.)
        self.assertEqual(np.float32(a/b), 1000.)
        self.assertEqual(np.float64(a/b), 1000.)
        if "float128" in dir(np):
            self.assertEqual(np.float128(a/b), 1000.)

    def test_issue252(self):
        ur = UnitRegistry()
        q = ur("3 F")
        t = copy.deepcopy(q)
        u = t.to(ur.mF)
        self.assertQuantityEqual(q.to(ur.mF), u)

    def test_issue323(self):
        from fractions import Fraction as F
        self.assertEqual((self.Q_(F(2,3), 's')).to('ms'), self.Q_(F(2000,3), 'ms'))
        self.assertEqual((self.Q_(F(2,3), 'm')).to('km'), self.Q_(F(1,1500), 'km'))

    def test_issue339(self):
        q1 = self.ureg('')
        self.assertEqual(q1.magnitude, 1)
        self.assertEqual(q1.units, self.ureg.dimensionless)
        q2 = self.ureg('1 dimensionless')
        self.assertEqual(q1, q2)

    def test_issue354_356_370(self):
        q = 1 * self.ureg.second / self.ureg.millisecond
        self.assertEqual('{0:~}'.format(1 * self.ureg.second / self.ureg.millisecond),
                         '1.0 s / ms')
        self.assertEqual("{0:~}".format(1 * self.ureg.count),
                         '1 count')
        self.assertEqual('{0:~}'.format(1 * self.ureg('MiB')),
                         '1 MiB')

    def test_issue482(self):
        q = self.ureg.Quantity(1, self.ureg.dimensionless)
        qe = np.exp(q)
        self.assertIsInstance(qe, self.ureg.Quantity)

    def test_issue468(self):
        ureg = UnitRegistry()

        @ureg.wraps(('kg'), 'meter')
        def f(x):
            return x

        x = ureg.Quantity(1., 'meter')
        y = f(x)
        z = x * y
        self.assertEquals(z, ureg.Quantity(1., 'meter * kilogram'))

    def test_issue483(self):
        ureg = self.ureg
        a = np.asarray([1, 2, 3])
        q = [1, 2, 3] * ureg.dimensionless
        p = (q ** q).m
        np.testing.assert_array_equal(p, a ** a)

    def test_issue532(self):
        ureg = self.ureg

        @ureg.check(ureg(''))
        def f(x):
            return 2 * x

        self.assertEqual(f(ureg.Quantity(1, '')), 2)
        self.assertRaises(DimensionalityError, f, ureg.Quantity(1, 'm'))

    def test_issue625a(self):
        try:
            from inspect import signature
        except ImportError:
            # Python2 does not have the inspect library. Import the backport.
            from funcsigs import signature

        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        from math import sqrt

        @ureg.wraps(ureg.second, (ureg.meters, ureg.meters/ureg.second**2))
        def calculate_time_to_fall(height, gravity=Q_(9.8, 'm/s^2')):
            """Calculate time to fall from a height h with a default gravity.

            By default, the gravity is assumed to be earth gravity,
            but it can be modified.

            d = .5 * g * t**2
            t = sqrt(2 * d / g)
            """
            return sqrt(2 * height / gravity)

        lunar_module_height = Q_(10, 'm')
        t1 = calculate_time_to_fall(lunar_module_height)
        print(t1)
        self.assertAlmostEqual(t1, Q_(1.4285714285714286, 's'))

        moon_gravity = Q_(1.625, 'm/s^2')
        t2 = calculate_time_to_fall(lunar_module_height, moon_gravity)
        self.assertAlmostEqual(t2, Q_(3.508232077228117, 's'))

    def test_issue625b(self):
        try:
            from inspect import signature
        except ImportError:
            # Python2 does not have the inspect library. Import the backport.
            from funcsigs import signature

        ureg = UnitRegistry()
        Q_ = ureg.Quantity

        @ureg.wraps('=A*B', ('=A', '=B'))
        def get_displacement(time, rate=Q_(1, 'm/s')):
            """Calculates displacement from a duration and default rate.
            """
            return time * rate

        d1 = get_displacement(Q_(2, 's'))
        self.assertAlmostEqual(d1, Q_(2, 'm'))

        d2 = get_displacement(Q_(2, 's'), Q_(1, 'deg/s'))
        self.assertAlmostEqual(d2, Q_(2,' deg'))

    def test_issue625c(self):        
        try:
            from inspect import signature
        except ImportError:
            # Python2 does not have the inspect library. Import the backport.
            from funcsigs import signature

        u = UnitRegistry()

        @u.wraps('=A*B*C', ('=A', '=B', '=C'))
        def get_product(a=2*u.m, b=3*u.m, c=5*u.m):
            return a*b*c

        self.assertEqual(get_product(a=3*u.m), 45*u.m**3)
        self.assertEqual(get_product(b=2*u.m), 20*u.m**3)
        self.assertEqual(get_product(c=1*u.dimensionless), 6*u.m**2)

    def test_issue655a(self):
        ureg = UnitRegistry()
        distance = 1 * ureg.m
        time = 1 * ureg.s
        velocity = distance / time
        self.assertEqual(distance.check('[length]'), True)
        self.assertEqual(distance.check('[time]'), False)
        self.assertEqual(velocity.check('[length] / [time]'), True)
        self.assertEqual(velocity.check('1 / [time] * [length]'), True)

    def test_issue(self):
        import math
        try:
            from inspect import signature
        except ImportError:
            # Python2 does not have the inspect library. Import the backport
            from funcsigs import signature

        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        @ureg.check('[length]', '[length]/[time]^2')
        def pendulum_period(length, G=Q_(1, 'standard_gravity')):
            print(length)
            return (2*math.pi*(length/G)**.5).to('s')
        l = 1 * ureg.m
        # Assume earth gravity
        t = pendulum_period(l)
        self.assertAlmostEqual(t, Q_('2.0064092925890407 second'))
        # Use moon gravity
        moon_gravity = Q_(1.625, 'm/s^2')
        t = pendulum_period(l, moon_gravity)
        self.assertAlmostEqual(t, Q_('4.928936075204336 second'))


