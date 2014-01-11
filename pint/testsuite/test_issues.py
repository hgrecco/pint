# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

from pint import UnitRegistry
from pint.unit import UnitsContainer
from pint.util import ParserHelper

from pint.testsuite import HAS_NUMPY, np, TestCase, ndarray, NUMPY_VER

class TestIssues(TestCase):

    FORCE_NDARRAY = False

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
        ureg.define('molar = mole / liter = M')
        t = 4 * ureg['mM']
        self.assertEqual(t.magnitude, 4)
        self.assertEqual(t.units, UnitsContainer(millimolar=1))
        self.assertEqual(t.to('mole / liter'), 4e-3 * ureg['M'])

    def test_issue52(self):
        u1 = UnitRegistry()
        u2 = UnitRegistry()
        q1 = u1.meter
        q2 = u2.meter
        import operator as op
        for fun in (op.add, op.iadd,
                    op.sub, op.isub,
                    op.mul, op.imul,
                    op.floordiv, op.ifloordiv,
                    op.truediv, op.itruediv):
            self.assertRaises(ValueError, fun, q1, q2)

    def test_issue54(self):
        ureg = UnitRegistry()
        self.assertEqual((ureg.km/ureg.m + 1).magnitude, 1001)

    def test_issue54_related(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.km/ureg.m, 1000)
        self.assertEqual(1000, ureg.km/ureg.m)
        self.assertLess(900, ureg.km/ureg.m)
        self.assertGreater(1100, ureg.km/ureg.m)

    def test_issue61(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        for value in ({}, {'a': 3}, '', None, True, False):
            self.assertRaises(ValueError, Q_, value)
            self.assertRaises(ValueError, Q_, value, 'meter')

    @unittest.skipIf(HAS_NUMPY, 'Numpy present')
    def test_issue61_notNP(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        for value in ([1, 2, 3], (1, 2, 3)):
            self.assertRaises(ValueError, Q_, value)
            self.assertRaises(ValueError, Q_, value, 'meter')

    def test_issue66(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_dimensionality(UnitsContainer({'[temperature]': 1})),
                         UnitsContainer({'[temperature]': 1}))
        self.assertEqual(ureg.get_dimensionality(ureg.kelvin.units),
                         UnitsContainer({'[temperature]': 1}))
        self.assertEqual(ureg.get_dimensionality(ureg.degC.units),
                         UnitsContainer({'[temperature]': 1}))

    def test_issue66b(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_base_units(ureg.kelvin.units),
                         (None, UnitsContainer({'kelvin': 1})))
        self.assertEqual(ureg.get_base_units(ureg.degC.units),
                         (None, UnitsContainer({'kelvin': 1})))

    def test_issue69(self):
        ureg = UnitRegistry()
        q = ureg['m'].to(ureg['in'])
        self.assertEqual(q, ureg['m'].to('in'))

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

        self.assertAlmostEqual(va.to_base_units(), vb.to_base_units())

    def test_issue86(self):
        def parts(q):
            return q.magnitude, q.units

        ureg = UnitRegistry()

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
        self.assertEqual(parts(q1 **  1), (k1m **  1, k1u **  1))
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

        self.assertAlmostEqual(v1, v2)
        self.assertAlmostEqual(v1, v2.to_base_units())
        self.assertAlmostEqual(v1.to_base_units(), v2)
        self.assertAlmostEqual(v1.to_base_units(), v2.to_base_units())

    def _test_issueXX(self):
        ureg = UnitRegistry()
        try:
            ureg.convert(1, ureg.degC, ureg.kelvin * ureg.meter / ureg.nanometer)
        except:
            self.assertTrue(False,
                            'Error while trying to convert {} to {}'.format(ureg.degC, ureg.kelvin * ureg.meter / ureg.nanometer))


@unittest.skipUnless(HAS_NUMPY, 'Numpy not present')
class TestIssuesNP(TestCase):

    FORCE_NDARRAY = False

    @unittest.expectedFailure
    def test_issue37(self):
        x = np.ma.masked_array([1, 2, 3], mask=[True, True, False])
        ureg = UnitRegistry()
        q = ureg.meter * x
        self.assertIsInstance(q, ureg.Quantity)
        self.assertSequenceEqual(q.magnitude, x)
        self.assertEquals(q.units, ureg.meter.units)
        q = x * ureg.meter
        self.assertIsInstance(q, ureg.Quantity)
        self.assertSequenceEqual(q.magnitude, x)
        self.assertEquals(q.units, ureg.meter.units)

        m = np.ma.masked_array(2 * np.ones(3,3))
        qq = q * m
        self.assertIsInstance(qq, ureg.Quantity)
        self.assertSequenceEqual(qq.magnitude, x * m)
        self.assertEquals(qq.units, ureg.meter.units)
        qq = m * q
        self.assertIsInstance(qq, ureg.Quantity)
        self.assertSequenceEqual(qq.magnitude, x * m)
        self.assertEquals(qq.units, ureg.meter.units)

    @unittest.expectedFailure
    def test_issue39(self):
        x = np.matrix([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        ureg = UnitRegistry()
        q = ureg.meter * x
        self.assertIsInstance(q, ureg.Quantity)
        self.assertSequenceEqual(q.magnitude, x)
        self.assertEquals(q.units, ureg.meter.units)
        q = x * ureg.meter
        self.assertIsInstance(q, ureg.Quantity)
        self.assertSequenceEqual(q.magnitude, x)
        self.assertEquals(q.units, ureg.meter.units)

        m = np.matrix(2 * np.ones(3,3))
        qq = q * m
        self.assertIsInstance(qq, ureg.Quantity)
        self.assertSequenceEqual(qq.magnitude, x * m)
        self.assertEquals(qq.units, ureg.meter.units)
        qq = m * q
        self.assertIsInstance(qq, ureg.Quantity)
        self.assertSequenceEqual(qq.magnitude, x * m)
        self.assertEquals(qq.units, ureg.meter.units)

    def test_issue44(self):
        ureg = UnitRegistry()
        x = 4. * ureg.dimensionless
        np.sqrt(x)
        self.assertAlmostEqual(np.sqrt([4.] * ureg.dimensionless), [2.] * ureg.dimensionless)
        self.assertAlmostEqual(np.sqrt(4. * ureg.dimensionless), 2. * ureg.dimensionless)

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
        m = ureg['m**0.5']
        self.assertEqual(str(m.units), 'meter ** 0.5')

    def test_issue74(self):
        ureg = UnitRegistry()
        v1 = np.asarray([1, 2, 3])
        v2 = np.asarray([3, 2, 1])
        q1 = v1 * ureg.ms
        q2 = v2 * ureg.ms

        self.assertSequenceEqual(q1 < q2, v1 < v2)
        self.assertSequenceEqual(q1 > q2, v1 > v2)

        self.assertSequenceEqual(q1 <= q2, v1 <= v2)
        self.assertSequenceEqual(q1 >= q2, v1 >= v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * ureg.s
        v2s = q2s.to('ms').magnitude

        self.assertSequenceEqual(q1 < q2s, v1 < v2s)
        self.assertSequenceEqual(q1 > q2s, v1 > v2s)

        self.assertSequenceEqual(q1 <= q2s, v1 <= v2s)
        self.assertSequenceEqual(q1 >= q2s, v1 >= v2s)

    def test_issue75(self):
        ureg = UnitRegistry()
        v1 = np.asarray([1, 2, 3])
        v2 = np.asarray([3, 2, 1])
        q1 = v1 * ureg.ms
        q2 = v2 * ureg.ms

        self.assertSequenceEqual(q1 == q2, v1 == v2)
        self.assertSequenceEqual(q1 != q2, v1 != v2)

        q2s = np.asarray([0.003, 0.002, 0.001]) * ureg.s
        v2s = q2s.to('ms').magnitude

        self.assertSequenceEqual(q1 == q2s, v1 == v2s)
        self.assertSequenceEqual(q1 != q2s, v1 != v2s)
