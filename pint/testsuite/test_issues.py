# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

from pint import UnitRegistry
from pint.unit import UnitsContainer
from pint.util import ParserHelper

from pint.testsuite import HAS_NUMPY, np, TestCase

class TestIssues(unittest.TestCase):

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
