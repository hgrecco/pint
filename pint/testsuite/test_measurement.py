# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

from pint import Measurement
from pint.testsuite import TestCase


class TestMeasurement(TestCase):

    FORCE_NDARRAY = False

    def test_build(self):
        v, u = self.Q_(4.0, 's'), self.Q_(.1, 's')

        ms = (Measurement(v, u),
              v.plus_minus(.1),
              v.plus_minus(0.025, True),
              v.plus_minus(u))

        for m in ms:
            self.assertEqual(m.value, v)
            self.assertEqual(m.error, u)
            self.assertEqual(m.rel, m.error / abs(m.value))

    def _test_format(self):
        v, u = self.Q_(4.0, 's'), self.Q_(.1, 's')
        m = Measurement(v, u)
        print(str(m))
        print(repr(m))
        print('{:!s}'.format(m))
        print('{:!r}'.format(m))
        print('{:!l}'.format(m))
        print('{:!p}'.format(m))
        print('{:.02f!l}'.format(m))
        print('{:.02f!p}'.format(m))

    def test_raise_build(self):
        v, u = self.Q_(1.0, 's'), self.Q_(.1, 's')
        o = self.Q_(.1, 'm')

        self.assertRaises(ValueError, Measurement, u, 1)
        self.assertRaises(ValueError, Measurement, u, o)
        self.assertRaises(ValueError, v.plus_minus, o)
        self.assertRaises(ValueError, v.plus_minus, u, True)

    def test_propagate_linear(self):

        v1, u1 = self.Q_(8.0, 's'), self.Q_(.7, 's')
        v2, u2 = self.Q_(5.0, 's'), self.Q_(.6, 's')
        v2, u3 = self.Q_(-5.0, 's'), self.Q_(.6, 's')

        m1 = v1.plus_minus(u1)
        m2 = v2.plus_minus(u2)
        m3 = v2.plus_minus(u3)

        for factor, m in zip((3, -3, 3, -3), (m1, m3, m1, m3)):
            r = factor * m
            self.assertAlmostEqual(r.value, factor * m.value)
            self.assertAlmostEqual(r.error ** 2.0, (factor * m.error) **2.0)

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml + mr
            self.assertAlmostEqual(r.value, ml.value + mr.value)
            self.assertAlmostEqual(r.error ** 2.0, ml.error **2.0 + mr.error ** 2.0)

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml - mr
            self.assertAlmostEqual(r.value, ml.value + mr.value)
            self.assertAlmostEqual(r.error ** 2.0, ml.error **2.0 + mr.error ** 2.0)

    def test_propagate_product(self):

        v1, u1 = self.Q_(8.0, 's'), self.Q_(.7, 's')
        v2, u2 = self.Q_(5.0, 's'), self.Q_(.6, 's')
        v2, u3 = self.Q_(-5.0, 's'), self.Q_(.6, 's')

        m1 = v1.plus_minus(u1)
        m2 = v2.plus_minus(u2)
        m3 = v2.plus_minus(u3)

        m4 = (2.3 * self.ureg.meter).plus_minus(0.1)
        m5 = (1.4 * self.ureg.meter).plus_minus(0.2)

        for ml, mr in zip((m1, m1, m1, m3, m4), (m1, m2, m3, m3, m5)):
            r = ml * mr
            self.assertAlmostEqual(r.value, ml.value * mr.value)
            self.assertAlmostEqual(r.rel ** 2.0, ml.rel ** 2.0 + mr.rel ** 2.0)

        for ml, mr in zip((m1, m1, m1, m3, m4), (m1, m2, m3, m3, m5)):
            r = ml / mr
            self.assertAlmostEqual(r.value, ml.value / mr.value)
            self.assertAlmostEqual(r.rel ** 2.0, ml.rel ** 2.0 + mr.rel ** 2.0)
