# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

from pint.compat import ufloat
from pint.testsuite import TestCase, unittest

@unittest.skipIf(ufloat is None, 'Uncertainties not installed.')
class TestMeasurement(TestCase):

    FORCE_NDARRAY = False

    def test_simple(self):
        M_ = self.ureg.Measurement
        M_(4.0, 0.1, 's')

    def test_build(self):
        M_ = self.ureg.Measurement
        v, u = self.Q_(4.0, 's'), self.Q_(.1, 's')
        M_(v.magnitude, u.magnitude, 's')
        ms = (M_(v.magnitude, u.magnitude, 's'),
              M_(v, u.magnitude),
              M_(v, u),
              v.plus_minus(.1),
              v.plus_minus(0.025, True),
              v.plus_minus(u),)

        for m in ms:
            self.assertEqual(m.value, v)
            self.assertEqual(m.error, u)
            self.assertEqual(m.rel, m.error / abs(m.value))

    def _test_format(self):
        v, u = self.Q_(4.0, 's'), self.Q_(.1, 's')
        m = self.ureg.Measurement(v, u)
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

        M_ = self.ureg.Measurement
        self.assertRaises(ValueError, M_, v, o)
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
            self.assertAlmostEqual(r.value.magnitude, factor * m.value.magnitude)
            self.assertAlmostEqual(r.error.magnitude, abs(factor * m.error.magnitude))
            self.assertEqual(r.value.units, m.value.units)

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml + mr
            self.assertAlmostEqual(r.value.magnitude, ml.value.magnitude + mr.value.magnitude)
            self.assertAlmostEqual(r.error.magnitude,
                                   ml.error.magnitude + mr.error.magnitude if ml is mr else
                                   (ml.error.magnitude ** 2 + mr.error.magnitude ** 2) ** .5)
            self.assertEqual(r.value.units, ml.value.units)

        for ml, mr in zip((m1, m1, m1, m3), (m1, m2, m3, m3)):
            r = ml - mr
            print(ml, mr, ml is mr, r)
            self.assertAlmostEqual(r.value.magnitude, ml.value.magnitude - mr.value.magnitude)
            self.assertAlmostEqual(r.error.magnitude,
                                   0 if ml is mr else
                                   (ml.error.magnitude ** 2 + mr.error.magnitude ** 2) ** .5)
            self.assertEqual(r.value.units, ml.value.units)

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
            self.assertAlmostEqual(r.value.magnitude, ml.value.magnitude * mr.value.magnitude)
            self.assertEqual(r.value.units, ml.value.units * mr.value.units)

        for ml, mr in zip((m1, m1, m1, m3, m4), (m1, m2, m3, m3, m5)):
            r = ml / mr
            self.assertAlmostEqual(r.value.magnitude, ml.value.magnitude / mr.value.magnitude)
            self.assertEqual(r.value.units, ml.value.units / mr.value.units)
