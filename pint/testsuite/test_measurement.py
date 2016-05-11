# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

from pint.testsuite import QuantityTestCase, helpers


@helpers.requires_not_uncertainties()
class TestNotMeasurement(QuantityTestCase):

    FORCE_NDARRAY = False

    def test_instantiate(self):
        M_ = self.ureg.Measurement
        self.assertRaises(RuntimeError, M_, 4.0, 0.1, 's')


@helpers.requires_uncertainties()
class TestMeasurement(QuantityTestCase):

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

    def test_format(self):
        v, u = self.Q_(4.0, 's ** 2'), self.Q_(.1, 's ** 2')
        m = self.ureg.Measurement(v, u)
        self.assertEqual(str(m), '(4.00 +/- 0.10) second ** 2')
        self.assertEqual(repr(m), '<Measurement(4.00, 0.10, second ** 2)>')
        #self.assertEqual('{:!s}'.format(m), '(4.00 +/- 0.10) second ** 2')
        #self.assertEqual('{:!r}'.format(m), '<Measurement(4.0-, 0.10, second ** 2)>')
        self.assertEqual('{0:P}'.format(m), '(4.00 ± 0.10) second²')
        self.assertEqual('{0:L}'.format(m), r'\left(4.00 \pm 0.10\right)\ \mathrm{second}^{2}')
        self.assertEqual('{0:H}'.format(m), '(4.00 &plusmn; 0.10) second<sup>2</sup>')
        self.assertEqual('{0:C}'.format(m), '(4.00+/-0.10) second**2')
        self.assertEqual('{0:Lx}'.format(m), r'\SI[separate-uncertainty=true]{4.00(10)}{\second\squared}')
        self.assertEqual('{0:.1f}'.format(m), '(4.0 +/- 0.1) second ** 2')
        self.assertEqual('{0:.1fP}'.format(m), '(4.0 ± 0.1) second²')
        self.assertEqual('{0:.1fL}'.format(m), r'\left(4.0 \pm 0.1\right)\ \mathrm{second}^{2}')
        self.assertEqual('{0:.1fH}'.format(m), '(4.0 &plusmn; 0.1) second<sup>2</sup>')
        self.assertEqual('{0:.1fC}'.format(m), '(4.0+/-0.1) second**2')
        self.assertEqual('{0:.1fLx}'.format(m), '\SI[separate-uncertainty=true]{4.0(1)}{\second\squared}')

    def test_format_paru(self):
        v, u = self.Q_(0.20, 's ** 2'), self.Q_(0.01, 's ** 2')
        m = self.ureg.Measurement(v, u)
        self.assertEqual('{0:uS}'.format(m), '0.200(10) second ** 2')
        self.assertEqual('{0:.3uS}'.format(m), '0.2000(100) second ** 2')
        self.assertEqual('{0:.3uSP}'.format(m), '0.2000(100) second²')
        self.assertEqual('{0:.3uSL}'.format(m), r'0.2000\left(100\right)\ \mathrm{second}^{2}')
        self.assertEqual('{0:.3uSH}'.format(m), '0.2000(100) second<sup>2</sup>')
        self.assertEqual('{0:.3uSC}'.format(m), '0.2000(100) second**2')

    def test_format_u(self):
        v, u = self.Q_(0.20, 's ** 2'), self.Q_(0.01, 's ** 2')
        m = self.ureg.Measurement(v, u)
        self.assertEqual('{0:.3u}'.format(m), '(0.2000 +/- 0.0100) second ** 2')
        self.assertEqual('{0:.3uP}'.format(m), '(0.2000 ± 0.0100) second²')
        self.assertEqual('{0:.3uL}'.format(m), r'\left(0.2000 \pm 0.0100\right)\ \mathrm{second}^{2}')
        self.assertEqual('{0:.3uH}'.format(m), '(0.2000 &plusmn; 0.0100) second<sup>2</sup>')
        self.assertEqual('{0:.3uC}'.format(m), '(0.2000+/-0.0100) second**2')
        self.assertEqual('{0:.3uLx}'.format(m), '\SI[separate-uncertainty=true]{0.2000(100)}{\second\squared}')
        self.assertEqual('{0:.1uLx}'.format(m), '\SI[separate-uncertainty=true]{0.20(1)}{\second\squared}')

    def test_format_percu(self):
        self.test_format_perce()
        v, u = self.Q_(0.20, 's ** 2'), self.Q_(0.01, 's ** 2')
        m = self.ureg.Measurement(v, u)
        self.assertEqual('{0:.1u%}'.format(m), '(20 +/- 1)% second ** 2')
        self.assertEqual('{0:.1u%P}'.format(m), '(20 ± 1)% second²')
        self.assertEqual('{0:.1u%L}'.format(m), r'\left(20 \pm 1\right) \%\ \mathrm{second}^{2}')
        self.assertEqual('{0:.1u%H}'.format(m), '(20 &plusmn; 1)% second<sup>2</sup>')
        self.assertEqual('{0:.1u%C}'.format(m), '(20+/-1)% second**2')

    def test_format_perce(self):
        v, u = self.Q_(0.20, 's ** 2'), self.Q_(0.01, 's ** 2')
        m = self.ureg.Measurement(v, u)
        self.assertEqual('{0:.1ue}'.format(m), '(2.0 +/- 0.1)e-01 second ** 2')
        self.assertEqual('{0:.1ueP}'.format(m), '(2.0 ± 0.1)×10⁻¹ second²')
        self.assertEqual('{0:.1ueL}'.format(m), r'\left(2.0 \pm 0.1\right) \times 10^{-1}\ \mathrm{second}^{2}')
        self.assertEqual('{0:.1ueH}'.format(m), '(2.0 &plusmn; 0.1)e-01 second<sup>2</sup>')
        self.assertEqual('{0:.1ueC}'.format(m), '(2.0+/-0.1)e-01 second**2')

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
