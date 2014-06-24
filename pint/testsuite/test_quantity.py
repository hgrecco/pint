# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import math
import operator as op

from pint import DimensionalityError, OffsetUnitCalculusError, UnitRegistry
from pint.unit import UnitsContainer
from pint.compat import string_types, PYTHON3, np, unittest
from pint.testsuite import QuantityTestCase, helpers
from pint.testsuite.parameterized import ParameterizedTestCase


class TestQuantity(QuantityTestCase):

    FORCE_NDARRAY = False

    def test_quantity_creation(self):
        for args in ((4.2, 'meter'),
                     (4.2,  UnitsContainer(meter=1)),
                     (4.2,  self.ureg.meter),
                     ('4.2*meter', ),
                     ('4.2/meter**(-1)', ),
                     (self.Q_(4.2, 'meter'),)):
            x = self.Q_(*args)
            self.assertEqual(x.magnitude, 4.2)
            self.assertEqual(x.units, UnitsContainer(meter=1))

        x = self.Q_(4.2, UnitsContainer(length=1))
        y = self.Q_(x)
        self.assertEqual(x.magnitude, y.magnitude)
        self.assertEqual(x.units, y.units)
        self.assertIsNot(x, y)

        x = self.Q_(4.2, None)
        self.assertEqual(x.magnitude, 4.2)
        self.assertEqual(x.units, UnitsContainer())

        with self.capture_log() as buffer:
            self.assertEqual(4.2 * self.ureg.meter, self.Q_(4.2, 2 * self.ureg.meter))
            self.assertEqual(len(buffer), 1)

    def test_quantity_bool(self):
        self.assertTrue(self.Q_(1, None))
        self.assertTrue(self.Q_(1, 'meter'))
        self.assertFalse(self.Q_(0, None))
        self.assertFalse(self.Q_(0, 'meter'))

    def test_quantity_comparison(self):
        x = self.Q_(4.2, 'meter')
        y = self.Q_(4.2, 'meter')
        z = self.Q_(5, 'meter')
        j = self.Q_(5, 'meter*meter')

        # identity for single object
        self.assertTrue(x == x)
        self.assertFalse(x != x)

        # identity for multiple objects with same value
        self.assertTrue(x == y)
        self.assertFalse(x != y)

        self.assertTrue(x <= y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x > y)

        self.assertFalse(x == z)
        self.assertTrue(x != z)
        self.assertTrue(x < z)

        self.assertTrue(z != j)

        self.assertNotEqual(z, j)
        self.assertEqual(self.Q_(0, 'meter'), self.Q_(0, 'centimeter'))
        self.assertNotEqual(self.Q_(0, 'meter'), self.Q_(0, 'second'))

        self.assertLess(self.Q_(10, 'meter'), self.Q_(5, 'kilometer'))

    def test_quantity_comparison_convert(self):
        self.assertEqual(self.Q_(1000, 'millimeter'), self.Q_(1, 'meter'))
        self.assertEqual(self.Q_(1000, 'millimeter/min'), self.Q_(1000/60, 'millimeter/s'))

    def test_quantity_repr(self):
        x = self.Q_(4.2, UnitsContainer(meter=1))
        self.assertEqual(str(x), '4.2 meter')
        self.assertEqual(repr(x), "<Quantity(4.2, 'meter')>")

    def test_quantity_format(self):
        x = self.Q_(4.12345678, UnitsContainer(meter=2, kilogram=1, second=-1))
        for spec, result in (('{0}', str(x)), ('{0!s}', str(x)), ('{0!r}', repr(x)),
                             ('{0.magnitude}',  str(x.magnitude)), ('{0.units}',  str(x.units)),
                             ('{0.magnitude!s}',  str(x.magnitude)), ('{0.units!s}',  str(x.units)),
                             ('{0.magnitude!r}',  repr(x.magnitude)), ('{0.units!r}',  repr(x.units)),
                             ('{0:.4f}', '{0:.4f} {1!s}'.format(x.magnitude, x.units)),
                             ('{0:L}', r'4.12345678 \frac{kilogram \cdot meter^{2}}{second}'),
                             ('{0:P}', '4.12345678 kilogram·meter²/second'),
                             ('{0:H}', '4.12345678 kilogram meter<sup>2</sup>/second'),
                             ('{0:C}', '4.12345678 kilogram*meter**2/second'),
                             ('{0:~}', '4.12345678 kg * m ** 2 / s'),
                             ('{0:L~}', r'4.12345678 \frac{kg \cdot m^{2}}{s}'),
                             ('{0:P~}', '4.12345678 kg·m²/s'),
                             ('{0:H~}', '4.12345678 kg m<sup>2</sup>/s'),
                             ('{0:C~}', '4.12345678 kg*m**2/s'),
                             ):
            self.assertEqual(spec.format(x), result)

    def test_default_formatting(self):
        ureg = UnitRegistry()
        x = ureg.Quantity(4.12345678, UnitsContainer(meter=2, kilogram=1, second=-1))
        for spec, result in (('L', r'4.12345678 \frac{kilogram \cdot meter^{2}}{second}'),
                             ('P', '4.12345678 kilogram·meter²/second'),
                             ('H', '4.12345678 kilogram meter<sup>2</sup>/second'),
                             ('C', '4.12345678 kilogram*meter**2/second'),
                             ('~', '4.12345678 kg * m ** 2 / s'),
                             ('L~', r'4.12345678 \frac{kg \cdot m^{2}}{s}'),
                             ('P~', '4.12345678 kg·m²/s'),
                             ('H~', '4.12345678 kg m<sup>2</sup>/s'),
                             ('C~', '4.12345678 kg*m**2/s'),
                             ):
            ureg.default_format = spec
            self.assertEqual('{0}'.format(x), result)

    def test_to_base_units(self):
        x = self.Q_('1*inch')
        self.assertQuantityAlmostEqual(x.to_base_units(), self.Q_(0.0254, 'meter'))
        x = self.Q_('1*inch*inch')
        self.assertQuantityAlmostEqual(x.to_base_units(), self.Q_(0.0254 ** 2.0, 'meter*meter'))
        x = self.Q_('1*inch/minute')
        self.assertQuantityAlmostEqual(x.to_base_units(), self.Q_(0.0254 / 60., 'meter/second'))

    def test_convert(self):
        x = self.Q_('2*inch')
        self.assertQuantityAlmostEqual(x.to('meter'), self.Q_(2. * 0.0254, 'meter'))
        x = self.Q_('2*meter')
        self.assertQuantityAlmostEqual(x.to('inch'), self.Q_(2. / 0.0254, 'inch'))
        x = self.Q_('2*sidereal_second')
        self.assertQuantityAlmostEqual(x.to('second'), self.Q_(1.994539133 , 'second'))
        x = self.Q_('2.54*centimeter/second')
        self.assertQuantityAlmostEqual(x.to('inch/second'), self.Q_(1, 'inch/second'))
        x = self.Q_('2.54*centimeter')
        self.assertQuantityAlmostEqual(x.to('inch').magnitude, 1)
        self.assertQuantityAlmostEqual(self.Q_(2, 'second').to('millisecond').magnitude, 2000)

    @helpers.requires_numpy()
    def test_convert(self):

        # Conversions with single units take a different codepath than
        # Conversions with more than one unit.
        src_dst1 = UnitsContainer(meter=1), UnitsContainer(inch=1)
        src_dst2 = UnitsContainer(meter=1, second=-1), UnitsContainer(inch=1, minute=-1)
        for src, dst in (src_dst1, src_dst2):
            a = np.ones((3, 1))
            ac = np.ones((3, 1))

            q = self.Q_(a, src)
            qac = self.Q_(ac, src).to(dst)
            r = q.to(dst)
            self.assertQuantityAlmostEqual(qac, r)
            self.assertIsNot(r, q)
            self.assertIsNot(r._magnitude, a)

    def test_context_attr(self):
        self.assertEqual(self.ureg.meter, self.Q_(1, 'meter'))

    def test_both_symbol(self):
        self.assertEqual(self.Q_(2, 'ms'), self.Q_(2, 'millisecond'))
        self.assertEqual(self.Q_(2, 'cm'), self.Q_(2, 'centimeter'))

    def test_dimensionless_units(self):
        self.assertAlmostEqual(self.Q_(360, 'degree').to('radian').magnitude, 2 * math.pi)
        self.assertAlmostEqual(self.Q_(2 * math.pi, 'radian'), self.Q_(360, 'degree'))
        self.assertEqual(self.Q_(1, 'radian').dimensionality, UnitsContainer())
        self.assertTrue(self.Q_(1, 'radian').dimensionless)
        self.assertFalse(self.Q_(1, 'radian').unitless)

        self.assertEqual(self.Q_(1, 'meter')/self.Q_(1, 'meter'), 1)
        self.assertEqual((self.Q_(1, 'meter')/self.Q_(1, 'mm')).to(''), 1000)

    def test_offset(self):
        self.assertQuantityAlmostEqual(self.Q_(0, 'kelvin').to('kelvin'), self.Q_(0, 'kelvin'))
        self.assertQuantityAlmostEqual(self.Q_(0, 'degC').to('kelvin'), self.Q_(273.15, 'kelvin'))
        self.assertQuantityAlmostEqual(self.Q_(0, 'degF').to('kelvin'), self.Q_(255.372222, 'kelvin'), rtol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(100, 'kelvin').to('kelvin'), self.Q_(100, 'kelvin'))
        self.assertQuantityAlmostEqual(self.Q_(100, 'degC').to('kelvin'), self.Q_(373.15, 'kelvin'))
        self.assertQuantityAlmostEqual(self.Q_(100, 'degF').to('kelvin'), self.Q_(310.92777777, 'kelvin'), rtol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(0, 'kelvin').to('degC'), self.Q_(-273.15, 'degC'))
        self.assertQuantityAlmostEqual(self.Q_(100, 'kelvin').to('degC'), self.Q_(-173.15, 'degC'))
        self.assertQuantityAlmostEqual(self.Q_(0, 'kelvin').to('degF'), self.Q_(-459.67, 'degF'), rtol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(100, 'kelvin').to('degF'), self.Q_(-279.67, 'degF'), rtol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(32, 'degF').to('degC'), self.Q_(0, 'degC'), atol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(100, 'degC').to('degF'), self.Q_(212, 'degF'), atol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(54, 'degF').to('degC'), self.Q_(12.2222, 'degC'), atol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(12, 'degC').to('degF'), self.Q_(53.6, 'degF'), atol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(12, 'kelvin').to('degC'), self.Q_(-261.15, 'degC'), atol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(12, 'degC').to('kelvin'), self.Q_(285.15, 'kelvin'), atol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(12, 'kelvin').to('degR'), self.Q_(21.6, 'degR'), atol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(12, 'degR').to('kelvin'), self.Q_(6.66666667, 'kelvin'), atol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(12, 'degC').to('degR'), self.Q_(513.27, 'degR'), atol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(12, 'degR').to('degC'), self.Q_(-266.483333, 'degC'), atol=0.01)


    def test_offset_delta(self):
        self.assertQuantityAlmostEqual(self.Q_(0, 'delta_degC').to('kelvin'), self.Q_(0, 'kelvin'))
        self.assertQuantityAlmostEqual(self.Q_(0, 'delta_degF').to('kelvin'), self.Q_(0, 'kelvin'), rtol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(100, 'kelvin').to('delta_degC'), self.Q_(100, 'delta_degC'))
        self.assertQuantityAlmostEqual(self.Q_(100, 'kelvin').to('delta_degF'), self.Q_(180, 'delta_degF'), rtol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(100, 'delta_degF').to('kelvin'), self.Q_(55.55555556, 'kelvin'), rtol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(100, 'delta_degC').to('delta_degF'), self.Q_(180, 'delta_degF'), rtol=0.01)
        self.assertQuantityAlmostEqual(self.Q_(100, 'delta_degF').to('delta_degC'), self.Q_(55.55555556, 'delta_degC'), rtol=0.01)

        self.assertQuantityAlmostEqual(self.Q_(12.3, 'delta_degC').to('delta_degF'), self.Q_(22.14, 'delta_degF'), rtol=0.01)


    def test_pickle(self):
        import pickle

        def pickle_test(q):
            self.assertEqual(q, pickle.loads(pickle.dumps(q)))

        pickle_test(self.Q_(32, ''))
        pickle_test(self.Q_(2.4, ''))
        pickle_test(self.Q_(32, 'm/s'))
        pickle_test(self.Q_(2.4, 'm/s'))


class TestQuantityBasicMath(QuantityTestCase):

    FORCE_NDARRAY = False

    def _test_inplace(self, operator, value1, value2, expected_result, unit=None):
        if isinstance(value1, string_types):
            value1 = self.Q_(value1)
        if isinstance(value2, string_types):
            value2 = self.Q_(value2)
        if isinstance(expected_result, string_types):
            expected_result = self.Q_(expected_result)

        if not unit is None:
            value1 = value1 * unit
            value2 = value2 * unit
            expected_result = expected_result * unit

        value1 = copy.copy(value1)
        value2 = copy.copy(value2)
        id1 = id(value1)
        id2 = id(value2)
        value1 = operator(value1, value2)
        value2_cpy = copy.copy(value2)
        self.assertQuantityAlmostEqual(value1, expected_result)
        self.assertEqual(id1, id(value1))
        self.assertQuantityAlmostEqual(value2, value2_cpy)
        self.assertEqual(id2, id(value2))

    def _test_not_inplace(self, operator, value1, value2, expected_result, unit=None):
        if isinstance(value1, string_types):
            value1 = self.Q_(value1)
        if isinstance(value2, string_types):
            value2 = self.Q_(value2)
        if isinstance(expected_result, string_types):
            expected_result = self.Q_(expected_result)

        if not unit is None:
            value1 = value1 * unit
            value2 = value2 * unit
            expected_result = expected_result * unit

        id1 = id(value1)
        id2 = id(value2)

        value1_cpy = copy.copy(value1)
        value2_cpy = copy.copy(value2)

        result = operator(value1, value2)

        self.assertQuantityAlmostEqual(expected_result, result)
        self.assertQuantityAlmostEqual(value1, value1_cpy)
        self.assertQuantityAlmostEqual(value2, value2_cpy)
        self.assertNotEqual(id(result), id1)
        self.assertNotEqual(id(result), id2)

    def _test_quantity_add_sub(self, unit, func):
        x = self.Q_(unit, 'centimeter')
        y = self.Q_(unit, 'inch')
        z = self.Q_(unit, 'second')
        a = self.Q_(unit, None)

        func(op.add, x, x, self.Q_(unit + unit, 'centimeter'))
        func(op.add, x, y, self.Q_(unit + 2.54 * unit, 'centimeter'))
        func(op.add, y, x, self.Q_(unit + unit / (2.54 * unit), 'inch'))
        func(op.add, a, unit, self.Q_(unit + unit, None))
        self.assertRaises(DimensionalityError, op.add, 10, x)
        self.assertRaises(DimensionalityError, op.add, x, 10)
        self.assertRaises(DimensionalityError, op.add, x, z)

        func(op.sub, x, x, self.Q_(unit - unit, 'centimeter'))
        func(op.sub, x, y, self.Q_(unit - 2.54 * unit, 'centimeter'))
        func(op.sub, y, x, self.Q_(unit - unit / (2.54 * unit), 'inch'))
        func(op.sub, a, unit, self.Q_(unit - unit, None))
        self.assertRaises(DimensionalityError, op.sub, 10, x)
        self.assertRaises(DimensionalityError, op.sub, x, 10)
        self.assertRaises(DimensionalityError, op.sub, x, z)

    def _test_quantity_iadd_isub(self, unit, func):
        x = self.Q_(unit, 'centimeter')
        y = self.Q_(unit, 'inch')
        z = self.Q_(unit, 'second')
        a = self.Q_(unit, None)

        func(op.iadd, x, x, self.Q_(unit + unit, 'centimeter'))
        func(op.iadd, x, y, self.Q_(unit + 2.54 * unit, 'centimeter'))
        func(op.iadd, y, x, self.Q_(unit + unit / 2.54, 'inch'))
        func(op.iadd, a, unit, self.Q_(unit + unit, None))
        self.assertRaises(DimensionalityError, op.iadd, 10, x)
        self.assertRaises(DimensionalityError, op.iadd, x, 10)
        self.assertRaises(DimensionalityError, op.iadd, x, z)

        func(op.isub, x, x, self.Q_(unit - unit, 'centimeter'))
        func(op.isub, x, y, self.Q_(unit - 2.54, 'centimeter'))
        func(op.isub, y, x, self.Q_(unit - unit / 2.54, 'inch'))
        func(op.isub, a, unit, self.Q_(unit - unit, None))
        self.assertRaises(DimensionalityError, op.sub, 10, x)
        self.assertRaises(DimensionalityError, op.sub, x, 10)
        self.assertRaises(DimensionalityError, op.sub, x, z)

    def _test_quantity_mul_div(self, unit, func):
        func(op.mul, unit * 10.0, '4.2*meter', '42*meter', unit)
        func(op.mul, '4.2*meter', unit * 10.0, '42*meter', unit)
        func(op.mul, '4.2*meter', '10*inch', '42*meter*inch', unit)
        func(op.truediv, unit * 42, '4.2*meter', '10/meter', unit)
        func(op.truediv, '4.2*meter', unit * 10.0, '0.42*meter', unit)
        func(op.truediv, '4.2*meter', '10*inch', '0.42*meter/inch', unit)

    def _test_quantity_imul_idiv(self, unit, func):
        #func(op.imul, 10.0, '4.2*meter', '42*meter')
        func(op.imul, '4.2*meter', 10.0, '42*meter', unit)
        func(op.imul, '4.2*meter', '10*inch', '42*meter*inch', unit)
        #func(op.truediv, 42, '4.2*meter', '10/meter')
        func(op.itruediv, '4.2*meter', unit * 10.0, '0.42*meter', unit)
        func(op.itruediv, '4.2*meter', '10*inch', '0.42*meter/inch', unit)

    def _test_quantity_floordiv(self, unit, func):
        func(op.floordiv, unit * 10.0, '4.2*meter', '2/meter', unit)
        func(op.floordiv, '24*meter', unit * 10.0, '2*meter', unit)
        func(op.floordiv, '10*meter', '4.2*inch', '2*meter/inch', unit)

    def _test_quantity_ifloordiv(self, unit, func):
        func(op.ifloordiv, 10.0, '4.2*meter', '2/meter', unit)
        func(op.ifloordiv, '24*meter', 10.0, '2*meter', unit)
        func(op.ifloordiv, '10*meter', '4.2*inch', '2*meter/inch', unit)

    def _test_numeric(self, unit, ifunc):
        self._test_quantity_add_sub(unit, self._test_not_inplace)
        self._test_quantity_iadd_isub(unit, ifunc)
        self._test_quantity_mul_div(unit, self._test_not_inplace)
        self._test_quantity_imul_idiv(unit, ifunc)
        self._test_quantity_floordiv(unit, self._test_not_inplace)
        #self._test_quantity_ifloordiv(unit, ifunc)

    def test_float(self):
        self._test_numeric(1., self._test_not_inplace)

    def test_fraction(self):
        import fractions
        self._test_numeric(fractions.Fraction(1, 1), self._test_not_inplace)

    @helpers.requires_numpy()
    def test_nparray(self):
        self._test_numeric(np.ones((1, 3)), self._test_inplace)

    def test_quantity_abs_round(self):

        x = self.Q_(-4.2, 'meter')
        y = self.Q_(4.2, 'meter')
        # In Python 3+ round of x is delegated to x.__round__, instead of round(x.__float__)
        # and therefore it can be properly implemented by Pint
        for fun in (abs, op.pos, op.neg) + (round, ) if PYTHON3 else ():
            zx = self.Q_(fun(x.magnitude), 'meter')
            zy = self.Q_(fun(y.magnitude), 'meter')
            rx = fun(x)
            ry = fun(y)
            self.assertEqual(rx, zx, 'while testing {0}'.format(fun))
            self.assertEqual(ry, zy, 'while testing {0}'.format(fun))
            self.assertIsNot(rx, zx, 'while testing {0}'.format(fun))
            self.assertIsNot(ry, zy, 'while testing {0}'.format(fun))

    def test_quantity_float_complex(self):
        x = self.Q_(-4.2, None)
        y = self.Q_(4.2, None)
        z = self.Q_(1, 'meter')
        for fun in (float, complex):
            self.assertEqual(fun(x), fun(x.magnitude))
            self.assertEqual(fun(y), fun(y.magnitude))
            self.assertRaises(DimensionalityError, fun, z)


class TestDimensions(QuantityTestCase):

    FORCE_NDARRAY = False

    def test_get_dimensionality(self):
        get = self.ureg.get_dimensionality
        self.assertEqual(get('[time]'), UnitsContainer({'[time]': 1}))
        self.assertEqual(get(UnitsContainer({'[time]': 1})), UnitsContainer({'[time]': 1}))
        self.assertEqual(get('seconds'), UnitsContainer({'[time]': 1}))
        self.assertEqual(get(UnitsContainer({'seconds': 1})), UnitsContainer({'[time]': 1}))
        self.assertEqual(get('[speed]'), UnitsContainer({'[length]': 1, '[time]': -1}))
        self.assertEqual(get('[acceleration]'), UnitsContainer({'[length]': 1, '[time]': -2}))

    def test_dimensionality(self):
        x = self.Q_(42, 'centimeter')
        x.to_base_units()
        x = self.Q_(42, 'meter*second')
        self.assertEqual(x.dimensionality, UnitsContainer({'[length]': 1., '[time]': 1.}))
        x = self.Q_(42, 'meter*second*second')
        self.assertEqual(x.dimensionality, UnitsContainer({'[length]': 1., '[time]': 2.}))
        x = self.Q_(42, 'inch*second*second')
        self.assertEqual(x.dimensionality, UnitsContainer({'[length]': 1., '[time]': 2.}))
        self.assertTrue(self.Q_(42, None).dimensionless)
        self.assertFalse(self.Q_(42, 'meter').dimensionless)
        self.assertTrue((self.Q_(42, 'meter') / self.Q_(1, 'meter')).dimensionless)
        self.assertFalse((self.Q_(42, 'meter') / self.Q_(1, 'second')).dimensionless)
        self.assertTrue((self.Q_(42, 'meter') / self.Q_(1, 'inch')).dimensionless)


class TestQuantityWithDefaultRegistry(TestDimensions):

    @classmethod
    def setUpClass(cls):
        from pint import _DEFAULT_REGISTRY
        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity


class TestDimensionsWithDefaultRegistry(TestDimensions):

    @classmethod
    def setUpClass(cls):
        from pint import _DEFAULT_REGISTRY
        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity


class TestOffsetUnitMath(QuantityTestCase, ParameterizedTestCase):

    def setup(self):
        self.ureg.autoconvert_offset_to_baseunit = False
        self.ureg.default_as_delta = True

    additions = [
        # --- input tuple -------------------- | -- expected result --
        (((100, 'kelvin'), (10, 'kelvin')),      (110, 'kelvin')),
        (((100, 'kelvin'), (10, 'degC')),        'error'),
        (((100, 'kelvin'), (10, 'degF')),        'error'),
        (((100, 'kelvin'), (10, 'degR')),        (105.56, 'kelvin')),
        (((100, 'kelvin'), (10, 'delta_degC')),  (110, 'kelvin')),
        (((100, 'kelvin'), (10, 'delta_degF')),  (105.56, 'kelvin')),

        (((100, 'degC'), (10, 'kelvin')),      'error'),
        (((100, 'degC'), (10, 'degC')),        'error'),
        (((100, 'degC'), (10, 'degF')),        'error'),
        (((100, 'degC'), (10, 'degR')),        'error'),
        (((100, 'degC'), (10, 'delta_degC')),  (110, 'degC')),
        (((100, 'degC'), (10, 'delta_degF')),  (105.56, 'degC')),

        (((100, 'degF'), (10, 'kelvin')),      'error'),
        (((100, 'degF'), (10, 'degC')),        'error'),
        (((100, 'degF'), (10, 'degF')),        'error'),
        (((100, 'degF'), (10, 'degR')),        'error'),
        (((100, 'degF'), (10, 'delta_degC')),  (118, 'degF')),
        (((100, 'degF'), (10, 'delta_degF')),  (110, 'degF')),

        (((100, 'degR'), (10, 'kelvin')),      (118, 'degR')),
        (((100, 'degR'), (10, 'degC')),        'error'),
        (((100, 'degR'), (10, 'degF')),        'error'),
        (((100, 'degR'), (10, 'degR')),        (110, 'degR')),
        (((100, 'degR'), (10, 'delta_degC')),  (118, 'degR')),
        (((100, 'degR'), (10, 'delta_degF')),  (110, 'degR')),

        (((100, 'delta_degC'), (10, 'kelvin')),     (110, 'kelvin')),
        (((100, 'delta_degC'), (10, 'degC')),       (110, 'degC')),
        (((100, 'delta_degC'), (10, 'degF')),       (190, 'degF')),
        (((100, 'delta_degC'), (10, 'degR')),       (190, 'degR')),
        (((100, 'delta_degC'), (10, 'delta_degC')), (110, 'delta_degC')),
        (((100, 'delta_degC'), (10, 'delta_degF')), (105.56, 'delta_degC')),

        (((100, 'delta_degF'), (10, 'kelvin')),     (65.56, 'kelvin')),
        (((100, 'delta_degF'), (10, 'degC')),       (65.56, 'degC')),
        (((100, 'delta_degF'), (10, 'degF')),       (110, 'degF')),
        (((100, 'delta_degF'), (10, 'degR')),       (110, 'degR')),
        (((100, 'delta_degF'), (10, 'delta_degC')), (118, 'delta_degF')),
        (((100, 'delta_degF'), (10, 'delta_degF')), (110, 'delta_degF')),
        ]

    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        additions)
    def test_addition(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        # update input tuple with new values to have correct values on failure
        input_tuple = q1, q2
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.add, q1, q2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.add(q1, q2).units, expected.units)
            self.assertQuantityAlmostEqual(op.add(q1, q2), expected,
                                           atol=0.01)

    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        additions)
    def test_inplace_addition(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        (q1v, q1u), (q2v, q2u) = input_tuple
        # update input tuple with new values to have correct values on failure
        input_tuple = ((np.array([q1v]*2, dtype=np.float), q1u),
                       (np.array([q2v]*2, dtype=np.float), q2u))
        Q_ = self.Q_
        qin1, qin2 = input_tuple
        q1, q2 = Q_(*qin1), Q_(*qin2)
        q1_cp = copy.copy(q1)
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.iadd, q1_cp, q2)
        else:
            expected = np.array([expected[0]]*2, dtype=np.float), expected[1]
            self.assertEqual(op.iadd(q1_cp, q2).units, Q_(*expected).units)
            q1_cp = copy.copy(q1)
            self.assertQuantityAlmostEqual(op.iadd(q1_cp, q2), Q_(*expected),
                                           atol=0.01)

    subtractions = [
        (((100, 'kelvin'), (10, 'kelvin')),      (90, 'kelvin')),
        (((100, 'kelvin'), (10, 'degC')),        (-183.15, 'kelvin')),
        (((100, 'kelvin'), (10, 'degF')),        (-160.93, 'kelvin')),
        (((100, 'kelvin'), (10, 'degR')),        (94.44, 'kelvin')),
        (((100, 'kelvin'), (10, 'delta_degC')),  (90, 'kelvin')),
        (((100, 'kelvin'), (10, 'delta_degF')),  (94.44, 'kelvin')),

        (((100, 'degC'), (10, 'kelvin')),      (363.15, 'delta_degC')),
        (((100, 'degC'), (10, 'degC')),        (90, 'delta_degC')),
        (((100, 'degC'), (10, 'degF')),        (112.22, 'delta_degC')),
        (((100, 'degC'), (10, 'degR')),        (367.59, 'delta_degC')),
        (((100, 'degC'), (10, 'delta_degC')),  (90, 'degC')),
        (((100, 'degC'), (10, 'delta_degF')),  (94.44, 'degC')),

        (((100, 'degF'), (10, 'kelvin')),      (541.67, 'delta_degF')),
        (((100, 'degF'), (10, 'degC')),        (50, 'delta_degF')),
        (((100, 'degF'), (10, 'degF')),        (90, 'delta_degF')),
        (((100, 'degF'), (10, 'degR')),        (549.67, 'delta_degF')),
        (((100, 'degF'), (10, 'delta_degC')),  (82, 'degF')),
        (((100, 'degF'), (10, 'delta_degF')),  (90, 'degF')),

        (((100, 'degR'), (10, 'kelvin')),      (82, 'degR')),
        (((100, 'degR'), (10, 'degC')),        (-409.67, 'degR')),
        (((100, 'degR'), (10, 'degF')),        (-369.67, 'degR')),
        (((100, 'degR'), (10, 'degR')),        (90, 'degR')),
        (((100, 'degR'), (10, 'delta_degC')),  (82, 'degR')),
        (((100, 'degR'), (10, 'delta_degF')),  (90, 'degR')),

        (((100, 'delta_degC'), (10, 'kelvin')),     (90, 'kelvin')),
        (((100, 'delta_degC'), (10, 'degC')),       (90, 'degC')),
        (((100, 'delta_degC'), (10, 'degF')),       (170, 'degF')),
        (((100, 'delta_degC'), (10, 'degR')),       (170, 'degR')),
        (((100, 'delta_degC'), (10, 'delta_degC')), (90, 'delta_degC')),
        (((100, 'delta_degC'), (10, 'delta_degF')), (94.44, 'delta_degC')),

        (((100, 'delta_degF'), (10, 'kelvin')),     (45.56, 'kelvin')),
        (((100, 'delta_degF'), (10, 'degC')),       (45.56, 'degC')),
        (((100, 'delta_degF'), (10, 'degF')),       (90, 'degF')),
        (((100, 'delta_degF'), (10, 'degR')),       (90, 'degR')),
        (((100, 'delta_degF'), (10, 'delta_degC')), (82, 'delta_degF')),
        (((100, 'delta_degF'), (10, 'delta_degF')), (90, 'delta_degF')),
        ]

    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        subtractions)
    def test_subtraction(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.sub, q1, q2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.sub(q1, q2).units, expected.units)
            self.assertQuantityAlmostEqual(op.sub(q1, q2), expected,
                                           atol=0.01)

#    @unittest.expectedFailure
    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        subtractions)
    def test_inplace_subtraction(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        (q1v, q1u), (q2v, q2u) = input_tuple
        # update input tuple with new values to have correct values on failure
        input_tuple = ((np.array([q1v]*2, dtype=np.float), q1u),
                       (np.array([q2v]*2, dtype=np.float), q2u))
        Q_ = self.Q_
        qin1, qin2 = input_tuple
        q1, q2 = Q_(*qin1), Q_(*qin2)
        q1_cp = copy.copy(q1)
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.isub, q1_cp, q2)
        else:
            expected = np.array([expected[0]]*2, dtype=np.float), expected[1]
            self.assertEqual(op.isub(q1_cp, q2).units, Q_(*expected).units)
            q1_cp = copy.copy(q1)
            self.assertQuantityAlmostEqual(op.isub(q1_cp, q2), Q_(*expected),
                                           atol=0.01)

    multiplications = [
        (((100, 'kelvin'), (10, 'kelvin')),     (1000, 'kelvin**2')),
        (((100, 'kelvin'), (10, 'degC')),       'error'),
        (((100, 'kelvin'), (10, 'degF')),       'error'),
        (((100, 'kelvin'), (10, 'degR')),       (1000, 'kelvin*degR')),
        (((100, 'kelvin'), (10, 'delta_degC')), (1000, 'kelvin*delta_degC')),
        (((100, 'kelvin'), (10, 'delta_degF')), (1000, 'kelvin*delta_degF')),

        (((100, 'degC'), (10, 'kelvin')),      'error'),
        (((100, 'degC'), (10, 'degC')),        'error'),
        (((100, 'degC'), (10, 'degF')),        'error'),
        (((100, 'degC'), (10, 'degR')),        'error'),
        (((100, 'degC'), (10, 'delta_degC')),  'error'),
        (((100, 'degC'), (10, 'delta_degF')),  'error'),

        (((100, 'degF'), (10, 'kelvin')),      'error'),
        (((100, 'degF'), (10, 'degC')),        'error'),
        (((100, 'degF'), (10, 'degF')),        'error'),
        (((100, 'degF'), (10, 'degR')),        'error'),
        (((100, 'degF'), (10, 'delta_degC')),  'error'),
        (((100, 'degF'), (10, 'delta_degF')),  'error'),

        (((100, 'degR'), (10, 'kelvin')),      (1000, 'degR*kelvin')),
        (((100, 'degR'), (10, 'degC')),        'error'),
        (((100, 'degR'), (10, 'degF')),        'error'),
        (((100, 'degR'), (10, 'degR')),        (1000, 'degR**2')),
        (((100, 'degR'), (10, 'delta_degC')),  (1000, 'degR*delta_degC')),
        (((100, 'degR'), (10, 'delta_degF')),  (1000, 'degR*delta_degF')),

        (((100, 'delta_degC'), (10, 'kelvin')),     (1000, 'delta_degC*kelvin')),
        (((100, 'delta_degC'), (10, 'degC')),       'error'),
        (((100, 'delta_degC'), (10, 'degF')),       'error'),
        (((100, 'delta_degC'), (10, 'degR')),       (1000, 'delta_degC*degR')),
        (((100, 'delta_degC'), (10, 'delta_degC')), (1000, 'delta_degC**2')),
        (((100, 'delta_degC'), (10, 'delta_degF')), (1000, 'delta_degC*delta_degF')),

        (((100, 'delta_degF'), (10, 'kelvin')),     (1000, 'delta_degF*kelvin')),
        (((100, 'delta_degF'), (10, 'degC')),       'error'),
        (((100, 'delta_degF'), (10, 'degF')),       'error'),
        (((100, 'delta_degF'), (10, 'degR')),       (1000, 'delta_degF*degR')),
        (((100, 'delta_degF'), (10, 'delta_degC')), (1000, 'delta_degF*delta_degC')),
        (((100, 'delta_degF'), (10, 'delta_degF')), (1000, 'delta_degF**2')),
        ]

    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        multiplications)
    def test_multiplication(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.mul, q1, q2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.mul(q1, q2).units, expected.units)
            self.assertQuantityAlmostEqual(op.mul(q1, q2), expected,
                                           atol=0.01)

    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        multiplications)
    def test_inplace_multiplication(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        (q1v, q1u), (q2v, q2u) = input_tuple
        # update input tuple with new values to have correct values on failure
        input_tuple = ((np.array([q1v]*2, dtype=np.float), q1u),
                       (np.array([q2v]*2, dtype=np.float), q2u))
        Q_ = self.Q_
        qin1, qin2 = input_tuple
        q1, q2 = Q_(*qin1), Q_(*qin2)
        q1_cp = copy.copy(q1)
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.imul, q1_cp, q2)
        else:
            expected = np.array([expected[0]]*2, dtype=np.float), expected[1]
            self.assertEqual(op.imul(q1_cp, q2).units, Q_(*expected).units)
            q1_cp = copy.copy(q1)
            self.assertQuantityAlmostEqual(op.imul(q1_cp, q2), Q_(*expected),
                                           atol=0.01)

    divisions = [
        (((100, 'kelvin'), (10, 'kelvin')),     (10, '')),
        (((100, 'kelvin'), (10, 'degC')),       'error'),
        (((100, 'kelvin'), (10, 'degF')),       'error'),
        (((100, 'kelvin'), (10, 'degR')),       (10, 'kelvin/degR')),
        (((100, 'kelvin'), (10, 'delta_degC')), (10, 'kelvin/delta_degC')),
        (((100, 'kelvin'), (10, 'delta_degF')), (10, 'kelvin/delta_degF')),

        (((100, 'degC'), (10, 'kelvin')),      'error'),
        (((100, 'degC'), (10, 'degC')),        'error'),
        (((100, 'degC'), (10, 'degF')),        'error'),
        (((100, 'degC'), (10, 'degR')),        'error'),
        (((100, 'degC'), (10, 'delta_degC')),  'error'),
        (((100, 'degC'), (10, 'delta_degF')),  'error'),

        (((100, 'degF'), (10, 'kelvin')),      'error'),
        (((100, 'degF'), (10, 'degC')),        'error'),
        (((100, 'degF'), (10, 'degF')),        'error'),
        (((100, 'degF'), (10, 'degR')),        'error'),
        (((100, 'degF'), (10, 'delta_degC')),  'error'),
        (((100, 'degF'), (10, 'delta_degF')),  'error'),

        (((100, 'degR'), (10, 'kelvin')),      (10, 'degR/kelvin')),
        (((100, 'degR'), (10, 'degC')),        'error'),
        (((100, 'degR'), (10, 'degF')),        'error'),
        (((100, 'degR'), (10, 'degR')),        (10, '')),
        (((100, 'degR'), (10, 'delta_degC')),  (10, 'degR/delta_degC')),
        (((100, 'degR'), (10, 'delta_degF')),  (10, 'degR/delta_degF')),

        (((100, 'delta_degC'), (10, 'kelvin')),     (10, 'delta_degC/kelvin')),
        (((100, 'delta_degC'), (10, 'degC')),       'error'),
        (((100, 'delta_degC'), (10, 'degF')),       'error'),
        (((100, 'delta_degC'), (10, 'degR')),       (10, 'delta_degC/degR')),
        (((100, 'delta_degC'), (10, 'delta_degC')), (10, '')),
        (((100, 'delta_degC'), (10, 'delta_degF')), (10, 'delta_degC/delta_degF')),

        (((100, 'delta_degF'), (10, 'kelvin')),     (10, 'delta_degF/kelvin')),
        (((100, 'delta_degF'), (10, 'degC')),       'error'),
        (((100, 'delta_degF'), (10, 'degF')),       'error'),
        (((100, 'delta_degF'), (10, 'degR')),       (10, 'delta_degF/degR')),
        (((100, 'delta_degF'), (10, 'delta_degC')), (10, 'delta_degF/delta_degC')),
        (((100, 'delta_degF'), (10, 'delta_degF')), (10, '')),
        ]

    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        divisions)
    def test_truedivision(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.truediv, q1, q2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.truediv(q1, q2).units, expected.units)
            self.assertQuantityAlmostEqual(op.truediv(q1, q2), expected,
                                           atol=0.01)

    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        divisions)
    def test_inplace_truedivision(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = False
        (q1v, q1u), (q2v, q2u) = input_tuple
        # update input tuple with new values to have correct values on failure
        input_tuple = ((np.array([q1v]*2, dtype=np.float), q1u),
                       (np.array([q2v]*2, dtype=np.float), q2u))
        Q_ = self.Q_
        qin1, qin2 = input_tuple
        q1, q2 = Q_(*qin1), Q_(*qin2)
        q1_cp = copy.copy(q1)
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.itruediv, q1_cp, q2)
        else:
            expected = np.array([expected[0]]*2, dtype=np.float), expected[1]
            self.assertEqual(op.itruediv(q1_cp, q2).units, Q_(*expected).units)
            q1_cp = copy.copy(q1)
            self.assertQuantityAlmostEqual(op.itruediv(q1_cp, q2),
                                           Q_(*expected), atol=0.01)

    multiplications_with_autoconvert_to_baseunit = [
        (((100, 'kelvin'), (10, 'degC')),     (28315., 'kelvin**2')),
        (((100, 'kelvin'), (10, 'degF')),     (26092.78, 'kelvin**2')),

        (((100, 'degC'), (10, 'kelvin')),     (3731.5, 'kelvin**2')),
        (((100, 'degC'), (10, 'degC')),       (105657.42, 'kelvin**2')),
        (((100, 'degC'), (10, 'degF')),       (97365.20, 'kelvin**2')),
        (((100, 'degC'), (10, 'degR')),       (3731.5, 'kelvin*degR')),
        (((100, 'degC'), (10, 'delta_degC')), (3731.5, 'kelvin*delta_degC')),
        (((100, 'degC'), (10, 'delta_degF')), (3731.5, 'kelvin*delta_degF')),

        (((100, 'degF'), (10, 'kelvin')),     (3109.28, 'kelvin**2')),
        (((100, 'degF'), (10, 'degC')),       (88039.20, 'kelvin**2')),
        (((100, 'degF'), (10, 'degF')),       (81129.69, 'kelvin**2')),
        (((100, 'degF'), (10, 'degR')),       (3109.28, 'kelvin*degR')),
        (((100, 'degF'), (10, 'delta_degC')), (3109.28, 'kelvin*delta_degC')),
        (((100, 'degF'), (10, 'delta_degF')), (3109.28, 'kelvin*delta_degF')),

        (((100, 'degR'), (10, 'degC')),       (28315., 'degR*kelvin')),
        (((100, 'degR'), (10, 'degF')),       (26092.78, 'degR*kelvin')),

        (((100, 'delta_degC'), (10, 'degC')), (28315., 'delta_degC*kelvin')),
        (((100, 'delta_degC'), (10, 'degF')), (26092.78, 'delta_degC*kelvin')),

        (((100, 'delta_degF'), (10, 'degC')), (28315., 'delta_degF*kelvin')),
        (((100, 'delta_degF'), (10, 'degF')), (26092.78, 'delta_degF*kelvin')),
        ]

    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"),
        multiplications_with_autoconvert_to_baseunit)
    def test_multiplication_with_autoconvert(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = True
        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.mul, q1, q2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.mul(q1, q2).units, expected.units)
            self.assertQuantityAlmostEqual(op.mul(q1, q2), expected,
                                           atol=0.01)

    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"),
        multiplications_with_autoconvert_to_baseunit)
    def test_inplace_multiplication_with_autoconvert(self, input_tuple, expected):
        self.ureg.autoconvert_offset_to_baseunit = True
        (q1v, q1u), (q2v, q2u) = input_tuple
        # update input tuple with new values to have correct values on failure
        input_tuple = ((np.array([q1v]*2, dtype=np.float), q1u),
                       (np.array([q2v]*2, dtype=np.float), q2u))
        Q_ = self.Q_
        qin1, qin2 = input_tuple
        q1, q2 = Q_(*qin1), Q_(*qin2)
        q1_cp = copy.copy(q1)
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.imul, q1_cp, q2)
        else:
            expected = np.array([expected[0]]*2, dtype=np.float), expected[1]
            self.assertEqual(op.imul(q1_cp, q2).units, Q_(*expected).units)
            q1_cp = copy.copy(q1)
            self.assertQuantityAlmostEqual(op.imul(q1_cp, q2), Q_(*expected),
                                           atol=0.01)

    multiplications_with_scalar = [
        (((10, 'kelvin'),    2),    (20., 'kelvin')),
        (((10, 'kelvin**2'), 2),    (20., 'kelvin**2')),
        (((10, 'degC'),      2),    (20., 'degC')),
        (((10, '1/degC'),    2),    'error'),
        (((10, 'degC**0.5'),   2),  'error'),
        (((10, 'degC**2'),   2),    'error'),
        (((10, 'degC**-2'),  2),    'error'),
        ]

    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"), multiplications_with_scalar)
    def test_multiplication_with_scalar(self, input_tuple, expected):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple:
            in1, in2 = self.Q_(*in1), in2
        else:
            in1, in2 = in1, self.Q_(*in2)
        input_tuple = in1, in2  # update input_tuple for better tracebacks
        if expected == 'error':
            self.assertRaises(OffsetUnitCalculusError, op.mul, in1, in2)
        else:
            expected = self.Q_(*expected)
            self.assertEqual(op.mul(in1, in2).units, expected.units)
            self.assertQuantityAlmostEqual(op.mul(in1, in2), expected,
                                           atol=0.01)

    divisions_with_scalar = [   # without / with autoconvert to base unit
        (((10, 'kelvin'), 2),       [(5., 'kelvin'), (5., 'kelvin')]),
        (((10, 'kelvin**2'), 2),    [(5., 'kelvin**2'), (5., 'kelvin**2')]),
        (((10, 'degC'), 2),         ['error', 'error']),
        (((10, 'degC**2'), 2),      ['error', 'error']),
        (((10, 'degC**-2'), 2),     ['error', 'error']),

        ((2, (10, 'kelvin')),       [(0.2, '1/kelvin'), (0.2, '1/kelvin')]),
        ((2, (10, 'degC')),         ['error', (2/283.15, '1/kelvin')]),
        ((2, (10, 'degC**2')),      ['error', 'error']),
        ((2, (10, 'degC**-2')),     ['error', 'error']),
        ]

    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"), divisions_with_scalar)
    def test_division_with_scalar(self, input_tuple, expected):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple:
            in1, in2 = self.Q_(*in1), in2
        else:
            in1, in2 = in1, self.Q_(*in2)
        input_tuple = in1, in2  # update input_tuple for better tracebacks
        expected_copy = expected[:]
        for i, mode in enumerate([False, True]):
            self.ureg.autoconvert_offset_to_baseunit = mode
            if expected_copy[i] == 'error':
                self.assertRaises(OffsetUnitCalculusError, op.truediv, in1, in2)
            else:
                expected = self.Q_(*expected_copy[i])
                self.assertEqual(op.truediv(in1, in2).units, expected.units)
                self.assertQuantityAlmostEqual(op.truediv(in1, in2), expected)

    exponentiation = [                  # resuls without / with autoconvert
        (((10, 'degC'),    1),          [(10, 'degC'), (10, 'degC')]),
        (((10, 'degC'),    0.5),        ['error', (283.15**0.5, 'kelvin**0.5')]),
        (((10, 'degC'),    0),          [(1., ''), (1., '')]),
        (((10, 'degC'),   -1),          ['error', (1/(10+273.15), 'kelvin**-1')]),
        (((10, 'degC'),   -2),          ['error', (1/(10+273.15)**2., 'kelvin**-2')]),
        ((( 0, 'degC'),   -2),          ['error', (1/(273.15)**2, 'kelvin**-2')]),
        (((10, 'degC'),   (2, '')),     ['error', ((283.15)**2, 'kelvin**2')]),
        (((10, 'degC'),  (10, 'degK')), ['error', 'error']),

        (((10, 'kelvin'), (2, '')),     [(100., 'kelvin**2'), (100., 'kelvin**2')]),

        ((  2,          (2, 'kelvin')), ['error', 'error']),
        ((  2,          (500., 'millikelvin/kelvin')), [2**0.5, 2**0.5]),
        ((  2,          (0.5, 'kelvin/kelvin')),      [2**0.5, 2**0.5]),
        (((10, 'degC'), (500., 'millikelvin/kelvin')),
                                        ['error', (283.15**0.5, 'kelvin**0.5')]),
         ]

    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"), exponentiation)
    def test_exponentiation(self, input_tuple, expected):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple and type(in2) is tuple:
            in1, in2 = self.Q_(*in1), self.Q_(*in2)
        elif not type(in1) is tuple and type(in2) is tuple:
            in2 = self.Q_(*in2)
        else:
            in1 = self.Q_(*in1)
        input_tuple = in1, in2
        expected_copy = expected[:]
        for i, mode in enumerate([False, True]):
            self.ureg.autoconvert_offset_to_baseunit = mode
            if expected_copy[i] == 'error':
                self.assertRaises((OffsetUnitCalculusError,
                                   DimensionalityError), op.pow, in1, in2)
            else:
                if type(expected_copy[i]) is tuple:
                    expected = self.Q_(*expected_copy[i])
                    self.assertEqual(op.pow(in1, in2).units, expected.units)
                else:
                    expected = expected_copy[i]
                self.assertQuantityAlmostEqual(op.pow(in1, in2), expected)

    @helpers.requires_numpy()
    @ParameterizedTestCase.parameterize(
        ("input", "expected_output"), exponentiation)
    def test_inplace_exponentiation(self, input_tuple, expected):
        self.ureg.default_as_delta = False
        in1, in2 = input_tuple
        if type(in1) is tuple and type(in2) is tuple:
            (q1v, q1u), (q2v, q2u) = in1, in2
            in1 = self.Q_(*(np.array([q1v]*2, dtype=np.float), q1u))
            in2 = self.Q_(q2v, q2u)
        elif not type(in1) is tuple and type(in2) is tuple:
            in2 = self.Q_(*in2)
        else:
            in1 = self.Q_(*in1)

        input_tuple = in1, in2

        expected_copy = expected[:]
        for i, mode in enumerate([False, True]):
            self.ureg.autoconvert_offset_to_baseunit = mode
            in1_cp = copy.copy(in1)
            if expected_copy[i] == 'error':
                self.assertRaises((OffsetUnitCalculusError,
                                   DimensionalityError), op.ipow, in1_cp, in2)
            else:
                if type(expected_copy[i]) is tuple:
                    expected = self.Q_(np.array([expected_copy[i][0]]*2,
                                                dtype=np.float),
                                       expected_copy[i][1])
                    self.assertEqual(op.ipow(in1_cp, in2).units, expected.units)
                else:
                    expected = np.array([expected_copy[i]]*2, dtype=np.float)


                in1_cp = copy.copy(in1)
                self.assertQuantityAlmostEqual(op.ipow(in1_cp, in2), expected)
