# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import sys

import copy
import math
import operator as op

PYTHON3 = sys.version >= '3'

import unittest

from pint import UnitRegistry, UnitsContainer, DimensionalityError, UndefinedUnitError
from pint.pint import _definitions_from_file

from tests import TestCase, string_types, u

class TestPint(TestCase):

    FORCE_NDARRAY = False

    def _test_inplace(self, operator, value1, value2, expected_result):
        if isinstance(value1, string_types):
            value1 = self.Q_(value1)
        if isinstance(value2, string_types):
            value2 = self.Q_(value2)
        if isinstance(expected_result, string_types):
            expected_result = self.Q_(expected_result)

        value1 = copy.copy(value1)
        value2 = copy.copy(value2)
        id1 = id(value1)
        id2 = id(value2)
        value1 = operator(value1, value2)
        value2_cpy = copy.copy(value2)
        self.assertAlmostEqual(value1, expected_result)
        self.assertEqual(id1, id(value1))
        self.assertAlmostEqual(value2, value2_cpy)
        self.assertEqual(id2, id(value2))

    def _test_not_inplace(self, operator, value1, value2, expected_result):
        if isinstance(value1, string_types):
            value1 = self.Q_(value1)
        if isinstance(value2, string_types):
            value2 = self.Q_(value2)
        if isinstance(expected_result, string_types):
            expected_result = self.Q_(expected_result)

        id1 = id(value1)
        id2 = id(value2)

        value1_cpy = copy.copy(value1)
        value2_cpy = copy.copy(value2)

        result = operator(value1, value2)

        self.assertAlmostEqual(expected_result, result)
        self.assertAlmostEqual(value1, value1_cpy)
        self.assertAlmostEqual(value2, value2_cpy)
        self.assertNotEqual(id(result), id1)
        self.assertNotEqual(id(result), id2)

    def test_units_creation(self):
        x = UnitsContainer(meter=1, second=2)
        y = UnitsContainer({'meter': 1.0, 'second': 2.0})
        self.assertIsInstance(x['meter'], float)
        self.assertEqual(x, y)
        self.assertIsNot(x, y)
        z = copy.copy(x)
        self.assertEqual(x, z)
        self.assertIsNot(x, z)
        z = UnitsContainer(x)
        self.assertEqual(x, z)
        self.assertIsNot(x, z)

    def test_units_repr(self):
        x = UnitsContainer()
        self.assertEqual(str(x), 'dimensionless')
        self.assertEqual(repr(x), '<UnitsContainer({})>')
        x = UnitsContainer(meter=1, second=2)
        self.assertEqual(str(x), 'meter * second ** 2')
        self.assertEqual(repr(x), "<UnitsContainer({'meter': 1.0, 'second': 2.0})>")
        x = UnitsContainer(meter=1, second=2.5)
        self.assertEqual(str(x), 'meter * second ** 2.5')
        self.assertEqual(repr(x), "<UnitsContainer({'meter': 1.0, 'second': 2.5})>")

    def test_units_bool(self):
        self.assertTrue(UnitsContainer(meter=1, second=2))
        self.assertFalse(UnitsContainer())

    def test_units_comp(self):
        x = UnitsContainer(meter=1, second=2)
        y = UnitsContainer(meter=1., second=2)
        z = UnitsContainer(meter=1, second=3)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x == z)
        self.assertTrue(x != z)

    def test_units_arithmetic(self):
        x = UnitsContainer(meter=1)
        y = UnitsContainer(second=1)
        z = UnitsContainer(meter=1, second=-2)

        self._test_not_inplace(op.mul, x, y, UnitsContainer(meter=1, second=1))
        self._test_not_inplace(op.truediv, x, y, UnitsContainer(meter=1, second=-1))
        self._test_not_inplace(op.pow, z, 2, UnitsContainer(meter=2, second=-4))
        self._test_not_inplace(op.pow, z, -2, UnitsContainer(meter=-2, second=4))

        self._test_inplace(op.imul, x, y, UnitsContainer(meter=1, second=1))
        self._test_inplace(op.itruediv, x, y, UnitsContainer(meter=1, second=-1))
        self._test_inplace(op.ipow, z, 2, UnitsContainer(meter=2, second=-4))
        self._test_inplace(op.ipow, z, -2, UnitsContainer(meter=-2, second=4))

    def test_str_errors(self):
        self.assertEqual(str(UndefinedUnitError('rabbits')), "'{!s}' is not defined in the unit registry.".format('rabbits'))
        self.assertEqual(str(UndefinedUnitError(('rabbits', ))), "{!s} are not defined in the unit registry.".format(('rabbits',)))
        self.assertEqual(u(str(DimensionalityError('meter', 'second'))),
                         "Cannot convert from 'meter' to 'second'.")
        self.assertEqual(str(DimensionalityError('meter', 'second', 'length', 'time')),
                         "Cannot convert from 'meter' (length) to 'second' (time).")

    def test_parse_single(self):
        self.assertEqual(self.ureg._parse_expression('meter'), self.Q_(1, UnitsContainer(meter=1.)))
        self.assertEqual(self.ureg._parse_expression('second'), self.Q_(1, UnitsContainer(second=1.)))

    def test_parse_alias(self):
        self.assertEqual(self.ureg._parse_expression('metre'), self.Q_(1, UnitsContainer(meter=1.)))

    def test_parse_plural(self):
        self.assertEqual(self.ureg._parse_expression('meters'), self.Q_(1, UnitsContainer(meter=1.)))

    def test_parse_prefix(self):
        self.assertEqual(self.ureg._parse_expression('kilometer'), self.Q_(1, UnitsContainer(kilometer=1.)))
        self.assertEqual(self.ureg._UNITS['kilometer'], self.Q_(1000., UnitsContainer(meter=1.)))

    def test_parse_complex(self):
        self.assertEqual(self.ureg._parse_expression('kilometre'), self.Q_(1, UnitsContainer(kilometer=1.)))
        self.assertEqual(self.ureg._parse_expression('kilometres'), self.Q_(1, UnitsContainer(kilometer=1.)))

    def test_parse_mul_div(self):
        self.assertEqual(self.ureg._parse_expression('meter*meter'), self.Q_(1, UnitsContainer(meter=2.)))
        self.assertEqual(self.ureg._parse_expression('meter**2'), self.Q_(1, UnitsContainer(meter=2.)))
        self.assertEqual(self.ureg._parse_expression('meter*second'), self.Q_(1, UnitsContainer(meter=1., second=1)))
        self.assertEqual(self.ureg._parse_expression('meter/second'), self.Q_(1, UnitsContainer(meter=1., second=-1)))
        self.assertEqual(self.ureg._parse_expression('meter/second**2'), self.Q_(1, UnitsContainer(meter=1., second=-2)))

    def test_parse_factor(self):
        self.assertEqual(self.ureg._parse_expression('42*meter'), self.Q_(42, UnitsContainer(meter=1.)))
        self.assertEqual(self.ureg._parse_expression('meter*42'), self.Q_(42, UnitsContainer(meter=1.)))

    def test_dimensionality(self):
        x = self.Q_(42, 'centimeter')
        x.convert_to_reference()
        x = self.Q_(42, 'meter*second')
        self.assertEqual(x.dimensionality, UnitsContainer(length=1., time=1.))
        x = self.Q_(42, 'meter*second*second')
        self.assertEqual(x.dimensionality, UnitsContainer(length=1., time=2.))
        x = self.Q_(42, 'inch*second*second')
        self.assertEqual(x.dimensionality, UnitsContainer(length=1., time=2.))
        self.assertTrue(self.Q_(42, None).dimensionless)
        self.assertFalse(self.Q_(42, 'meter').dimensionless)
        self.assertTrue((self.Q_(42, 'meter') / self.Q_(1, 'meter')).dimensionless)
        self.assertFalse((self.Q_(42, 'meter') / self.Q_(1, 'second')).dimensionless)
        self.assertTrue((self.Q_(42, 'meter') / self.Q_(1, 'inch')).dimensionless)

    def test_quantity_creation(self):
        for args in ((4.2, 'meter'),
                     (4.2,  UnitsContainer(meter=1)),
                     ('4.2*meter', ),
                     ('4.2/meter**(-1)', ),
                     (self.Q_(4.2, 'meter'),)):
            x = self.Q_(*args)
            self.assertEqual(x.magnitude, 4.2)
            self.assertEqual(x.units, UnitsContainer(meter=1))

        x = self.Q_(None, UnitsContainer(length=1))
        self.assertEqual(x.magnitude, None)
        self.assertEqual(x.units, UnitsContainer(length=1))

        x = self.Q_(4.2, UnitsContainer(length=1))
        y = self.Q_(x)
        self.assertEqual(x.magnitude, y.magnitude)
        self.assertEqual(x.units, y.units)
        self.assertIsNot(x, y)

        x = self.Q_(4.2, None)
        self.assertEqual(x.magnitude, 4.2)
        self.assertEqual(x.units, UnitsContainer())

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
        self.assertTrue(x == x)
        self.assertFalse(x != x)

        self.assertTrue(x <= y)
        self.assertTrue(x >= y)
        self.assertFalse(x < y)
        self.assertFalse(x > y)

        self.assertTrue(x != z)
        self.assertTrue(x < z)

        self.assertTrue(z != j)

        self.assertNotEqual(z, j)
        self.assertEqual(self.Q_(0, 'meter'), self.Q_(0, 'centimeter'))
        self.assertNotEqual(self.Q_(0, 'meter'), self.Q_(0, 'second'))

    def test_quantity_comparison_convert(self):
        self.assertEqual(self.Q_(1000, 'millimeter'), self.Q_(1, 'meter'))
        self.assertEqual(self.Q_(1000, 'millimeter/min'), self.Q_(1000/60, 'millimeter/s'))

    def test_quantity_repr(self):
        x = self.Q_(4.2, UnitsContainer(meter=1))
        self.assertEqual(str(x), '4.2 meter')
        self.assertEqual(repr(x), "<Quantity(4.2, 'meter')>")

    def test_quantity_format(self):
        x = self.Q_(4.12345678, UnitsContainer(meter=2, gram=1, second=-1))
        for spec, result in (('{}', str(x)), ('{!s}', str(x)), ('{!r}', repr(x)),
                             ('{0.magnitude}',  str(x.magnitude)), ('{0.units}',  str(x.units)),
                             ('{0.magnitude!s}',  str(x.magnitude)), ('{0.units!s}',  str(x.units)),
                             ('{0.magnitude!r}',  repr(x.magnitude)), ('{0.units!r}',  repr(x.units)),
                             ('{:.4f}', '{:.4f} {!s}'.format(x.magnitude, x.units)),
                             ('{:!l}', r'4.12345678 \frac{gram \cdot meter^{2}}{second}'),
                             ('{:!p}', '4.12345678 gram·meter²/second')):
            self.assertEqual(spec.format(x), result)

    def test_quantity_add_sub(self):
        x = self.Q_(1., 'centimeter')
        y = self.Q_(1., 'inch')
        z = self.Q_(1., 'second')
        a = self.Q_(1., None)

        self._test_not_inplace(op.add, x, x, self.Q_(2., 'centimeter'))
        self._test_not_inplace(op.add, x, y, self.Q_(1 + 2.54, 'centimeter'))
        self._test_not_inplace(op.add, y, x, self.Q_(1 + 1 / 2.54, 'inch'))
        self._test_not_inplace(op.add, a, 1, self.Q_(1 + 1, None))
        self.assertRaises(DimensionalityError, op.add, 10, x)
        self.assertRaises(DimensionalityError, op.add, x, 10)
        self.assertRaises(DimensionalityError, op.add, x, z)

        self._test_not_inplace(op.sub, x, x, self.Q_(0., 'centimeter'))
        self._test_not_inplace(op.sub, x, y, self.Q_(1 - 2.54, 'centimeter'))
        self._test_not_inplace(op.sub, y, x, self.Q_(1 - 1 / 2.54, 'inch'))
        self._test_not_inplace(op.sub, a, 1, self.Q_(1 - 1, None))
        self.assertRaises(DimensionalityError, op.sub, 10, x)
        self.assertRaises(DimensionalityError, op.sub, x, 10)
        self.assertRaises(DimensionalityError, op.sub, x, z)

    def test_quantity_iadd_isub(self):
        x = self.Q_(1., 'centimeter')
        y = self.Q_(1., 'inch')
        z = self.Q_(1., 'second')
        a = self.Q_(1., None)

        self._test_inplace(op.iadd, x, x, self.Q_(2., 'centimeter'))
        self._test_inplace(op.iadd, x, y, self.Q_(1 + 2.54, 'centimeter'))
        self._test_inplace(op.iadd, y, x, self.Q_(1 + 1 / 2.54, 'inch'))
        self._test_inplace(op.iadd, a, 1, self.Q_(1 + 1, None))
        self.assertRaises(DimensionalityError, op.iadd, 10, x)
        self.assertRaises(DimensionalityError, op.iadd, x, 10)
        self.assertRaises(DimensionalityError, op.iadd, x, z)

        self._test_inplace(op.isub, x, x, self.Q_(0., 'centimeter'))
        self._test_inplace(op.isub, x, y, self.Q_(1 - 2.54, 'centimeter'))
        self._test_inplace(op.isub, y, x, self.Q_(1 - 1 / 2.54, 'inch'))
        self._test_inplace(op.isub, a, 1, self.Q_(1 - 1, None))
        self.assertRaises(DimensionalityError, op.sub, 10, x)
        self.assertRaises(DimensionalityError, op.sub, x, 10)
        self.assertRaises(DimensionalityError, op.sub, x, z)

    def test_quantity_mul_div(self):
        self._test_not_inplace(op.mul, 10.0, '4.2*meter', '42*meter')
        self._test_not_inplace(op.mul, '4.2*meter', 10.0, '42*meter')
        self._test_not_inplace(op.mul, '4.2*meter', '10*inch', '42*meter*inch')
        self._test_not_inplace(op.truediv, 42, '4.2*meter', '10/meter')
        self._test_not_inplace(op.truediv, '4.2*meter', 10.0, '0.42*meter')
        self._test_not_inplace(op.truediv, '4.2*meter', '10*inch', '0.42*meter/inch')

    def test_quantity_imul_idiv(self):
        #self._test_inplace(op.imul, 10.0, '4.2*meter', '42*meter')
        self._test_inplace(op.imul, '4.2*meter', 10.0, '42*meter')
        self._test_inplace(op.imul, '4.2*meter', '10*inch', '42*meter*inch')
        #self._test_not_inplace(op.truediv, 42, '4.2*meter', '10/meter')
        self._test_inplace(op.itruediv, '4.2*meter', 10.0, '0.42*meter')
        self._test_inplace(op.itruediv, '4.2*meter', '10*inch', '0.42*meter/inch')

    def test_quantity_floordiv(self):
        self._test_not_inplace(op.floordiv, 10.0, '4.2*meter', '2/meter')
        self._test_not_inplace(op.floordiv, '24*meter', 10.0, '2*meter')
        self._test_not_inplace(op.floordiv, '10*meter', '4.2*inch', '2*meter/inch')

        #self._test_inplace(op.ifloordiv, 10.0, '4.2*meter', '2/meter')
        self._test_inplace(op.ifloordiv, '24*meter', 10.0, '2*meter')
        self._test_inplace(op.ifloordiv, '10*meter', '4.2*inch', '2*meter/inch')

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
            self.assertEqual(rx, zx, 'while testing {}'.format(fun))
            self.assertEqual(ry, zy, 'while testing {}'.format(fun))
            self.assertIsNot(rx, zx, 'while testing {}'.format(fun))
            self.assertIsNot(ry, zy, 'while testing {}'.format(fun))

    def test_quantity_float_complex(self):
        x = self.Q_(-4.2, None)
        y = self.Q_(4.2, None)
        z = self.Q_(1, 'meter')
        for fun in (float, complex):
            self.assertEqual(fun(x), fun(x.magnitude))
            self.assertEqual(fun(y), fun(y.magnitude))
            self.assertRaises(DimensionalityError, fun, z)

    def test_convert_to_reference(self):
        x = self.Q_('1*inch')
        self.assertAlmostEqual(x.convert_to_reference(), self.Q_(0.0254, 'meter'))
        x = self.Q_('1*inch*inch')
        self.assertAlmostEqual(x.convert_to_reference(), self.Q_(0.0254 ** 2.0, 'meter*meter'))
        x = self.Q_('1*inch/minute')
        self.assertAlmostEqual(x.convert_to_reference(), self.Q_(0.0254 / 60., 'meter/second'))

    def test_convert(self):
        x = self.Q_('2*inch')
        self.assertAlmostEqual(x.to('meter'), self.Q_(2. * 0.0254, 'meter'))
        x = self.Q_('2*meter')
        self.assertAlmostEqual(x.to('inch'), self.Q_(2. / 0.0254, 'inch'))
        x = self.Q_('2*sidereal_second')
        self.assertAlmostEqual(x.to('second'), self.Q_(1.994539133 , 'second'))
        x = self.Q_('2.54*centimeter/second')
        self.assertAlmostEqual(x.to('inch/second'), self.Q_(1, 'inch/second'))
        x = self.Q_('2.54*centimeter')
        self.assertAlmostEqual(x.to('inch').magnitude, 1)
        self.assertAlmostEqual(self.Q_(2, 'second').to('millisecond').magnitude, 2000)

    def test_context_attr(self):
        self.assertEqual(self.ureg.meter, self.Q_(1, 'meter'))

    def test_short(self):
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

    @unittest.expectedFailure
    def test_offset(self):
        self.assertAlmostEqual(self.Q_(0, 'degK').to('degK'), self.Q_(0, 'degK'))
        self.assertAlmostEqual(self.Q_(0, 'degC').to('degK'), self.Q_(273.15, 'degK'))
        self.assertAlmostEqual(self.Q_(0, 'degF').to('degK'), self.Q_(255.372222, 'degK'), places=2)

        self.assertAlmostEqual(self.Q_(100, 'degK').to('degK'), self.Q_(100, 'degK'))
        self.assertAlmostEqual(self.Q_(100, 'degC').to('degK'), self.Q_(373.15, 'degK'))
        self.assertAlmostEqual(self.Q_(100, 'degF').to('degK'), self.Q_(310.92777777, 'degK'), places=2)

        self.assertAlmostEqual(self.Q_(0, 'degK').to('degC'), self.Q_(-273.15, 'degC'))
        self.assertAlmostEqual(self.Q_(100, 'degK').to('degC'), self.Q_(-173.15, 'degC'))
        self.assertAlmostEqual(self.Q_(0, 'degK').to('degF'), self.Q_(-459.67, 'degF'), 2)
        self.assertAlmostEqual(self.Q_(100, 'degK').to('degF'), self.Q_(-279.67, 'degF'), 2)


        self.assertAlmostEqual(self.Q_(32, 'degF').to('degC'), self.Q_(0, 'degC'), 2)
        self.assertAlmostEqual(self.Q_(100, 'degC').to('degF'), self.Q_(212, 'degF'), 2)

    def test_from_definitions(self):
        #self.assertEqual(self.ureg.mm, self.ureg.millimeter)
        for filename in self.ureg._definition_files:
            for name, value, aliases, modifiers in _definitions_from_file(filename):
                if '[' in value or '-' in name:
                    continue
                msg = '{} to {}'.format(name, value)
                if not modifiers:
                    self.assertAlmostEqual(self.Q_(name), self.Q_(value), msg=msg)
                    self.assertAlmostEqual(self.Q_(name) ** 2, self.Q_(value) ** 2, msg=msg)
                    self.assertAlmostEqual(self.Q_(name) * 23.23, self.Q_(value) * 23.23, msg=msg)
