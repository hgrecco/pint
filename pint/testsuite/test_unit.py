# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import math
import copy
import unittest
import operator as op

from pint.unit import (ScaleConverter, OffsetConverter, UnitsContainer,
                       Definition, PrefixDefinition, UnitDefinition,
                       DimensionDefinition)
from pint import DimensionalityError, UndefinedUnitError
from pint.testsuite import TestCase, u

class TestConverter(unittest.TestCase):

    def test_multiplicative_converter(self):
        c = ScaleConverter(20.)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

    def test_offset_converter(self):
        c = OffsetConverter(20., 2)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

class TestDefinition(unittest.TestCase):

    def test_prefix_definition(self):
        for definition in ('m- = 1e-3', 'm- = 10**-3', 'm- = 0.001'):
            x = Definition.from_string(definition)
            self.assertIsInstance(x, PrefixDefinition)
            self.assertEqual(x.name, 'm')
            self.assertEqual(x.aliases, ())
            self.assertEqual(x.converter.to_reference(.001), 1)
            self.assertEqual(x.converter.from_reference(1000), 1)

        x = Definition.from_string('kilo- = 1e-3 = k-')
        self.assertIsInstance(x, PrefixDefinition)
        self.assertEqual(x.name, 'kilo')
        self.assertEqual(x.aliases, ())
        self.assertEqual(x.symbol, 'k')
        self.assertEqual(x.converter.to_reference(.001), 1)
        self.assertEqual(x.converter.from_reference(1000), 1)

        x = Definition.from_string('kilo- = 1e-3 = k- = anotherk-')
        self.assertIsInstance(x, PrefixDefinition)
        self.assertEqual(x.name, 'kilo')
        self.assertEqual(x.aliases, ('anotherk', ))
        self.assertEqual(x.symbol, 'k')
        self.assertEqual(x.converter.to_reference(.001), 1)
        self.assertEqual(x.converter.from_reference(1000), 1)

    def test_baseunit_definition(self):
        x = Definition.from_string('meter = [length]')
        self.assertIsInstance(x, UnitDefinition)
        self.assertTrue(x.is_base)
        self.assertEqual(x.reference, UnitsContainer({'[length]': 1}))

    def test_unit_definition(self):
        x = Definition.from_string('coulomb = ampere * second')
        self.assertIsInstance(x, UnitDefinition)
        self.assertFalse(x.is_base)
        self.assertIsInstance(x.converter, ScaleConverter)
        self.assertEqual(x.converter.scale, 1)
        self.assertEqual(x.reference, UnitsContainer(ampere=1, second=1))

        x = Definition.from_string('faraday =  96485.3399 * coulomb')
        self.assertIsInstance(x, UnitDefinition)
        self.assertFalse(x.is_base)
        self.assertIsInstance(x.converter, ScaleConverter)
        self.assertEqual(x.converter.scale,  96485.3399)
        self.assertEqual(x.reference, UnitsContainer(coulomb=1))

        x = Definition.from_string('degF = 9 / 5 * degK; offset: 255.372222')
        self.assertIsInstance(x, UnitDefinition)
        self.assertFalse(x.is_base)
        self.assertIsInstance(x.converter, OffsetConverter)
        self.assertEqual(x.converter.scale, 9/5)
        self.assertEqual(x.converter.offset, 255.372222)
        self.assertEqual(x.reference, UnitsContainer(degK=1))

    def test_dimension_definition(self):
        x = Definition.from_string('[speed] = [length]/[time]')
        self.assertIsInstance(x, DimensionDefinition)
        self.assertEqual(x.reference, UnitsContainer({'[length]': 1, '[time]': -1}))


class TestUnitsContainer(unittest.TestCase):

    def _test_inplace(self, operator, value1, value2, expected_result):
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

    def test_unitcontainer_creation(self):
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

    def test_unitcontainer_repr(self):
        x = UnitsContainer()
        self.assertEqual(str(x), 'dimensionless')
        self.assertEqual(repr(x), '<UnitsContainer({})>')
        x = UnitsContainer(meter=1, second=2)
        self.assertEqual(str(x), 'meter * second ** 2')
        self.assertEqual(repr(x), "<UnitsContainer({'meter': 1.0, 'second': 2.0})>")
        x = UnitsContainer(meter=1, second=2.5)
        self.assertEqual(str(x), 'meter * second ** 2.5')
        self.assertEqual(repr(x), "<UnitsContainer({'meter': 1.0, 'second': 2.5})>")

    def test_unitcontainer_bool(self):
        self.assertTrue(UnitsContainer(meter=1, second=2))
        self.assertFalse(UnitsContainer())

    def test_unitcontainer_comp(self):
        x = UnitsContainer(meter=1, second=2)
        y = UnitsContainer(meter=1., second=2)
        z = UnitsContainer(meter=1, second=3)
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        self.assertFalse(x == z)
        self.assertTrue(x != z)

    def test_unitcontainer_arithmetic(self):
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

    def test_string_comparison(self):
        x = UnitsContainer(meter=1)
        y = UnitsContainer(second=1)
        z = UnitsContainer(meter=1, second=-2)
        self.assertEqual(x, 'meter')
        self.assertEqual('meter', x)
        self.assertNotEqual(x, 'meter ** 2')
        self.assertNotEqual(x, 'meter * meter')
        self.assertNotEqual(x, 'second')
        self.assertEqual(y, 'second')
        self.assertEqual(z, 'meter/second/second')


class TestRegistry(TestCase):

    FORCE_NDARRAY = False

    def test_parse_number(self):
        self.assertEqual(self.ureg.parse_expression('pi'), math.pi)
        self.assertEqual(self.ureg.parse_expression('x', x=2), 2)
        self.assertEqual(self.ureg.parse_expression('x', x=2.3), 2.3)
        self.assertEqual(self.ureg.parse_expression('x * y', x=2.3, y=3), 2.3 * 3)
        self.assertEqual(self.ureg.parse_expression('x', x=(1+1j)), (1+1j))

    def test_parse_single(self):
        self.assertEqual(self.ureg.parse_expression('meter'), self.Q_(1, UnitsContainer(meter=1.)))
        self.assertEqual(self.ureg.parse_expression('second'), self.Q_(1, UnitsContainer(second=1.)))

    def test_parse_alias(self):
        self.assertEqual(self.ureg.parse_expression('metre'), self.Q_(1, UnitsContainer(meter=1.)))

    def test_parse_plural(self):
        self.assertEqual(self.ureg.parse_expression('meters'), self.Q_(1, UnitsContainer(meter=1.)))

    def test_parse_prefix(self):
        self.assertEqual(self.ureg.parse_expression('kilometer'), self.Q_(1, UnitsContainer(kilometer=1.)))
        #self.assertEqual(self.ureg._units['kilometer'], self.Q_(1000., UnitsContainer(meter=1.)))

    def test_parse_complex(self):
        self.assertEqual(self.ureg.parse_expression('kilometre'), self.Q_(1, UnitsContainer(kilometer=1.)))
        self.assertEqual(self.ureg.parse_expression('kilometres'), self.Q_(1, UnitsContainer(kilometer=1.)))


    def test_str_errors(self):
        self.assertEqual(str(UndefinedUnitError('rabbits')), "'{!s}' is not defined in the unit registry".format('rabbits'))
        self.assertEqual(str(UndefinedUnitError(('rabbits', 'horses'))), "{!s} are not defined in the unit registry".format(('rabbits', 'horses')))
        self.assertEqual(u(str(DimensionalityError('meter', 'second'))),
                         "Cannot convert from 'meter' to 'second'")
        self.assertEqual(str(DimensionalityError('meter', 'second', 'length', 'time')),
                         "Cannot convert from 'meter' (length) to 'second' (time)")

    def test_parse_mul_div(self):
        self.assertEqual(self.ureg.parse_expression('meter*meter'), self.Q_(1, UnitsContainer(meter=2.)))
        self.assertEqual(self.ureg.parse_expression('meter**2'), self.Q_(1, UnitsContainer(meter=2.)))
        self.assertEqual(self.ureg.parse_expression('meter*second'), self.Q_(1, UnitsContainer(meter=1., second=1)))
        self.assertEqual(self.ureg.parse_expression('meter/second'), self.Q_(1, UnitsContainer(meter=1., second=-1)))
        self.assertEqual(self.ureg.parse_expression('meter/second**2'), self.Q_(1, UnitsContainer(meter=1., second=-2)))

    def test_parse_factor(self):
        self.assertEqual(self.ureg.parse_expression('42*meter'), self.Q_(42, UnitsContainer(meter=1.)))
        self.assertEqual(self.ureg.parse_expression('meter*42'), self.Q_(42, UnitsContainer(meter=1.)))

    def test_rep_and_parse(self):
        q = self.Q_(1, 'g/(m**2*s)')
        self.assertEqual(self.Q_(q.magnitude, str(q.units)), q)

    def test_to_delta(self):
        parse = self.ureg.parse_units
        self.assertEqual(parse('degK', to_delta=True), UnitsContainer(degK=1))
        self.assertEqual(parse('degK', to_delta=False), UnitsContainer(degK=1))
        self.assertEqual(parse('degK**(-1)', to_delta=True), UnitsContainer(degK=-1))
        self.assertEqual(parse('degK**(-1)', to_delta=False), UnitsContainer(degK=-1))
        self.assertEqual(parse('degK**2', to_delta=True), UnitsContainer(delta_degK=2))
        self.assertEqual(parse('degK**2', to_delta=False), UnitsContainer(degK=2))
        self.assertEqual(parse('degK*meter', to_delta=True), UnitsContainer(delta_degK=1, meter= 1))
        self.assertEqual(parse('degK*meter', to_delta=False), UnitsContainer(degK=1, meter=1))

    def test_symbol(self):
        self.assertEqual(self.ureg.get_symbol('meter'), 'm')
        self.assertEqual(self.ureg.get_symbol('second'), 's')
        self.assertEqual(self.ureg.get_symbol('hertz'), 'Hz')

        self.assertEqual(self.ureg.get_symbol('kilometer'), 'km')
        self.assertEqual(self.ureg.get_symbol('megahertz'), 'MHz')
        self.assertEqual(self.ureg.get_symbol('millisecond'), 'ms')

    @unittest.expectedFailure
    def test_delta_in_diff(self):
        """This might be supported in future versions
        """
        xk = 1 * self.ureg.degK
        yk = 2 * self.ureg.degK
        yf = yk.to('degF')
        yc = yk.to('degC')
        self.assertEqual(yk - xk, 1 * self.ureg.delta_degK)
        self.assertEqual(yf - xk, 1 * self.ureg.delta_degK)
        self.assertEqual(yc - xk, 1 * self.ureg.delta_degK)

    def test_pint(self):
        p = self.ureg.pint
        l = self.ureg.liter
        ip = self.ureg.imperial_pint
        self.assertLess(p, l)
        self.assertLess(p, ip)

    def test_wraps(self):
        def func(x):
            return x

        ureg = self.ureg

        f0 = ureg.wraps(None, [None, ])(func)
        self.assertEqual(f0(3.), 3.)

        f0 = ureg.wraps(None, None, )(func)
        self.assertEqual(f0(3.), 3.)

        f1 = ureg.wraps(None, ['meter', ])(func)
        self.assertRaises(ValueError, f1, 3.)
        self.assertEqual(f1(3. * ureg.centimeter), 0.03)
        self.assertEqual(f1(3. * ureg.meter), 3.)
        self.assertRaises(ValueError, f1, 3 * ureg.second)

        f1 = ureg.wraps(None, 'meter')(func)
        self.assertRaises(ValueError, f1, 3.)
        self.assertEqual(f1(3. * ureg.centimeter), 0.03)
        self.assertEqual(f1(3. * ureg.meter), 3.)
        self.assertRaises(ValueError, f1, 3 * ureg.second)

        f2 = ureg.wraps('centimeter', ['meter', ])(func)
        self.assertRaises(ValueError, f2, 3.)
        self.assertEqual(f2(3. * ureg.centimeter), 0.03 * ureg.centimeter)
        self.assertEqual(f2(3. * ureg.meter), 3 * ureg.centimeter)

        f3 = ureg.wraps('centimeter', ['meter', ], strict=False)(func)
        self.assertEqual(f3(3), 3 * ureg.centimeter)
        self.assertEqual(f3(3. * ureg.centimeter), 0.03 * ureg.centimeter)
        self.assertEqual(f3(3. * ureg.meter), 3. * ureg.centimeter)

        def gfunc(x, y):
            return x + y

        g0 = ureg.wraps(None, [None, None])(gfunc)
        self.assertEqual(g0(3, 1), 4)

        g1 = ureg.wraps(None, ['meter', 'centimeter'])(gfunc)
        self.assertRaises(ValueError, g1, 3 * ureg.meter, 1)
        self.assertEqual(g1(3 * ureg.meter, 1 * ureg.centimeter), 4)
        self.assertEqual(g1(3 * ureg.meter, 1 * ureg.meter), 3 + 100)

        def hfunc(x, y):
            return x, y

        h0 = ureg.wraps(None, [None, None])(hfunc)
        self.assertEqual(h0(3, 1), (3, 1))

        h1 = ureg.wraps(['meter', 'cm'], [None, None])(hfunc)
        self.assertEqual(h1(3, 1), [3 * ureg.meter, 1 * ureg.cm])

        h2 = ureg.wraps(('meter', 'cm'), [None, None])(hfunc)
        self.assertEqual(h2(3, 1), (3 * ureg.meter, 1 * ureg.cm))
