# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import unittest
import operator as op

from pint.unit import (ScaleConverter, OffsetConverter, UnitsContainer,
                       Definition, PrefixDefinition, UnitDefinition,
                       DimensionDefinition)


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
