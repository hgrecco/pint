# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

from pint.util import (UnitsContainer)
from pint.converters import (ScaleConverter, OffsetConverter)
from pint.definitions import (Definition, PrefixDefinition, UnitDefinition,
                              DimensionDefinition)

from pint.testsuite import BaseTestCase

class TestDefinition(BaseTestCase):

    def test_invalid(self):
        self.assertRaises(ValueError, Definition.from_string, 'x = [time] * meter')
        self.assertRaises(ValueError, Definition.from_string, '[x] = [time] * meter')

    def test_prefix_definition(self):
        for definition in ('m- = 1e-3', 'm- = 10**-3', 'm- = 0.001'):
            x = Definition.from_string(definition)
            self.assertIsInstance(x, PrefixDefinition)
            self.assertEqual(x.name, 'm')
            self.assertEqual(x.aliases, ())
            self.assertEqual(x.converter.to_reference(1000), 1)
            self.assertEqual(x.converter.from_reference(0.001), 1)
            self.assertEqual(str(x), 'm')

        x = Definition.from_string('kilo- = 1e-3 = k-')
        self.assertIsInstance(x, PrefixDefinition)
        self.assertEqual(x.name, 'kilo')
        self.assertEqual(x.aliases, ())
        self.assertEqual(x.symbol, 'k')
        self.assertEqual(x.converter.to_reference(1000), 1)
        self.assertEqual(x.converter.from_reference(.001), 1)

        x = Definition.from_string('kilo- = 1e-3 = k- = anotherk-')
        self.assertIsInstance(x, PrefixDefinition)
        self.assertEqual(x.name, 'kilo')
        self.assertEqual(x.aliases, ('anotherk', ))
        self.assertEqual(x.symbol, 'k')
        self.assertEqual(x.converter.to_reference(1000), 1)
        self.assertEqual(x.converter.from_reference(.001), 1)

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

        x = Definition.from_string('degF = 9 / 5 * kelvin; offset: 255.372222')
        self.assertIsInstance(x, UnitDefinition)
        self.assertFalse(x.is_base)
        self.assertIsInstance(x.converter, OffsetConverter)
        self.assertEqual(x.converter.scale, 9/5)
        self.assertEqual(x.converter.offset, 255.372222)
        self.assertEqual(x.reference, UnitsContainer(kelvin=1))

    def test_dimension_definition(self):
        x = DimensionDefinition('[time]', '', (), converter='')
        self.assertTrue(x.is_base)
        self.assertEqual(x.name, '[time]')

        x = Definition.from_string('[speed] = [length]/[time]')
        self.assertIsInstance(x, DimensionDefinition)
        self.assertEqual(x.reference, UnitsContainer({'[length]': 1, '[time]': -1}))
