# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import math
import copy
import itertools
import operator as op

from pint.unit import (ScaleConverter, OffsetConverter, UnitsContainer,
                       Definition, PrefixDefinition, UnitDefinition,
                       DimensionDefinition, _freeze, Converter, UnitRegistry,
                       LazyRegistry, ParserHelper)
from pint import DimensionalityError, UndefinedUnitError
from pint.compat import u, unittest, np, string_types
from pint.testsuite import QuantityTestCase, helpers, BaseTestCase
from pint.testsuite.parameterized import ParameterizedTestCase


class TestConverter(BaseTestCase):

    def test_converter(self):
        c = Converter()
        self.assertTrue(c.is_multiplicative)
        self.assertTrue(c.to_reference(8))
        self.assertTrue(c.from_reference(8))

    def test_multiplicative_converter(self):
        c = ScaleConverter(20.)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

    def test_offset_converter(self):
        c = OffsetConverter(20., 2)
        self.assertEqual(c.from_reference(c.to_reference(100)), 100)
        self.assertEqual(c.to_reference(c.from_reference(100)), 100)

    @helpers.requires_numpy()
    def test_converter_inplace(self):
        for c in (ScaleConverter(20.), OffsetConverter(20., 2)):
            fun1 = lambda x, y: c.from_reference(c.to_reference(x, y), y)
            fun2 = lambda x, y: c.to_reference(c.from_reference(x, y), y)
            for fun, (inplace, comp) in itertools.product((fun1, fun2),
                                                          ((True, self.assertIs), (False, self.assertIsNot))):
                a = np.ones((1, 10))
                ac = np.ones((1, 10))
                r = fun(a, inplace)
                np.testing.assert_allclose(r, ac)
                comp(a, r)


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


class TestUnitsContainer(QuantityTestCase):

    def _test_inplace(self, operator, value1, value2, expected_result):
        value1 = copy.copy(value1)
        value2 = copy.copy(value2)
        id1 = id(value1)
        id2 = id(value2)
        value1 = operator(value1, value2)
        value2_cpy = copy.copy(value2)
        self.assertEqual(value1, expected_result)
        self.assertEqual(id1, id(value1))
        self.assertEqual(value2, value2_cpy)
        self.assertEqual(id2, id(value2))

    def _test_not_inplace(self, operator, value1, value2, expected_result):
        id1 = id(value1)
        id2 = id(value2)

        value1_cpy = copy.copy(value1)
        value2_cpy = copy.copy(value2)

        result = operator(value1, value2)

        self.assertEqual(expected_result, result)
        self.assertEqual(value1, value1_cpy)
        self.assertEqual(value2, value2_cpy)
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

    def test_invalid(self):
        self.assertRaises(TypeError, UnitsContainer, {1: 2})
        self.assertRaises(TypeError, UnitsContainer, {'1': '2'})
        d = UnitsContainer()
        self.assertRaises(TypeError, d.__setitem__, 1, 2)
        self.assertRaises(TypeError, d.__setitem__, '1', '2')
        self.assertRaises(TypeError, d.__mul__, list())
        self.assertRaises(TypeError, d.__imul__, list())
        self.assertRaises(TypeError, d.__pow__, list())
        self.assertRaises(TypeError, d.__ipow__, list())
        self.assertRaises(TypeError, d.__truediv__, list())
        self.assertRaises(TypeError, d.__itruediv__, list())
        self.assertRaises(TypeError, d.__rtruediv__, list())


class TestRegistry(QuantityTestCase):

    FORCE_NDARRAY = False

    def setup(self):
        self.ureg.autoconvert_offset_to_baseunit = False

    def test_base(self):
        ureg = UnitRegistry(None)
        ureg.define('meter = [length]')
        self.assertRaises(ValueError, ureg.define, 'meter = [length]')
        self.assertRaises(TypeError, ureg.define, list())
        x = ureg.define('degC = kelvin; offset: 273.15')

    def test_define(self):
        ureg = UnitRegistry(None)
        self.assertIsInstance(dir(ureg), list)
        self.assertGreater(len(dir(ureg)), 0)

    def test_load(self):
        import pkg_resources
        from pint import unit
        data = pkg_resources.resource_filename(unit.__name__, 'default_en.txt')
        ureg1 = UnitRegistry()
        ureg2 = UnitRegistry(data)
        self.assertEqual(dir(ureg1), dir(ureg2))
        self.assertRaises(ValueError, UnitRegistry(None).load_definitions, 'notexisting')

    def test_default_format(self):
        ureg = UnitRegistry()
        q = ureg.meter
        s1 = '{0}'.format(q)
        s2 = '{0:~}'.format(q)
        ureg.default_format = '~'
        s3 = '{0}'.format(q)
        self.assertEqual(s2, s3)
        self.assertNotEqual(s1, s3)
        self.assertEqual(ureg.default_format, '~')

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
        self.assertEqual(str(UndefinedUnitError('rabbits')), "'{0!s}' is not defined in the unit registry".format('rabbits'))
        self.assertEqual(str(UndefinedUnitError(('rabbits', 'horses'))), "{0!s} are not defined in the unit registry".format(('rabbits', 'horses')))
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

    def test_parse_pretty(self):
        self.assertEqual(self.ureg.parse_expression('meter/second²'),
                         self.Q_(1, UnitsContainer(meter=1., second=-2)))
        self.assertEqual(self.ureg.parse_expression('m³/s³'),
                         self.Q_(1, UnitsContainer(meter=3., second=-3)))
        self.assertEqual(self.ureg.parse_expression('meter² · second'),
                         self.Q_(1, UnitsContainer(meter=2., second=1)))
        self.assertEqual(self.ureg.parse_expression('meter⁰.⁵·second'),
                         self.Q_(1, UnitsContainer(meter=0.5, second=1)))
        self.assertEqual(self.ureg.parse_expression('meter³⁷/second⁴.³²¹'),
                         self.Q_(1, UnitsContainer(meter=37, second=-4.321)))

    def test_parse_factor(self):
        self.assertEqual(self.ureg.parse_expression('42*meter'), self.Q_(42, UnitsContainer(meter=1.)))
        self.assertEqual(self.ureg.parse_expression('meter*42'), self.Q_(42, UnitsContainer(meter=1.)))

    def test_rep_and_parse(self):
        q = self.Q_(1, 'g/(m**2*s)')
        self.assertEqual(self.Q_(q.magnitude, str(q.units)), q)

    def test_as_delta(self):
        parse = self.ureg.parse_units
        self.assertEqual(parse('kelvin', as_delta=True), UnitsContainer(kelvin=1))
        self.assertEqual(parse('kelvin', as_delta=False), UnitsContainer(kelvin=1))
        self.assertEqual(parse('kelvin**(-1)', as_delta=True), UnitsContainer(kelvin=-1))
        self.assertEqual(parse('kelvin**(-1)', as_delta=False), UnitsContainer(kelvin=-1))
        self.assertEqual(parse('kelvin**2', as_delta=True), UnitsContainer(kelvin=2))
        self.assertEqual(parse('kelvin**2', as_delta=False), UnitsContainer(kelvin=2))
        self.assertEqual(parse('kelvin*meter', as_delta=True), UnitsContainer(kelvin=1, meter= 1))
        self.assertEqual(parse('kelvin*meter', as_delta=False), UnitsContainer(kelvin=1, meter=1))

    def test_name(self):
        self.assertRaises(UndefinedUnitError, self.ureg.get_name, 'asdf')

    def test_symbol(self):
        self.assertRaises(UndefinedUnitError, self.ureg.get_symbol, 'asdf')

        self.assertEqual(self.ureg.get_symbol('meter'), 'm')
        self.assertEqual(self.ureg.get_symbol('second'), 's')
        self.assertEqual(self.ureg.get_symbol('hertz'), 'Hz')

        self.assertEqual(self.ureg.get_symbol('kilometer'), 'km')
        self.assertEqual(self.ureg.get_symbol('megahertz'), 'MHz')
        self.assertEqual(self.ureg.get_symbol('millisecond'), 'ms')

    def test_imperial_symbol(self):
        self.assertEqual(self.ureg.get_symbol('inch'), 'in')
        self.assertEqual(self.ureg.get_symbol('foot'), 'ft')
        self.assertEqual(self.ureg.get_symbol('inches'), 'in')
        self.assertEqual(self.ureg.get_symbol('feet'), 'ft')
        self.assertEqual(self.ureg.get_symbol('international_foot'), 'ft')
        self.assertEqual(self.ureg.get_symbol('international_inch'), 'in')

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

        f1b = ureg.wraps(None, [ureg.meter, ])(func)
        self.assertRaises(ValueError, f1b, 3.)
        self.assertEqual(f1b(3. * ureg.centimeter), 0.03)
        self.assertEqual(f1b(3. * ureg.meter), 3.)
        self.assertRaises(ValueError, f1b, 3 * ureg.second)

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

    def test_to_ref_vs_to(self):
        self.ureg.autoconvert_offset_to_baseunit = True
        q = 8. * self.ureg.inch
        t = 8. * self.ureg.degF
        dt = 8. * self.ureg.delta_degF
        self.assertEqual(q.to('cm').magnitude, self.ureg._units['inch'].converter.to_reference(8.))
        self.assertEqual(t.to('kelvin').magnitude, self.ureg._units['degF'].converter.to_reference(8.))
        self.assertEqual(dt.to('kelvin').magnitude, self.ureg._units['delta_degF'].converter.to_reference(8.))

    def test_redefinition(self):
        d = UnitRegistry().define

        with self.capture_log() as buffer:
            d('meter = [fruits]')
            d('kilo- = 1000')
            d('[speed] = [vegetables]')

            # aliases
            d('bla = 3.2 meter = inch')
            d('myk- = 1000 = kilo-')

            self.assertEqual(len(buffer), 5)

    def test_convert_parse_str(self):
        ureg = self.ureg
        self.assertEqual(ureg.convert(1, 'meter', 'inch'),
                         ureg.convert(1, UnitsContainer(meter=1), UnitsContainer(inch=1)))

    @helpers.requires_numpy()
    def test_convert_inplace(self):
        ureg = self.ureg

        # Conversions with single units take a different codepath than
        # Conversions with more than one unit.
        src_dst1 = UnitsContainer(meter=1), UnitsContainer(inch=1)
        src_dst2 = UnitsContainer(meter=1, second=-1), UnitsContainer(inch=1, minute=-1)
        for src, dst in (src_dst1, src_dst2):
            v = ureg.convert(1, src, dst),

            a = np.ones((3, 1))
            ac = np.ones((3, 1))

            r1 = ureg.convert(a, src, dst)
            np.testing.assert_allclose(r1, v * ac)
            self.assertIsNot(r1, a)

            r2 = ureg.convert(a, src, dst, inplace=True)
            np.testing.assert_allclose(r2, v * ac)
            self.assertIs(r2, a)

    def test_repeated_convert(self):
        # Because of caching, repeated conversions were failing.
        self.ureg.convert(1, "m", "ft")
        self.ureg.convert(1, "m", "ft")

    def test_singular_SI_prefix_convert(self):
        # Fix for issue 156
        self.ureg.convert(1, 'mm', 'm')
        self.ureg.convert(1, 'ms', 's')
        self.ureg.convert(1, 'm', 'mm')
        self.ureg.convert(1, 's', 'ms')

    def test_parse_units(self):
        ureg = self.ureg
        self.assertEqual(ureg.parse_units(''), UnitsContainer())
        self.assertRaises(ValueError, ureg.parse_units, '2 * meter')


class TestCompatibleUnits(QuantityTestCase):

    FORCE_NDARRAY= False

    def _test(self, input_units):
        gd = self.ureg.get_dimensionality
        dim = gd(input_units)
        equiv = self.ureg.get_compatible_units(input_units)
        for eq in equiv:
            self.assertEqual(gd(eq), dim)
        self.assertEqual(equiv, self.ureg.get_compatible_units(dim))

    def _test2(self, units1, units2):
        equiv1 = self.ureg.get_compatible_units(units1)
        equiv2 = self.ureg.get_compatible_units(units2)
        self.assertEqual(equiv1, equiv2)

    def test_many(self):
        self._test(self.ureg.meter.units)
        self._test(self.ureg.seconds.units)
        self._test(self.ureg.newton.units)
        self._test(self.ureg.kelvin.units)

    def test_context_sp(self):


        gd = self.ureg.get_dimensionality

        # length, frequency, energy
        valid = [gd(self.ureg.meter.units), gd(self.ureg.hertz.units), gd(self.ureg.joule.units)]

        with self.ureg.context('sp'):
            equiv = self.ureg.get_compatible_units(self.ureg.meter.units)
            result = set()
            for eq in equiv:
                dim = gd(eq)
                result.add(_freeze(dim))
                self.assertIn(dim, valid)

            self.assertEqual(len(result), len(valid))

    def test_get_base_units(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_base_units(''), (1, UnitsContainer()))
        self.assertEqual(ureg.get_base_units('meter'), ureg.get_base_units(ParserHelper(meter=1)))

    def test_get_compatible_units(self):
        ureg = UnitRegistry()
        self.assertEqual(ureg.get_compatible_units(''), (1, UnitsContainer()))
        self.assertEqual(ureg.get_compatible_units('meter'), ureg.get_compatible_units(ParserHelper(meter=1)))


class TestRegistryWithDefaultRegistry(TestRegistry):

    @classmethod
    def setUpClass(cls):
        from pint import _DEFAULT_REGISTRY
        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity

    def test_lazy(self):
        x = LazyRegistry()
        x.test = 'test'
        self.assertIsInstance(x, UnitRegistry)
        y = LazyRegistry()
        q = y('meter')
        self.assertIsInstance(y, UnitRegistry)

    def test_redefinition(self):
        d = self.ureg.define
        self.assertRaises(ValueError, d, 'meter = [time]')
        self.assertRaises(ValueError, d, 'kilo- = 1000')
        self.assertRaises(ValueError, d, '[speed] = [length]')

        # aliases
        self.assertIn('inch', self.ureg._units)
        self.assertRaises(ValueError, d, 'bla = 3.2 meter = inch')
        self.assertRaises(ValueError, d, 'myk- = 1000 = kilo-')


class TestErrors(BaseTestCase):

    def test_errors(self):
        x = ('meter', )
        msg = "'meter' is not defined in the unit registry"
        self.assertEqual(str(UndefinedUnitError(x)), msg)
        self.assertEqual(str(UndefinedUnitError(list(x))), msg)
        self.assertEqual(str(UndefinedUnitError(set(x))), msg)

        msg = "Cannot convert from 'a' (c) to 'b' (d)msg"
        ex = DimensionalityError('a', 'b', 'c', 'd', 'msg')
        self.assertEqual(str(ex), msg)


class TestConvertWithOffset(QuantityTestCase, ParameterizedTestCase):

    # The dicts in convert_with_offset are used to create a UnitsContainer.
    # We create UnitsContainer to avoid any auto-conversion of units.
    convert_with_offset = [
        (({'degC': 1}, {'degC': 1}), 10),
        (({'degC': 1}, {'kelvin': 1}), 283.15),
        (({'degC': 1}, {'degC': 1, 'millimeter': 1, 'meter': -1}), 'error'),
        (({'degC': 1}, {'kelvin': 1, 'millimeter': 1, 'meter': -1}), 283150),

        (({'kelvin': 1}, {'degC': 1}), -263.15),
        (({'kelvin': 1}, {'kelvin': 1}), 10),
        (({'kelvin': 1}, {'degC': 1, 'millimeter': 1, 'meter': -1}), 'error'),
        (({'kelvin': 1}, {'kelvin': 1, 'millimeter': 1, 'meter': -1}), 10000),

        (({'degC': 1, 'millimeter': 1, 'meter': -1}, {'degC': 1}), 'error'),
        (({'degC': 1, 'millimeter': 1, 'meter': -1}, {'kelvin': 1}), 'error'),
        (({'degC': 1, 'millimeter': 1, 'meter': -1}, {'degC': 1, 'millimeter': 1, 'meter': -1}), 10),
        (({'degC': 1, 'millimeter': 1, 'meter': -1}, {'kelvin': 1, 'millimeter': 1, 'meter': -1}), 'error'),

        (({'kelvin': 1, 'millimeter': 1, 'meter': -1}, {'degC': 1}), -273.14),
        (({'kelvin': 1, 'millimeter': 1, 'meter': -1}, {'kelvin': 1}), 0.01),
        (({'kelvin': 1, 'millimeter': 1, 'meter': -1}, {'degC': 1, 'millimeter': 1, 'meter': -1}), 'error'),
        (({'kelvin': 1, 'millimeter': 1, 'meter': -1}, {'kelvin': 1, 'millimeter': 1, 'meter': -1}), 10),

        (({'degC': 2}, {'kelvin': 2}), 'error'),
        (({'degC': 1, 'degF': 1}, {'kelvin': 2}), 'error'),
        (({'degC': 1, 'kelvin': 1}, {'kelvin': 2}), 'error'),
        ]

    @ParameterizedTestCase.parameterize(("input", "expected_output"),
                                        convert_with_offset)
    def test_to_and_from_offset_units(self, input_tuple, expected):
        src, dst = input_tuple
        src, dst = UnitsContainer(src), UnitsContainer(dst)
        value = 10.
        convert = self.ureg.convert
        if isinstance(expected, string_types):
            self.assertRaises(DimensionalityError, convert, value, src, dst)
            if src != dst:
                self.assertRaises(DimensionalityError, convert, value, dst, src)
        else:
            self.assertQuantityAlmostEqual(convert(value, src, dst),
                                           expected, atol=0.001)
            if src != dst:
                self.assertQuantityAlmostEqual(convert(expected, dst, src),
                                               value, atol=0.001)
