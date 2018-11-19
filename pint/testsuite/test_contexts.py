# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import itertools
from collections import defaultdict

from pint import UnitRegistry, errors
from pint.context import Context
from pint.util import UnitsContainer
from pint.testsuite import QuantityTestCase


def add_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('lc')
    d.add_transformation(a, b, lambda ureg, x: ureg.speed_of_light / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.speed_of_light / x)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': 1})
    d = Context('ab')
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_arg_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('lc')
    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': 1})
    d = Context('ab')
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_argdef_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('lc', defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': 1})
    d = Context('ab')
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_sharedargdef_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('lc', defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': 1})
    d = Context('ab', defaults=dict(n=0))
    d.add_transformation(a, b, lambda ureg, x, n: ureg.ampere * ureg.meter * n / x)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.ampere * ureg.meter * n / x)

    ureg.add_context(d)


class TestContexts(QuantityTestCase):

    def test_known_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        with ureg.context('lc'):
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

        with ureg.context('lc', n=1):
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_known_context_enable(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        ureg.enable_contexts('lc')
        self.assertTrue(ureg._active_ctx)
        self.assertTrue(ureg._active_ctx.graph)
        ureg.disable_contexts(1)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

        ureg.enable_contexts('lc', n=1)
        self.assertTrue(ureg._active_ctx)
        self.assertTrue(ureg._active_ctx.graph)
        ureg.disable_contexts(1)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_graph(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = UnitsContainer({'[length]': 1.})
        t = UnitsContainer({'[time]': -1.})
        c = UnitsContainer({'[current]': 1.})

        g_sp = defaultdict(set)
        g_sp.update({l: set((t,)),
                     t: set((l,))})

        g_ab = defaultdict(set)
        g_ab.update({l: set((c,)),
                     c: set((l,))})

        g = defaultdict(set)
        g.update({l: set((t, c)),
                  t: set((l,)),
                  c: set((l,))})

        with ureg.context('lc'):
            self.assertEqual(ureg._active_ctx.graph, g_sp)

        with ureg.context('lc', n=1):
            self.assertEqual(ureg._active_ctx.graph, g_sp)

        with ureg.context('ab'):
            self.assertEqual(ureg._active_ctx.graph, g_ab)

        with ureg.context('lc'):
            with ureg.context('ab'):
                self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('ab'):
            with ureg.context('lc'):
                self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('lc', 'ab'):
            self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('ab', 'lc'):
            self.assertEqual(ureg._active_ctx.graph, g)

    def test_graph_enable(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = UnitsContainer({'[length]': 1.})
        t = UnitsContainer({'[time]': -1.})
        c = UnitsContainer({'[current]': 1.})

        g_sp = defaultdict(set)
        g_sp.update({l: set((t,)),
                     t: set((l,))})

        g_ab = defaultdict(set)
        g_ab.update({l: set((c,)),
                     c: set((l,))})

        g = defaultdict(set)
        g.update({l: set((t, c)),
                  t: set((l,)),
                  c: set((l,))})

        ureg.enable_contexts('lc')
        self.assertEqual(ureg._active_ctx.graph, g_sp)
        ureg.disable_contexts(1)

        ureg.enable_contexts('lc', n=1)
        self.assertEqual(ureg._active_ctx.graph, g_sp)
        ureg.disable_contexts(1)

        ureg.enable_contexts('ab')
        self.assertEqual(ureg._active_ctx.graph, g_ab)
        ureg.disable_contexts(1)

        ureg.enable_contexts('lc')
        ureg.enable_contexts('ab')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('ab')
        ureg.enable_contexts('lc')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('lc', 'ab')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('ab', 'lc')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

    def test_known_nested_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context('lc'):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx.graph)
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx.graph)

            with ureg.context('ab'):
                self.assertTrue(ureg._active_ctx)
                self.assertTrue(ureg._active_ctx.graph)
                self.assertNotEqual(x, ureg._active_ctx)
                self.assertNotEqual(y, ureg._active_ctx.graph)

            self.assertEqual(x, ureg._active_ctx)
            self.assertEqual(y, ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_unknown_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        try:
            with ureg.context('la'):
                pass
        except KeyError as e:
            value = True
        except Exception as e:
            value = False
        self.assertTrue(value)
        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_unknown_nested_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context('lc'):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx.graph)
            try:
                with ureg.context('la'):
                    pass
            except KeyError as e:
                value = True
            except Exception as e:
                value = False

            self.assertTrue(value)

            self.assertEqual(x, ureg._active_ctx)
            self.assertEqual(y, ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)


    def test_one_context(self):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        meter_units = ureg.get_compatible_units(ureg.meter)
        hertz_units = ureg.get_compatible_units(ureg.hertz)

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc'):
            self.assertEqual(q.to('Hz'), s)
            self.assertEqual(ureg.get_compatible_units(q), meter_units | hertz_units)
        self.assertRaises(ValueError, q.to, 'Hz')
        self.assertEqual(ureg.get_compatible_units(q), meter_units)


    def test_multiple_context(self):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        meter_units = ureg.get_compatible_units(ureg.meter)
        hertz_units = ureg.get_compatible_units(ureg.hertz)
        ampere_units = ureg.get_compatible_units(ureg.ampere)

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc', 'ab'):
            self.assertEqual(q.to('Hz'), s)
            self.assertEqual(ureg.get_compatible_units(q), meter_units | hertz_units | ampere_units)
        self.assertRaises(ValueError, q.to, 'Hz')
        self.assertEqual(ureg.get_compatible_units(q), meter_units)


    def test_nested_context(self):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('lc'):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

    def test_context_with_arg(self):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc', n=1):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('lc', n=1):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

        with ureg.context('lc'):
            self.assertRaises(TypeError, q.to, 'Hz')

    def test_enable_context_with_arg(self):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.enable_contexts('lc', n=1)
        self.assertEqual(q.to('Hz'), s)
        ureg.enable_contexts('ab')
        self.assertEqual(q.to('Hz'), s)
        self.assertEqual(q.to('Hz'), s)
        ureg.disable_contexts(1)
        ureg.disable_contexts(1)

        ureg.enable_contexts('ab')
        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.enable_contexts('lc', n=1)
        self.assertEqual(q.to('Hz'), s)
        ureg.disable_contexts(1)
        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.disable_contexts(1)

        ureg.enable_contexts('lc')
        self.assertRaises(TypeError, q.to, 'Hz')
        ureg.disable_contexts(1)


    def test_context_with_arg_def(self):

        ureg = UnitRegistry()

        add_argdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('lc'):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('lc', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s / 2)
            self.assertEqual(q.to('Hz'), s / 2)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('lc', n=2):
                self.assertEqual(q.to('Hz'), s / 2)
            self.assertRaises(ValueError, q.to, 'Hz')


    def test_context_with_sharedarg_def(self):

        ureg = UnitRegistry()

        add_sharedargdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')
        u = (1 / 500) * ureg.ampere

        with ureg.context('lc'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('ampere'), u)

        with ureg.context('ab'):
            self.assertEqual(q.to('ampere'), 0 * u)
            with ureg.context('lc'):
                self.assertRaises(ZeroDivisionError, ureg.Quantity.to, q, 'Hz')

        with ureg.context('lc', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab'):
                self.assertEqual(q.to('ampere'), 2 * u)

        with ureg.context('ab', n=3):
            self.assertEqual(q.to('ampere'), 3 * u)
            with ureg.context('lc'):
                self.assertEqual(q.to('Hz'), s / 3)

        with ureg.context('lc', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab', n=4):
                self.assertEqual(q.to('ampere'), 4 * u)

        with ureg.context('ab', n=3):
            self.assertEqual(q.to('ampere'), 3 * u)
            with ureg.context('lc', n=6):
                self.assertEqual(q.to('Hz'), s / 6)

    def _test_ctx(self, ctx):
        ureg = UnitRegistry()
        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        nctx = len(ureg._contexts)

        self.assertNotIn(ctx.name, ureg._contexts)
        ureg.add_context(ctx)

        self.assertIn(ctx.name, ureg._contexts)
        self.assertEqual(len(ureg._contexts), nctx + 1 + len(ctx.aliases))

        with ureg.context(ctx.name):
            self.assertEqual(q.to('Hz'), s)
            self.assertEqual(s.to('meter'), q)

        ureg.remove_context(ctx.name)
        self.assertNotIn(ctx.name, ureg._contexts)
        self.assertEqual(len(ureg._contexts), nctx)

    def test_parse_invalid(self):
        s = ['@context longcontextname',
             '[length] = 1 / [time]: c / value',
             '1 / [time] = [length]: c / value']

        self.assertRaises(ValueError, Context.from_lines, s)

    def test_parse_simple(self):

        a = Context.__keytransform__(UnitsContainer({'[time]': -1}), UnitsContainer({'[length]': 1}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1}))

        s = ['@context longcontextname',
             '[length] -> 1 / [time]: c / value',
             '1 / [time] -> [length]: c / value']

        c = Context.from_lines(s)
        self.assertEqual(c.name, 'longcontextname')
        self.assertEqual(c.aliases, ())
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

        s = ['@context longcontextname = lc',
             '[length] <-> 1 / [time]: c / value']

        c = Context.from_lines(s)
        self.assertEqual(c.name, 'longcontextname')
        self.assertEqual(c.aliases, ('lc', ))
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

        s = ['@context longcontextname = lc = lcn',
             '[length] <-> 1 / [time]: c / value']

        c = Context.from_lines(s)
        self.assertEqual(c.name, 'longcontextname')
        self.assertEqual(c.aliases, ('lc', 'lcn', ))
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

    def test_parse_auto_inverse(self):

        a = Context.__keytransform__(UnitsContainer({'[time]': -1.}), UnitsContainer({'[length]': 1.}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1.}), UnitsContainer({'[time]': -1.}))

        s = ['@context longcontextname',
             '[length] <-> 1 / [time]: c / value']

        c = Context.from_lines(s)
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

    def test_parse_define(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': -1}), UnitsContainer({'[length]': 1.}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1.}))

        s = ['@context longcontextname',
             '[length] <-> 1 / [time]: c / value']
        c = Context.from_lines(s)
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

    def test_parse_parameterized(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': -1.}), UnitsContainer({'[length]': 1.}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1.}), UnitsContainer({'[time]': -1.}))

        s = ['@context(n=1) longcontextname',
             '[length] <-> 1 / [time]: n * c / value']

        c = Context.from_lines(s)
        self.assertEqual(c.defaults, {'n': 1})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))
        self._test_ctx(c)

        s = ['@context(n=1, bla=2) longcontextname',
             '[length] <-> 1 / [time]: n * c / value / bla']

        c = Context.from_lines(s)
        self.assertEqual(c.defaults, {'n': 1, 'bla': 2})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

        # If the variable is not present in the definition, then raise an error
        s = ['@context(n=1) longcontextname',
             '[length] <-> 1 / [time]: c / value']
        self.assertRaises(ValueError, Context.from_lines, s)

    def test_warnings(self):

        ureg = UnitRegistry()

        with self.capture_log() as buffer:
            add_ctxs(ureg)

            d = Context('ab')
            ureg.add_context(d)

            self.assertEqual(len(buffer), 1)
            self.assertIn("ab", str(buffer[-1]))

            d = Context('ab1', aliases=('ab',))
            ureg.add_context(d)

            self.assertEqual(len(buffer), 2)
            self.assertIn("ab", str(buffer[-1]))


class TestDefinedContexts(QuantityTestCase):

    FORCE_NDARRAY = False

    def test_defined(self):
        ureg = self.ureg
        with ureg.context('sp'):
            pass

        a = Context.__keytransform__(UnitsContainer({'[time]': -1.}), UnitsContainer({'[length]': 1.}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1.}), UnitsContainer({'[time]': -1.}))
        self.assertIn(a, ureg._contexts['sp'].funcs)
        self.assertIn(b, ureg._contexts['sp'].funcs)
        with ureg.context('sp'):
            self.assertIn(a, ureg._active_ctx)
            self.assertIn(b, ureg._active_ctx)

    def test_spectroscopy(self):
        ureg = self.ureg
        eq = (532. * ureg.nm, 563.5 * ureg.terahertz, 2.33053 * ureg.eV)
        with ureg.context('sp'):
            from pint.util import find_shortest_path
            for a, b in itertools.product(eq, eq):
                for x in range(2):
                    if x == 1:
                        a = a.to_base_units()
                        b = b.to_base_units()
                    da, db = Context.__keytransform__(a.dimensionality,
                                                      b.dimensionality)
                    p = find_shortest_path(ureg._active_ctx.graph, da, db)
                    self.assertTrue(p)
                    msg = '{} <-> {}'.format(a, b)
                    # assertAlmostEqualRelError converts second to first
                    self.assertQuantityAlmostEqual(b, a, rtol=0.01, msg=msg)


        for a, b in itertools.product(eq, eq):
            self.assertQuantityAlmostEqual(a.to(b.units, 'sp'), b, rtol=0.01)

    def test_textile(self):
        ureg = self.ureg
        qty_direct = 1.331 * ureg.tex
        with self.assertRaises(errors.DimensionalityError):
            qty_indirect = qty_direct.to('Nm')

        with ureg.context('textile'):
            from pint.util import find_shortest_path
            qty_indirect = qty_direct.to('Nm')
            a = qty_direct.to_base_units()
            b = qty_indirect.to_base_units()
            da, db = Context.__keytransform__(a.dimensionality,
                                              b.dimensionality)
            p = find_shortest_path(ureg._active_ctx.graph, da, db)
            self.assertTrue(p)
            msg = '{} <-> {}'.format(a, b)
            self.assertQuantityAlmostEqual(b, a, rtol=0.01, msg=msg)

    def test_decorator(self):
        ureg = self.ureg

        a = 532. * ureg.nm
        with ureg.context('sp'):
            b = a.to('terahertz')

        def f(wl):
            return wl.to('terahertz')

        self.assertRaises(errors.DimensionalityError, f, a)

        @ureg.with_context('sp')
        def g(wl):
            return wl.to('terahertz')

        self.assertEqual(b, g(a))

    def test_decorator_composition(self):
        ureg = self.ureg

        a = 532. * ureg.nm
        with ureg.context('sp'):
            b = a.to('terahertz')

        @ureg.with_context('sp')
        @ureg.check('[length]')
        def f(wl):
            return wl.to('terahertz')

        @ureg.with_context('sp')
        @ureg.check('[length]')
        def g(wl):
            return wl.to('terahertz')

        self.assertEqual(b, f(a))
        self.assertEqual(b, g(a))
