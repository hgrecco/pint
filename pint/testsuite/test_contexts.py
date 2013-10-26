# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest
from collections import defaultdict

from pint import UnitRegistry
from pint.context import Context, _freeze
from pint.unit import UnitsContainer

from pint import logger

from logging.handlers import BufferingHandler

class TestHandler(BufferingHandler):
    def __init__(self):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        BufferingHandler.__init__(self, 0)

    def shouldFlush(self):
        return False

    def emit(self, record):
        self.buffer.append(record.__dict__)

th = TestHandler()
logger.addHandler(th)


def add_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('sp')
    d.add_transformation(a, b, lambda x: ureg.speed_of_light / x)
    d.add_transformation(b, a, lambda x: ureg.speed_of_light / x)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': -1})
    d = Context('ab')
    d.add_transformation(a, b, lambda x: 1 / x)
    d.add_transformation(b, a, lambda x: 1 / x)

    ureg.add_context(d)


def add_arg_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('sp')
    d.add_transformation(a, b, lambda x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': -1})
    d = Context('ab')
    d.add_transformation(a, b, lambda x: 1 / x)
    d.add_transformation(b, a, lambda x: 1 / x)

    ureg.add_context(d)


def add_argdef_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('sp', defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': -1})
    d = Context('ab')
    d.add_transformation(a, b, lambda x: 1 / x)
    d.add_transformation(b, a, lambda x: 1 / x)

    ureg.add_context(d)


def add_sharedargdef_ctxs(ureg):
    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1})
    d = Context('sp', defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({'[length]': 1}), UnitsContainer({'[current]': 1})
    d = Context('ab', defaults=dict(n=0))
    d.add_transformation(a, b, lambda x, n: ureg.ampere * ureg.meter * n / x)
    d.add_transformation(b, a, lambda x, n: ureg.ampere * ureg.meter * n / x)

    ureg.add_context(d)


class TestContexts(unittest.TestCase):

    def test_known_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        with ureg.context('sp'):
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

        with ureg.context('sp', n=1):
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx.graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_known_context_enable(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        ureg.enable_contexts('sp')
        self.assertTrue(ureg._active_ctx)
        self.assertTrue(ureg._active_ctx.graph)
        ureg.disable_contexts(1)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

        ureg.enable_contexts('sp', n=1)
        self.assertTrue(ureg._active_ctx)
        self.assertTrue(ureg._active_ctx.graph)
        ureg.disable_contexts(1)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx.graph)

    def test_graph(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = _freeze({'[length]': 1.})
        t = _freeze({'[time]': -1.})
        c = _freeze({'[current]': -1.})

        g_sp = defaultdict(set)
        g_sp.update({l: {t, },
                     t: {l, }})

        g_ab = defaultdict(set)
        g_ab.update({l: {c, },
                     c: {l, }})

        g = defaultdict(set)
        g.update({l: {t, c},
                  t: {l, },
                  c: {l, }})

        with ureg.context('sp'):
            self.assertEqual(ureg._active_ctx.graph, g_sp)

        with ureg.context('sp', n=1):
            self.assertEqual(ureg._active_ctx.graph, g_sp)

        with ureg.context('ab'):
            self.assertEqual(ureg._active_ctx.graph, g_ab)

        with ureg.context('sp'):
            with ureg.context('ab'):
                self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('ab'):
            with ureg.context('sp'):
                self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('sp', 'ab'):
            self.assertEqual(ureg._active_ctx.graph, g)

        with ureg.context('ab', 'sp'):
            self.assertEqual(ureg._active_ctx.graph, g)

    def test_graph_enable(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = _freeze({'[length]': 1.})
        t = _freeze({'[time]': -1.})
        c = _freeze({'[current]': -1.})

        g_sp = defaultdict(set)
        g_sp.update({l: {t, },
                     t: {l, }})

        g_ab = defaultdict(set)
        g_ab.update({l: {c, },
                     c: {l, }})

        g = defaultdict(set)
        g.update({l: {t, c},
                  t: {l, },
                  c: {l, }})

        ureg.enable_contexts('sp')
        self.assertEqual(ureg._active_ctx.graph, g_sp)
        ureg.disable_contexts(1)

        ureg.enable_contexts('sp', n=1)
        self.assertEqual(ureg._active_ctx.graph, g_sp)
        ureg.disable_contexts(1)

        ureg.enable_contexts('ab')
        self.assertEqual(ureg._active_ctx.graph, g_ab)
        ureg.disable_contexts(1)

        ureg.enable_contexts('sp')
        ureg.enable_contexts('ab')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('ab')
        ureg.enable_contexts('sp')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('sp', 'ab')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

        ureg.enable_contexts('ab', 'sp')
        self.assertEqual(ureg._active_ctx.graph, g)
        ureg.disable_contexts(2)

    def test_known_nested_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context('sp'):
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

        with ureg.context('sp'):
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

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp'):
            self.assertEqual(q.to('Hz'), s)
        self.assertRaises(ValueError, q.to, 'Hz')

    def test_multiple_context(self):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp', 'ab'):
            self.assertEqual(q.to('Hz'), s)
        self.assertRaises(ValueError, q.to, 'Hz')

    def test_nested_context(self):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('sp'):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

    def test_context_with_arg(self):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp', n=1):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('sp', n=1):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

        with ureg.context('sp'):
            self.assertRaises(TypeError, q.to, 'Hz')

    def test_enable_context_with_arg(self):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.enable_contexts('sp', n=1)
        self.assertEqual(q.to('Hz'), s)
        ureg.enable_contexts('ab')
        self.assertEqual(q.to('Hz'), s)
        self.assertEqual(q.to('Hz'), s)
        ureg.disable_contexts(1)
        ureg.disable_contexts(1)

        ureg.enable_contexts('ab')
        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.enable_contexts('sp', n=1)
        self.assertEqual(q.to('Hz'), s)
        ureg.disable_contexts(1)
        self.assertRaises(ValueError, q.to, 'Hz')
        ureg.disable_contexts(1)

        ureg.enable_contexts('sp')
        self.assertRaises(TypeError, q.to, 'Hz')
        ureg.disable_contexts(1)


    def test_context_with_arg_def(self):

        ureg = UnitRegistry()

        add_argdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s)
            self.assertEqual(q.to('Hz'), s)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('sp'):
                self.assertEqual(q.to('Hz'), s)
            self.assertRaises(ValueError, q.to, 'Hz')

        self.assertRaises(ValueError, q.to, 'Hz')
        with ureg.context('sp', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab'):
                self.assertEqual(q.to('Hz'), s / 2)
            self.assertEqual(q.to('Hz'), s / 2)

        with ureg.context('ab'):
            self.assertRaises(ValueError, q.to, 'Hz')
            with ureg.context('sp', n=2):
                self.assertEqual(q.to('Hz'), s / 2)
            self.assertRaises(ValueError, q.to, 'Hz')


    def test_context_with_sharedarg_def(self):

        ureg = UnitRegistry()

        add_sharedargdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to('Hz')
        u = (1 / 500) * ureg.ampere

        with ureg.context('sp'):
            self.assertEqual(q.to('Hz'), s)
            with ureg.context('ab'):
                self.assertEqual(q.to('ampere'), u)

        with ureg.context('ab'):
            self.assertEqual(q.to('ampere'), 0 * u)
            with ureg.context('sp'):
                self.assertRaises(ZeroDivisionError, ureg.Quantity.to, q, 'Hz')

        with ureg.context('sp', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab'):
                self.assertEqual(q.to('ampere'), 2 * u)

        with ureg.context('ab', n=3):
            self.assertEqual(q.to('ampere'), 3 * u)
            with ureg.context('sp'):
                self.assertEqual(q.to('Hz'), s / 3)

        with ureg.context('sp', n=2):
            self.assertEqual(q.to('Hz'), s / 2)
            with ureg.context('ab', n=4):
                self.assertEqual(q.to('ampere'), 4 * u)

        with ureg.context('ab', n=3):
            self.assertEqual(q.to('ampere'), 3 * u)
            with ureg.context('sp', n=6):
                self.assertEqual(q.to('Hz'), s / 6)

    def test_simple_from_string(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': 1}), UnitsContainer({'[length]': -1}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1}))

        s = """@context spectral
            [length] -> 1 / [time] = 1 / value
            [time] -> 1 / [length] = 1 / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.name, 'spectral')
        self.assertEqual(c.aliases, ())
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

        s = """@context spectral = sp
            [time] <-> 1 / [length] = 1 / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.name, 'spectral')
        self.assertEqual(c.aliases, ('sp', ))
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

        s = """@context spectral = sp = spe
            [time] <-> 1 / [length] = 1 / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.name, 'spectral')
        self.assertEqual(c.aliases, ('sp', 'spe', ))
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

    def test_auto_inverse_from_string(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': 1}), UnitsContainer({'[length]': -1}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1}))

        s = """@context spectral
            [time] <-> 1 / [length] = 1 / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

    def test_definedvar_from_string(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': 1}), UnitsContainer({'[length]': -1}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1}))

        s = """@context spectral
            [time] <-> 1 / [length] = 1 / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.defaults, {})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

    def test_parameterized_from_string(self):
        a = Context.__keytransform__(UnitsContainer({'[time]': 1}), UnitsContainer({'[length]': -1}))
        b = Context.__keytransform__(UnitsContainer({'[length]': 1}), UnitsContainer({'[time]': -1}))

        s = """@context(n=1) spectral
            [time] <-> 1 / [length] = n * c / value
        """
        c = Context.from_string(s)
        self.assertEqual(c.defaults, {'n': 1})
        self.assertEqual(set(c.funcs.keys()), set((a, b)))

        # If the variable is not present in the definition, then raise an error
        s = """@context(n=1) spectral
            [time] <-> 1 / [length] = c / value
        """
        self.assertRaises(ValueError, Context.from_string, s)

    def test_warnings(self):

        ureg = UnitRegistry()

        add_ctxs(ureg)

        d = Context('ab')
        ureg.add_context(d)

        self.assertEqual(len(th.buffer), 1)
        self.assertIn("ab", str(th.buffer[-1]['message']))

        d = Context('ab1', aliases=('ab',))
        ureg.add_context(d)

        self.assertEqual(len(th.buffer), 2)
        self.assertIn("ab", str(th.buffer[-1]['message']))

