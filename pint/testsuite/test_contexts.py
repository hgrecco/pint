# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

from pint import UnitRegistry
from pint.unit import UnitsContainer, _freeze


def add_ctxs(ureg):
    a, b = _freeze(UnitsContainer({'[length]': 1})), _freeze(UnitsContainer({'[time]': -1}))
    d = {}
    d[(a, b)] = lambda x: ureg.speed_of_light / x
    d[(b, a)] = lambda x: ureg.speed_of_light / x

    ureg._contexts['sp'] = d

    a, b = _freeze(UnitsContainer({'[length]': 1})), _freeze(UnitsContainer({'[current]': -1}))
    d = {}
    d[(a, b)] = lambda x: 1 / x
    d[(b, a)] = lambda x: 1 / x

    ureg._contexts['ab'] = d


class TestContexts(unittest.TestCase):

    def test_known_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        with ureg.context('sp'):
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx_graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx_graph)

    def test_known_nested_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context('sp'):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx_graph)
            self.assertTrue(ureg._active_ctx)
            self.assertTrue(ureg._active_ctx_graph)

            with ureg.context('ab'):
                self.assertTrue(ureg._active_ctx)
                self.assertTrue(ureg._active_ctx_graph)
                self.assertNotEqual(x, ureg._active_ctx)
                self.assertNotEqual(y, ureg._active_ctx_graph)

            self.assertEqual(x, ureg._active_ctx)
            self.assertEqual(y, ureg._active_ctx_graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx_graph)

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
        self.assertFalse(ureg._active_ctx_graph)

    def test_unknown_nested_context(self):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context('sp'):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx_graph)
            try:
                with ureg.context('la'):
                    pass
            except KeyError as e:
                value = True
            except Exception as e:
                value = False

            self.assertTrue(value)

            self.assertEqual(x, ureg._active_ctx)
            self.assertEqual(y, ureg._active_ctx_graph)

        self.assertFalse(ureg._active_ctx)
        self.assertFalse(ureg._active_ctx_graph)


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
