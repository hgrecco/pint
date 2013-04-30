# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

from pint.util import string_preprocessor


class TestStringProcessor(unittest.TestCase):

    def _test(self, bef, aft):
        for pattern in ('{}', '+{}+'):
            b = pattern.format(bef)
            a = pattern.format(aft)
            self.assertEqual(string_preprocessor(b), a)

    def test_rules(self):
        self._test('bcd^3', 'bcd**3')
        self._test('bcd squared', 'bcd**2')
        self._test('bcd cubed', 'bcd**3')
        self._test('sq bcd', 'bcd**2')
        self._test('square bcd', 'bcd**2')
        self._test('cubic bcd', 'bcd**3')
        self._test('bcd efg', 'bcd*efg')
        self._test('bcd  efg', 'bcd*efg')
        self._test('miles per hour', 'miles/hour')
        self._test('1,234,567', '1234567')
        self._test('1hour', '1*hour')
        self._test('1.1hour', '1.1*hour')
        self._test('1e-24', '1e-24')
        self._test('1e+24', '1e+24')
        self._test('1e24', '1e24')
        self._test('1E-24', '1E-24')
        self._test('1E+24', '1E+24')
        self._test('1E24', '1E24')
        self._test('g_0', 'g_0')
        self._test('1g_0', '1*g_0')
        self._test('g0', 'g0')
        self._test('1g0', '1*g0')
        self._test('g', 'g')
        self._test('1g', '1*g')
