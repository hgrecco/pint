# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest
import collections

from pint.util import string_preprocessor, find_shortest_path


class TestStringProcessor(unittest.TestCase):

    def _test(self, bef, aft):
        for pattern in ('{}', '+{}+'):
            b = pattern.format(bef)
            a = pattern.format(aft)
            self.assertEqual(string_preprocessor(b), a)

    def test_square_cube(self):
        self._test('bcd^3', 'bcd**3')
        self._test('bcd^ 3', 'bcd** 3')
        self._test('bcd ^3', 'bcd **3')
        self._test('bcd squared', 'bcd**2')
        self._test('bcd squared', 'bcd**2')
        self._test('bcd cubed', 'bcd**3')
        self._test('sq bcd', 'bcd**2')
        self._test('square bcd', 'bcd**2')
        self._test('cubic bcd', 'bcd**3')
        self._test('bcd efg', 'bcd*efg')
        
    def test_per(self):
        self._test('miles per hour', 'miles/hour')

    def test_numbers(self):
        self._test('1,234,567', '1234567')
        self._test('1e-24', '1e-24')
        self._test('1e+24', '1e+24')
        self._test('1e24', '1e24')
        self._test('1E-24', '1E-24')
        self._test('1E+24', '1E+24')
        self._test('1E24', '1E24')

    def test_space_multiplication(self):
        self._test('bcd efg', 'bcd*efg')
        self._test('bcd  efg', 'bcd*efg')
        self._test('1 hour', '1*hour')
        self._test('1. hour', '1.*hour')
        self._test('1.1 hour', '1.1*hour')
        self._test('1E24 hour', '1E24*hour')
        self._test('1E-24 hour', '1E-24*hour')
        self._test('1E+24 hour', '1E+24*hour')
        self._test('1.2E24 hour', '1.2E24*hour')
        self._test('1.2E-24 hour', '1.2E-24*hour')
        self._test('1.2E+24 hour', '1.2E+24*hour')

    def names(self):
        self._test('g_0', 'g_0')
        self._test('g0', 'g0')
        self._test('g', 'g')
        self._test('water_60F', 'water_60F')


class TestGraph(unittest.TestCase):

    def test_shortest_path(self):
        g = collections.defaultdict(list)
        g[1] = {2,}
        g[2] = {3,}
        p = find_shortest_path(g, 1, 2)
        self.assertEqual(p, [1, 2])
        p = find_shortest_path(g, 1, 3)
        self.assertEqual(p, [1, 2, 3])
        p = find_shortest_path(g, 3, 1)
        self.assertIs(p, None)

        g = collections.defaultdict(list)
        g[1] = {2,}
        g[2] = {3, 1}
        g[3] = {2, }
        p = find_shortest_path(g, 1, 2)
        self.assertEqual(p, [1, 2])
        p = find_shortest_path(g, 1, 3)
        self.assertEqual(p, [1, 2, 3])
        p = find_shortest_path(g, 3, 1)
        self.assertEqual(p, [3, 2, 1])
        p = find_shortest_path(g, 2, 1)
        self.assertEqual(p, [2, 1])
