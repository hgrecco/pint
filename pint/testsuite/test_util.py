# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import collections

from pint.testsuite import BaseTestCase
from pint.util import (string_preprocessor, find_shortest_path, matrix_to_string,
                       transpose, find_connected_nodes, ParserHelper)


class TestParseHelper(BaseTestCase):

    def test_basic(self):
        # Parse Helper ar mutables, so we build one everytime
        x = lambda: ParserHelper(1, meter=2)
        xp = lambda: ParserHelper(1, meter=2)
        y = lambda: ParserHelper(2, meter=2)

        self.assertEqual(x(), xp())
        self.assertNotEqual(x(), y())
        self.assertEqual(ParserHelper.from_string(''), ParserHelper())
        self.assertEqual(repr(x()), "<ParserHelper(1, {'meter': 2})>")

        self.assertEqual(ParserHelper(2), 2)

        self.assertEqual(x(), dict(meter=2))
        self.assertEqual(x(), 'meter ** 2')
        self.assertNotEqual(y(), dict(meter=2))
        self.assertNotEqual(y(), 'meter ** 2')

        self.assertNotEqual(xp(), object())

    def test_calculate(self):
        # Parse Helper ar mutables, so we build one everytime
        x = lambda: ParserHelper(1., meter=2)
        y = lambda: ParserHelper(2., meter=-2)
        z = lambda: ParserHelper(2., meter=2)

        self.assertEqual(x() * 4., ParserHelper(4., meter=2))
        self.assertEqual(x() * y(), ParserHelper(2.))
        self.assertEqual(x() * 'second', ParserHelper(1., meter=2, second=1))

        self.assertEqual(x() / 4., ParserHelper(0.25, meter=2))
        self.assertEqual(x() / 'second', ParserHelper(1., meter=2, second=-1))
        self.assertEqual(x() / z(), ParserHelper(0.5))

        self.assertEqual(4. / z(), ParserHelper(2., meter=-2))
        self.assertEqual('seconds' / z(), ParserHelper(0.5, seconds=1, meter=-2))
        self.assertEqual(dict(seconds=1) / z(), ParserHelper(0.5, seconds=1, meter=-2))


class TestStringProcessor(BaseTestCase):

    def _test(self, bef, aft):
        for pattern in ('{0}', '+{0}+'):
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

    def test_joined_multiplication(self):
        self._test('1hour', '1*hour')
        self._test('1.hour', '1.*hour')
        self._test('1.1hour', '1.1*hour')
        self._test('1h', '1*h')
        self._test('1.h', '1.*h')
        self._test('1.1h', '1.1*h')

    def test_names(self):
        self._test('g_0', 'g_0')
        self._test('g0', 'g0')
        self._test('g', 'g')
        self._test('water_60F', 'water_60F')


class TestGraph(BaseTestCase):

    def test_start_not_in_graph(self):
        g = collections.defaultdict(list)
        g[1] = set((2,))
        g[2] = set((3,))
        self.assertIs(find_connected_nodes(g, 9), None)

    def test_shortest_path(self):
        g = collections.defaultdict(list)
        g[1] = set((2,))
        g[2] = set((3,))
        p = find_shortest_path(g, 1, 2)
        self.assertEqual(p, [1, 2])
        p = find_shortest_path(g, 1, 3)
        self.assertEqual(p, [1, 2, 3])
        p = find_shortest_path(g, 3, 1)
        self.assertIs(p, None)

        g = collections.defaultdict(list)
        g[1] = set((2,))
        g[2] = set((3, 1))
        g[3] = set((2,))
        p = find_shortest_path(g, 1, 2)
        self.assertEqual(p, [1, 2])
        p = find_shortest_path(g, 1, 3)
        self.assertEqual(p, [1, 2, 3])
        p = find_shortest_path(g, 3, 1)
        self.assertEqual(p, [3, 2, 1])
        p = find_shortest_path(g, 2, 1)
        self.assertEqual(p, [2, 1])


class TestMatrix(BaseTestCase):

    def test_matrix_to_string(self):

        self.assertEqual(matrix_to_string([[1, 2], [3, 4]],
                                          row_headers=None,
                                          col_headers=None),
                         '1\t2\n'
                         '3\t4')

        self.assertEqual(matrix_to_string([[1, 2], [3, 4]],
                                          row_headers=None,
                                          col_headers=None,
                                          fmtfun=lambda x: '{0:.2f}'.format(x)),
                         '1.00\t2.00\n'
                         '3.00\t4.00')

        self.assertEqual(matrix_to_string([[1, 2], [3, 4]],
                                          row_headers=['c', 'd'],
                                          col_headers=None),
                         'c\t1\t2\n'
                         'd\t3\t4')

        self.assertEqual(matrix_to_string([[1, 2], [3, 4]],
                                          row_headers=None,
                                          col_headers=['a', 'b']),
                         'a\tb\n'
                         '1\t2\n'
                         '3\t4')

        self.assertEqual(matrix_to_string([[1, 2], [3, 4]],
                                          row_headers=['c', 'd'],
                                          col_headers=['a', 'b']),
                         '\ta\tb\n'
                         'c\t1\t2\n'
                         'd\t3\t4')

    def test_transpose(self):

        self.assertEqual(transpose([[1, 2], [3, 4]]), [[1, 3], [2, 4]])
