# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import sys

import itertools

PYTHON3 = sys.version >= '3'


from pint import pi_theorem

from tests import TestCase, string_types, u

class TestPiTheorem(TestCase):

    FORCE_NDARRAY = False

    def test_simple(self):

        # simple movement
        self.assertEqual(pi_theorem({'V': 'm/s', 'T': 's', 'L': 'm'}), [{'V': 1.0, 'T': 1.0, 'L': -1.0}])

        # pendulum
        self.assertEqual(pi_theorem({'T': 's', 'M': 'grams', 'L': 'm', 'g': 'm/s**2'}), [{'g': 1.0, 'T': 2.0, 'L': -1}])

    def test_inputs(self):
        V = 'km/hour'
        T = 'ms'
        L = 'cm'

        f1 = lambda x: x
        f2 = lambda x: self.Q_(1, x)
        f3 = lambda x: self.Q_(1, x).units
        f4 = lambda x: self.Q_(1, x).dimensionality

        fs = f1, f2, f3, f4
        for fv, ft, fl in itertools.product(fs, fs, fs):
            qv = fv(V)
            qt = ft(T)
            ql = ft(L)
            self.assertEqual(pi_theorem({'V': qv, 'T': qt, 'L': ql}), [{'V': 1.0, 'T': 1.0, 'L': -1.0}])
