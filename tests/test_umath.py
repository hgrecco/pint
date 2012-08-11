# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

import numpy as np

from pint import UnitRegistry, UnitsContainer

from tests import TestCase

class TestUmath(TestCase):

    FORCE_NDARRAY = True

    @property
    def q(self):
        return [1,2,3,4] * self.ureg.J

    def test_prod(self):
        self.assertEqual(np.prod(self.q), 24 * self.ureg.J**4)

    def test_sum(self):
        self.assertEqual(np.sum(self.q), 10 * self.ureg.J)

    def test_nansum(self):
        c = [1,2,3, np.NaN] * self.ureg.m
        self.assertEqual(np.nansum(c), 6 * self.ureg.m)

    def test_cumprod(self):
        self.assertRaises(ValueError, np.cumprod, self.q)

        q = [10, .1, 5, 50] * self.ureg.dimensionless
        self.assertEqual(np.cumprod(q).tolist(), [10, 1, 5, 250])

    def test_cumsum(self):
        self.assertEqual(np.cumsum(self.q), [1, 3, 6, 10] * self.ureg.J)

    def test_diff(self):
        self.assertEqual(np.diff(self.q, 1), [1, 1, 1] * self.ureg.J)

    def test_ediff1d(self):
        self.assertEqual(np.ediff1d(self.q, 1), [1, 1, 1] * self.ureg.J)

    def test_gradient(self):
        try:
            l = np.gradient([[1,1],[3,4]] * self.ureg.J, 1 * self.ureg.m)
            self.assertEqual(l[0], [[2., 3.], [2., 3.]] * self.ureg.J / self.ureg.m)
            self.assertEqual(l[1], [[0., 0.], [1., 1.]] * self.ureg.J / self.ureg.m)
        except ValueError as e:
            raise self.failureException(e)

    def test_cross(self):
        a = [3,-3, 1] * self.ureg.kPa
        b = [4, 9, 2] * self.ureg.m**2
        self.assertEqual(np.cross(a,b), [-15,-2,39]*self.ureg.kPa*self.ureg.m**2)

    def test_trapz(self):
        self.assertEqual(np.trapz(self.q, dx = 1*self.ureg.m), 7.5 * self.ureg.J*self.ureg.m)

    def test_sinh(self):
        q = [1, 2, 3, 4, 6] * self.ureg.radian
        self.assertEqual(
            np.sinh(q),
            np.sinh(q.magnitude)
        )

    def test_arcsinh(self):
        q = [1, 2, 3, 4, 6] * self.ureg.dimensionless
        self.assertEqual(
            np.arcsinh(q),
            np.arcsinh(q.magnitude) * self.ureg.rad
        )

    def test_cosh(self):
        q = [1, 2, 3, 4, 6] * self.ureg.radian
        self.assertEqual(
            np.cosh(q),
            np.cosh(q.magnitude) * self.ureg.dimensionless
        )

    def test_arccosh(self):
        q = [1, 2, 3, 4, 6] * self.ureg.dimensionless
        x = np.ones((1,1))  * self.ureg.rad
        self.assertEqual(
            np.arccosh(q),
            np.arccosh(q.magnitude * self.ureg.rad)
        )

    def test_tanh(self):
        q = [1, 2, 3, 4, 6] * self.ureg.rad
        self.assertEqual(
            np.tanh(q),
            np.tanh(q.magnitude)
        )

    def test_arctanh(self):
        q = [.01, .5, .6, .8, .99] * self.ureg.dimensionless
        self.assertEqual(
            np.arctanh(q),
            np.arctanh(q.magnitude) * self.ureg.rad
        )

    def test_around(self):
        self.assertEqual(
            np.around([.5, 1.5, 2.5, 3.5, 4.5] * self.ureg.J) ,
            [0., 2., 2., 4., 4.] * self.ureg.J
        )

        self.assertEqual(
            np.around([1,2,3,11] * self.ureg.J, decimals=1),
            [1, 2, 3, 11] * self.ureg.J
        )

        self.assertEqual(
            np.around([1,2,3,11] * self.ureg.J, decimals=-1),
            [0, 0, 0, 10] * self.ureg.J
        )

    def test_round_(self):
        self.assertEqual(
            np.round_([.5, 1.5, 2.5, 3.5, 4.5] * self.ureg.J),
            [0., 2., 2., 4., 4.] * self.ureg.J
        )

        self.assertEqual(
            np.round_([1,2,3,11] * self.ureg.J, decimals=1),
            [1, 2, 3, 11] * self.ureg.J
        )

        self.assertEqual(
            np.round_([1,2,3,11] * self.ureg.J, decimals=-1),
            [0, 0, 0, 10] * self.ureg.J
        )

    def test_rint(self):
        a = [-4.1, -3.6, -2.5, 0.1, 2.5, 3.1, 3.9] * self.ureg.m
        self.assertEqual(
            np.rint(a),
            [-4., -4., -2., 0., 2., 3., 4.]*self.ureg.m
        )

    def test_floor(self):
        a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * self.ureg.m
        self.assertEqual(
            np.floor(a),
            [-2., -2., -1., 0., 1., 1., 2.] * self.ureg.m
        )

    def test_ceil(self):
        a = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0] * self.ureg.m
        self.assertEqual(
            np.ceil(a),
            [-1., -1., -0., 1., 2., 2., 2.] * self.ureg.m
        )

    def test_fix(self):
        try:
            self.assertEqual(np.fix(3.14 * self.ureg.degF), 3.0 * self.ureg.degF)
            self.assertEqual(np.fix(3.0 * self.ureg.degF), 3.0 * self.ureg.degF)
            self.assertEqual(
                np.fix([2.1, 2.9, -2.1, -2.9] * self.ureg.degF),
                [2., 2., -2., -2.] * self.ureg.degF
            )
        except ValueError as e:
            raise self.failureException(e)

    def test_exp(self):
        self.assertEqual(np.exp(1*self.ureg.dimensionless), np.e)
        self.assertRaises(ValueError, np.exp, 1*self.ureg.m)

    def test_log(self):
        self.assertEqual(np.log(1*self.ureg.dimensionless), 0)
        self.assertRaises(ValueError, np.log, 1*self.ureg.m)

    def test_log10(self):
        self.assertEqual(np.log10(1*self.ureg.dimensionless), 0)
        self.assertRaises(ValueError, np.log10, 1*self.ureg.m)

    def test_log2(self):
        self.assertEqual(np.log2(1*self.ureg.dimensionless), 0)
        self.assertRaises(ValueError, np.log2, 1*self.ureg.m)

    def test_expm1(self):
        self.assertAlmostEqual(np.expm1(1*self.ureg.dimensionless), np.e-1, delta=1e-6)
        self.assertRaises(ValueError, np.expm1, 1*self.ureg.m)

    def test_log1p(self):
        self.assertEqual(np.log1p(0*self.ureg.dimensionless), 0)
        self.assertRaises(ValueError, np.log1p, 1*self.ureg.m)

    def test_sin(self):
        self.assertEqual(np.sin(np.pi/2*self.ureg.radian), 1)
        self.assertRaises(ValueError, np.sin, 1*self.ureg.m)

    def test_arcsin(self):
        self.assertEqual(
            np.arcsin(1*self.ureg.dimensionless),
            np.pi/2 * self.ureg.radian
        )
        self.assertRaises(ValueError, np.arcsin, 1*self.ureg.m)

    def test_cos(self):
        self.assertEqual(np.cos(np.pi*self.ureg.radians), -1)
        self.assertRaises(ValueError, np.cos, 1*self.ureg.m)

    def test_arccos(self):
        self.assertEqual(np.arccos(1*self.ureg.dimensionless), 0*self.ureg.radian)
        self.assertRaises(ValueError, np.arccos, 1*self.ureg.m)

    def test_tan(self):
        self.assertEqual(np.tan(0*self.ureg.radian), 0)
        self.assertRaises(ValueError, np.tan, 1*self.ureg.m)

    def test_arctan(self):
        self.assertEqual(np.arctan(0*self.ureg.dimensionless), 0*self.ureg.radian)
        self.assertRaises(ValueError, np.arctan, 1*self.ureg.m)

    def test_arctan2(self):
        self.assertEqual(
            np.arctan2(0*self.ureg.dimensionless, 0*self.ureg.dimensionless),
            0
        )
        self.assertRaises(ValueError, np.arctan2, (1*self.ureg.m, 1*self.ureg.m))

    def test_hypot(self):
        self.assertEqual(np.hypot(3 * self.ureg.m, 4 * self.ureg.m),  5 * self.ureg.m)
        self.assertRaises(ValueError, np.hypot, 1*self.ureg.m, 2*self.ureg.J)

    def test_degrees(self):
        self.assertAlmostEqual(
            np.degrees(6. * self.ureg.radians),
            (6. * self.ureg.radians).to(self.ureg.degree)
        )
        self.assertRaises(ValueError, np.degrees, 0.*self.ureg.m)

    def test_radians(self):
        (6. * self.ureg.degree).to(self.ureg.radian)
        self.assertAlmostEqual(
            np.radians(6. * self.ureg.degree),
            (6. * self.ureg.degree).to(self.ureg.radian)
        )
        self.assertRaises(ValueError, np.radians, 0*self.ureg.m)

    def test_unwrap(self):
        self.assertEqual(np.unwrap([0,3*np.pi]*self.ureg.radians), [0,np.pi])
        self.assertEqual(np.unwrap([0,540]*self.ureg.deg), [0,180]*self.ureg.deg)
