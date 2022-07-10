import unittest
from math import sqrt

from pint.testsuite import QuantityTestCase, helpers

from ..polynomial import Polynomial


@helpers.requires_numpy
class TestPolynomial(QuantityTestCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        polys: list[Polynomial] = [
            Polynomial([0, 1], cls.ureg.ft, cls.ureg.s),
            Polynomial([5, -5], cls.ureg.ft, cls.ureg.s),
            Polynomial([0, 0, 2], cls.ureg.gal, cls.ureg.psi),
            Polynomial([1, -3, 5], cls.ureg.ft, cls.ureg.s),
            Polynomial([0, 0, 0, 1], cls.ureg.ft, cls.ureg.s),
            Polynomial([3, -5, 9, 0.1], cls.ureg.ft, cls.ureg.s),
            Polynomial([0, 0, 0, 0, 0, 1], cls.ureg.gal, cls.ureg.psi),
            Polynomial([9, 2, 4, 1, 2], cls.ureg.ft, cls.ureg.s),
        ]
        cls.polys = polys
        cls.linear_1, cls.linear_2 = polys[0], polys[1]
        cls.parabolic_1, cls.parabolic_2 = polys[2], polys[3]
        cls.cubic_1, cls.cubic_2 = polys[4], polys[5]
        cls.polynomial_1, cls.polynomial_2 = polys[6], polys[7]

    def test_polynomial_addition(self):

        self.assertTrue(
            all([a == b for a, b in zip((self.linear_1 + 5).coef, [5.0, 1.0])])
        )
        self.assertTrue(
            all([a == b for a, b in zip((self.cubic_1 + 9.5).coef, [9.5, 0, 0, 1])])
        )
        self.assertTrue(
            all([a == b for a, b in zip((9.5 + self.cubic_1).coef, [9.5, 0, 0, 1])])
        )
        self.assertRaises(TypeError, self.linear_1.__add__, self.parabolic_1)
        self.assertTrue(
            all(
                [
                    a == b
                    for a, b in zip(
                        (self.cubic_2 + self.polynomial_2.coef).coef,
                        [12, -3, 13, 1.1, 2],
                    )
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    a == b
                    for a, b in zip(
                        (self.cubic_2 + self.polynomial_2).coef, [12, -3, 13, 1.1, 2]
                    )
                ]
            )
        )

    def test_polynomial_subtraction(self):
        self.assertTrue(
            all([a == b for a, b in zip((self.linear_1 - 5).coef, [-5, 1])])
        )
        self.assertTrue(
            all([a == b for a, b in zip((self.cubic_1 - 9.5).coef, [-9.5, 0, 0, 1])])
        )
        self.assertTrue(
            all([a == b for a, b in zip((9.5 - self.cubic_1).coef, [9.5, 0, 0, -1])])
        )

        self.assertRaises(TypeError, self.linear_1.__sub__, self.parabolic_1)
        self.assertTrue(
            all(
                [
                    a == b
                    for a, b in zip(
                        (self.cubic_2 - self.polynomial_2.coef).coef,
                        [-6, -7, 5, -0.9, -2],
                    )
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    a == b
                    for a, b in zip(
                        (self.cubic_2 - self.polynomial_2).coef, [-6, -7, 5, -0.9, -2]
                    )
                ]
            )
        )

    def test_correct_units_after_math(self):
        self.assertEqual((self.linear_1 + 1).x_unit, self.linear_1.x_unit)
        self.assertEqual((self.linear_1 - 1).x_unit, self.linear_1.x_unit)
        self.assertEqual((self.linear_1 * 1).x_unit, self.linear_1.x_unit)
        self.assertEqual((self.polynomial_1 / 2).x_unit, self.polynomial_1.x_unit)

        self.assertEqual((1 + self.linear_1).y_unit, self.linear_1.y_unit)
        self.assertEqual((1 - self.linear_1).y_unit, self.linear_1.y_unit)
        self.assertEqual((1 * self.linear_1).y_unit, self.linear_1.y_unit)
        # self.assertEqual((1 / self.linear_1).y_unit, self.linear_1.y_unit ** -1)

        self.assertEqual(
            (self.parabolic_1 + self.polynomial_1).x_unit, self.parabolic_1.x_unit
        )
        self.assertEqual((self.linear_1 - self.linear_2).y_unit, self.linear_1.y_unit)
        self.assertEqual(
            (self.linear_1 * self.parabolic_1).x_unit,
            self.linear_1.x_unit * self.parabolic_1.x_unit,
        )
        self.assertEqual(
            (self.linear_1 / self.parabolic_1).y_unit,
            self.linear_1.y_unit / self.parabolic_1.y_unit,
        )

    def test_derivative_units(self):
        for p in self.polys:
            self.assertEqual(p.derivative.y_unit, p.y_unit / p.x_unit)
            self.assertEqual(p.derivative.x_unit, p.x_unit)

    def test_integral_units(self):
        for p in self.polys:
            self.assertEqual(p.integral.y_unit, p.y_unit * p.x_unit)
            self.assertEqual(p.integral.x_unit, p.x_unit)

    def test_poly_iter(self):
        ys = [1, 1, 1, 1, 1, 5, 5, 10]
        xs = [
            1,
            0.8,
            sqrt(0.5),
            [0, 0.6],
            1,
            0.819323306231648,
            1.379729661461215,
            0.29934454191522236,
        ]
        for p, y, x in zip(self.polys, ys, xs):
            try:
                solution = p.solve_for_x(y * p.y_unit, min_value=0).m_as(p.x_unit)
                if isinstance(solution, float):
                    self.assertAlmostEqual(solution, x)
                else:
                    self.assertListEqual(list(solution), x)
            except (AssertionError, ValueError) as e:
                print(p, solution, x * p.x_unit)
                raise e

    def test_solve_derivative(self):
        # y_intercept of the derivative should always be the second coefficient
        for p in self.polys:
            self.assertEqual(p.derivative_at(0 * p.x_unit).m, p.coef[1])

    def test_derivatives(self):
        # y_intercept of the derivative should always be the second coefficient
        for p in self.polys:
            self.assertEqual(p.derivative.y_intercept.m, p.coef[1])

    def test_y_intercept(self):
        # y_intercept should always be the first coefficient
        for p in self.polys:
            self.assertEqual(p.y_intercept.m, p.coef[0])

    def test_x_intercept(self):
        intercepts = [0, 1, 0, None, 0, -90.55580410259788, 0, None]
        for p, intercept in zip(self.polys, intercepts):
            if intercept is None:
                self.assertEqual(p.x_intercept, None)
            else:
                self.assertEqual(p.x_intercept, intercept * p.x_unit)

    def test_solve_linear(self):
        self.assertEqual(
            self.linear_1.solve(-3 * self.linear_1.x_unit), -3 * self.linear_1.y_unit
        )
        self.assertEqual(
            self.linear_1.solve(0 * self.linear_1.x_unit), self.linear_1.y_intercept
        )
        self.assertEqual(
            self.linear_1.solve(10 * self.linear_1.x_unit), 10 * self.linear_1.y_unit
        )

        self.assertEqual(
            self.linear_2.solve(-3 * self.linear_2.x_unit), 20 * self.linear_2.y_unit
        )
        self.assertEqual(
            self.linear_2.solve(0.0 * self.linear_2.x_unit), self.linear_2.y_intercept
        )
        self.assertEqual(
            self.linear_2.solve(10 * self.linear_2.x_unit), -45 * self.linear_2.y_unit
        )

    def test_solve_parabolic(self):
        self.assertEqual(
            self.parabolic_1.solve(-3 * self.parabolic_1.x_unit),
            18 * self.parabolic_1.y_unit,
        )
        self.assertEqual(
            self.parabolic_1.solve(0 * self.parabolic_1.x_unit),
            self.parabolic_1.y_intercept,
        )
        self.assertEqual(
            self.parabolic_1.solve(-10 * self.parabolic_1.x_unit),
            200 * self.parabolic_1.y_unit,
        )

        self.assertEqual(
            self.parabolic_2.solve(-3 * self.parabolic_2.x_unit),
            55 * self.parabolic_2.y_unit,
        )
        self.assertEqual(
            self.parabolic_2.solve(0 * self.parabolic_2.x_unit),
            self.parabolic_2.y_intercept,
        )
        self.assertEqual(
            self.parabolic_2.solve(-10 * self.parabolic_2.x_unit),
            531 * self.parabolic_2.y_unit,
        )
