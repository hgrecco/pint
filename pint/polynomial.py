"""
    pint.polynomial
    ~~~~~~~~~~~~~~

    A polynomial class inheriting from numpy.polynomial.polynomial.Polynomial incorporating the pint Quantity unit values.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from math import inf
from typing import Optional, Union

from numpy.polynomial import polynomial as p

from . import _DEFAULT_REGISTRY, Quantity, Unit


class Polynomial(p.Polynomial):
    def __init__(
        self,
        coef: list[float],
        x_unit: Unit = _DEFAULT_REGISTRY.dimensionless,
        y_unit: Unit = _DEFAULT_REGISTRY.dimensionless,
    ):
        super(Polynomial, self).__init__(coef)
        self.x_unit = x_unit
        self.y_unit = y_unit

    @property
    def y_intercept(self) -> Quantity:
        return self.coef[0] * self.y_unit

    @staticmethod
    def _get_roots_of_polynomial(
        poly: p.Polynomial,
        x_min: float = -inf,
        x_max: float = inf,
        real_only: bool = True,
    ):
        roots: set = set(poly.roots())
        if real_only:
            roots = {float(root.real) for root in roots if root == root.real}
        roots: list = list({root for root in roots if x_min <= float(root) <= x_max})
        return roots

    def _x_at_y(
        self,
        y_value: float,
        x_min: float = -inf,
        x_max: float = inf,
        real_only: bool = True,
    ) -> list[Union[float, complex]]:
        solutions = self._get_roots_of_polynomial(
            (self - y_value), x_min, x_max, real_only
        )
        if len(solutions) == 1:
            return solutions[0]
        return solutions

    @property
    def real_roots(self) -> list[float]:
        return self._get_roots_of_polynomial(self, real_only=True)

    @property
    def positive_real_roots(self) -> list[float]:
        return self._get_roots_of_polynomial(self, x_min=0, real_only=True)

    @property
    def x_intercept(self) -> Optional[Quantity]:
        roots = self.real_roots
        if len(roots) == 0:
            return None
        else:
            root = min(roots)
        return root * self.x_unit

    def solve(self, value: Quantity, min_value: float = -inf) -> Quantity:
        x = value.m_as(self.x_unit)
        return max(self(x), min_value) * self.y_unit

    def solve_for_x(self, y_value: Quantity, min_value: float = -inf) -> Quantity:
        y = y_value.m_as(self.y_unit)
        try:
            return self._x_at_y(y, x_min=min_value, real_only=True) * self.x_unit
        except TypeError:
            raise ValueError(
                "There are no values of {} that output {}!".format(self, y_value)
            )

    def integ(self, m=1, k=None, lbnd=None) -> "Polynomial":
        k = k if k is not None else []
        return self.__class__(
            super(Polynomial, self).integ(m, k, lbnd).coef,
            self.x_unit,
            self.y_unit * self.x_unit**m,
        )

    @property
    def integral(self) -> "Polynomial":
        return self.integ(1)

    def deriv(self, m=1) -> "Polynomial":
        return self.__class__(
            super(Polynomial, self).deriv(m).coef,
            self.x_unit,
            self.y_unit * self.x_unit**-m,
        )

    @property
    def derivative(self) -> "Polynomial":
        return self.deriv(1)

    def derivative_at(self, x: Quantity, derivative_order: int = 1) -> Quantity:
        return self.deriv(derivative_order).solve(x)

    def __pow__(self, power, modulo=None):
        x_unit, y_unit = self.x_unit, self.y_unit
        if isinstance(power, self.__class__):
            x_unit **= power.x_unit
            y_unit **= power.y_unit
            new_coefficients = p.polypow(self.coef, power)
        else:
            new_coefficients = p.polypow(
                self.coef, super(Polynomial, self)._get_coefficients(power)
            )

        new_poly = sum(map(self.__class__, new_coefficients))
        return self.__class__(new_poly.coef, x_unit, y_unit)

    def __truediv__(self, other) -> "Polynomial":
        x_unit, y_unit = self.x_unit, self.y_unit
        if isinstance(other, self.__class__):
            x_unit /= other.x_unit
            y_unit /= other.y_unit
            new_coefficients = p.polydiv(self.coef, other.coef)
        else:
            new_coefficients = p.polydiv(
                self.coef, super(Polynomial, self)._get_coefficients(other)
            )

        new_poly = sum(map(self.__class__, new_coefficients))
        return self.__class__(new_poly.coef, x_unit, y_unit)

    def __mul__(self, other) -> "Polynomial":
        if isinstance(other, self.__class__):
            return self.__class__(
                super(Polynomial, self).__mul__(other).coef,
                self.x_unit * other.x_unit,
                self.y_unit * other.y_unit,
            )
        return self.__class__(
            super(Polynomial, self).__mul__(other).coef, self.x_unit, self.y_unit
        )

    def __add__(self, other) -> "Polynomial":
        self._polynomials_have_compatible_units(other)
        return self.__class__(
            super(Polynomial, self).__add__(other).coef, self.x_unit, self.y_unit
        )

    def __sub__(self, other) -> "Polynomial":
        self._polynomials_have_compatible_units(other)
        return self.__class__(
            super(Polynomial, self).__sub__(other).coef, self.x_unit, self.y_unit
        )

    def __neg__(self) -> "Polynomial":
        return self * -1

    def __rtruediv__(self, other) -> "Polynomial":
        if isinstance(other, self.__class__):
            return other.__truediv__(self)
        return super(Polynomial, self).__rtruediv__(other)

    def __rmul__(self, other) -> "Polynomial":
        return self * other

    def __radd__(self, other) -> "Polynomial":
        return self + other

    def __rsub__(self, other) -> "Polynomial":
        return -self + other

    def _polynomials_have_compatible_units(self, other):
        if not isinstance(other, self.__class__):
            return
        if self.x_unit == other.x_unit and self.y_unit == other.y_unit:
            return
        raise TypeError(
            "Units between {} 1 ({}, {}) {} 2 ({}, {}) are not compatible".format(
                self.__class__.__name__,
                self.x_unit,
                self.y_unit,
                other.__class__.__name__,
                other.x_unit,
                other.y_unit,
            )
        )
