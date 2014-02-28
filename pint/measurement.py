# -*- coding: utf-8 -*-
"""
    pint.measurement
    ~~~~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from .compat import ufloat
import operator

MISSING = object()

class _Measurement(object):
    """Implements a class to describe a quantity with uncertainty.

    :param value: The most likely value of the measurement.
    :type value: Quantity or Number
    :param error: The error or uncertainty of the measurement.
    :type error: Quantity or Number
    """

    def __new__(cls, value, error, units=MISSING):
        if units is MISSING:
            try:
                value, units = value.magnitude, value.units
            except AttributeError:
                try:
                    value, error, units = value.nominal_value, value.std_dev, error
                except AttributeError:
                    units = ''
        try:
            error = error.to(units).magnitude
        except AttributeError:
            pass

        inst = super(_Measurement, cls).__new__(cls, ufloat(value, error), units)

        if error < 0:
            raise ValueError('The magnitude of the error cannot be negative'.format(value, error))
        return inst

    @property
    def value(self):
        return self._REGISTRY.Quantity(self.magnitude.nominal_value, self.units)

    @property
    def error(self):
        return self._REGISTRY.Quantity(self.magnitude.std_dev, self.units)

    @property
    def rel(self):
        return float(abs(self.magnitude.std_dev / self.magnitude.nominal_value))

    def _add_sub(self, other, operator):
        result = self.value + other.value
        if isinstance(other, self.__class__):
            error = (self.error ** 2.0 + other.error ** 2.0) ** (1/2)
        else:
            error = self.error
        return result.plus_minus(error)

    def __add__(self, other):
        return self._add_sub(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._add_sub(other, operator.sub)

    __rsub__ = __sub__

    def _mul_div(self, other, operator):
        if isinstance(other, self.__class__):
            result = operator(self.value, other.value)
            return result.plus_minus((self.rel ** 2.0 + other.rel ** 2.0) ** (1/2), relative=True)
        else:
            result = operator(self.value, other)
            return result.plus_minus(abs(operator(self.error, other)))

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._mul_div(other, operator.truediv)

    def __floordiv__(self, other):
        return self._mul_div(other, operator.floordiv)

    __div__ = __floordiv__

    def __str__(self):
        return '{}'.format(self)

    def __repr__(self):
        return "<Measurement({0:!r}, {1:!r})>".format(self._value, self._error)

    def __format__(self, spec):
        if '!' in spec:
            fmt, conv = spec.split('!')
            conv = '!' + conv
        else:
            fmt, conv = spec, ''

        left, right = '(', ')'
        if '!l' == conv:
            pm = r'\pm'
            left = r'\left' + left
            right = r'\right' + right
        elif '!p' == conv:
            pm = '±'
        else:
            pm = '+/-'

        if hasattr(self.value, 'units'):
            vmag = format(self.value.magnitude, fmt)
            if self.value.units != self.error.units:
                emag = self.error.to(self.value.units).magnitude
            else:
                emag = self.error.magnitude
            emag = format(emag, fmt)
            units = ' ' + format(self.value.units, conv)
        else:
            vmag, emag, units = self.value, self.error, ''

        return left + vmag + ' ' + pm + ' ' +  emag + right + units if units else ''
