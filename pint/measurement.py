# -*- coding: utf-8 -*-
"""
    pint.measurement
    ~~~~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import


import operator


class Measurement(object):
    """Implements a class to describe a quantity with uncertainty.

    :param value: The most likely value of the measurement.
    :type value: Quantity or Number
    :param error: The error or uncertainty of the measurement.
    :type value: Quantity or Number
    """

    def __init__(self, value, error):
        if not (value/error).unitless:
            raise ValueError('{} and {} have incompatible units'.format(value, error))
        try:
            emag = error.magnitude
        except AttributeError:
            emag = error

        if emag < 0:
            raise ValueError('The magnitude of the error cannot be negative'.format(value, error))

        self._value = value
        self._error = error

    @property
    def value(self):
        return self._value

    @property
    def error(self):
        return self._error

    @property
    def rel(self):
        return float(abs(self._error / self._value))

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
        return "<Measurement({:!r}, {:!r})>".format(self._value, self._error)

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
