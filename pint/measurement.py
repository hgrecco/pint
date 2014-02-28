# -*- coding: utf-8 -*-
"""
    pint.measurement
    ~~~~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from .compat import ufloat

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

    def __repr__(self):
        return "<Measurement({0:!r}, {1:!r})>".format(self._value, self._error)

