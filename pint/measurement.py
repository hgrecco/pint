# -*- coding: utf-8 -*-
"""
    pint.measurement
    ~~~~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from .compat import ufloat
from .formatting import _FORMATS

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
        return "<Measurement({0:.2f}, {1:.2f}, {2})>".format(self.magnitude.nominal_value,
                                                             self.magnitude.std_dev,
                                                             self.units)

    def __str__(self):
        return '{0}'.format(self)

    def __format__(self, spec):
        if 'L' in spec:
            newpm = pm = r'  \pm  '
            pars = _FORMATS['L']['parentheses_fmt']
        elif 'P' in spec:
            newpm = pm = '±'
            pars = _FORMATS['P']['parentheses_fmt']
        else:
            newpm = pm = '+/-'
            pars = _FORMATS['']['parentheses_fmt']

        if 'C' in spec:
            sp = ''
            newspec = spec.replace('C', '')
            pars = _FORMATS['C']['parentheses_fmt']
        else:
            sp = ' '
            newspec = spec

        if 'H' in spec:
            newpm = '&plusmn;'
            newspec = spec.replace('H', '')
            pars = _FORMATS['H']['parentheses_fmt']

        mag = format(self.magnitude, newspec).replace(pm, sp + newpm + sp)

        if 'L' in newspec and 'S' in newspec:
            mag = mag.replace('(', r'\left(').replace(')', r'\right)')

        if 'uS' in newspec or 'ue' in newspec or 'u%' in newspec:
            return mag + ' ' + format(self.units, spec)
        else:
            return pars.format(mag) + ' ' + format(self.units, spec)



