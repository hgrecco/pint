# -*- coding: utf-8 -*-
"""
    pint.measurement
    ~~~~~~~~~~~~~~~~

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from .compat import ufloat
from .formatting import _FORMATS, siunitx_format_unit

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
                #if called with two arguments and the first looks like a ufloat
                # then assume the second argument is the units, keep value intact
                if hasattr(value,"nominal_value"):
                    units = error
                    error = MISSING #used for check below
                else:
                    units = ''
        try:
            error = error.to(units).magnitude
        except AttributeError:
            pass
        
        if error is MISSING:
            mag = value
        elif error < 0:
            raise ValueError('The magnitude of the error cannot be negative'.format(value, error))
        else:
            mag = ufloat(value,error)
            
        inst = super(_Measurement, cls).__new__(cls, mag, units)
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
        # special cases
        if 'Lx' in spec: # the LaTeX siunitx code
            # the uncertainties module supports formatting
            # numbers in value(unc) notation (i.e. 1.23(45) instead of 1.23 +/- 0.45),
            # which siunitx actually accepts as input. we just need to give the 'S'
            # formatting option for the uncertainties module.
            spec = spec.replace('Lx','S')
            # todo: add support for extracting options
            opts = 'separate-uncertainty=true'
            mstr = format( self.magnitude, spec )
            ustr = siunitx_format_unit(self.units)
            ret = r'\SI[%s]{%s}{%s}'%( opts, mstr, ustr )
            return ret


        # standard cases
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

        if 'L' in newspec:
            space = r'\ '
        else:
            space = ' '

        if 'uS' in newspec or 'ue' in newspec or 'u%' in newspec:
            return mag + space + format(self.units, spec)
        else:
            return pars.format(mag) + space + format(self.units, spec)


def build_measurement_class(registry, force_ndarray=False):

    if ufloat is None:
        class Measurement(object):

            def __init__(self, *args):
                raise RuntimeError("Pint requires the 'uncertainties' package to create a Measurement object.")

    else:
        class Measurement(_Measurement, registry.Quantity):
            pass

    Measurement._REGISTRY = registry
    Measurement.force_ndarray = force_ndarray

    return Measurement
