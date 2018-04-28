# -*- coding: utf-8 -*-
"""
    pint.unit
    ~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import operator
from numbers import Number

from .util import (
    PrettyIPython, UnitsContainer, SharedRegistryObject, fix_str_conversions)

from .compat import string_types, NUMERIC_TYPES, long_type
from .formatting import siunitx_format_unit
from .definitions import UnitDefinition


@fix_str_conversions
class _Unit(PrettyIPython, SharedRegistryObject):
    """Implements a class to describe a unit supporting math operations.

    :type units: UnitsContainer, str, Unit or Quantity.

    """

    #: Default formatting string.
    default_format = ''

    def __reduce__(self):
        from . import _build_unit
        return _build_unit, (self._units, )

    def __new__(cls, units):
        inst = object.__new__(cls)
        if isinstance(units, (UnitsContainer, UnitDefinition)):
            inst._units = units
        elif isinstance(units, string_types):
            inst._units = inst._REGISTRY.parse_units(units)._units
        elif isinstance(units, _Unit):
            inst._units = units._units
        else:
            raise TypeError('units must be of type str, Unit or '
                            'UnitsContainer; not {}.'.format(type(units)))

        inst.__used = False
        inst.__handling = None
        return inst

    @property
    def debug_used(self):
        return self.__used

    def __copy__(self):
        ret = self.__class__(self._units)
        ret.__used = self.__used
        return ret

    def __deepcopy__(self, memo):
      ret = self.__class__(copy.deepcopy(self._units))
      ret.__used = self.__used
      return ret

    def __str__(self):
        return format(self)

    def __repr__(self):
        return "<Unit('{}')>".format(self._units)

    def __format__(self, spec):
        spec = spec or self.default_format
        # special cases
        if 'Lx' in spec: # the LaTeX siunitx code
          opts = ''
          ustr = siunitx_format_unit(self)
          ret = r'\si[%s]{%s}'%( opts, ustr )
          return ret


        if '~' in spec:
            if not self._units:
                return ''
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                         value)
                                   for key, value in self._units.items()))
            spec = spec.replace('~', '')
        else:
            units = self._units

        return '%s' % (format(units, spec))

    def format_babel(self, spec='', **kwspec):
        spec = spec or self.default_format

        if '~' in spec:
            if self.dimensionless:
                return ''
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                         value)
                                   for key, value in self._units.items()))
            spec = spec.replace('~', '')
        else:
            units = self._units

        return '%s' % (units.format_babel(spec, **kwspec))

    @property
    def dimensionless(self):
        """Return true if the Unit is dimensionless.

        """
        return not bool(self.dimensionality)

    @property
    def dimensionality(self):
        """Unit's dimensionality (e.g. {length: 1, time: -1})

        """
        try:
            return self._dimensionality
        except AttributeError:
            dim = self._REGISTRY._get_dimensionality(self._units)
            self._dimensionality = dim

        return self._dimensionality

    def compatible_units(self, *contexts):
        if contexts:
            with self._REGISTRY.context(*contexts):
                return self._REGISTRY.get_compatible_units(self)

        return self._REGISTRY.get_compatible_units(self)

    def __mul__(self, other):
        if self._check(other):
            if isinstance(other, self.__class__):
                return self.__class__(self._units*other._units)
            else:
                qself = self._REGISTRY.Quantity(1.0, self._units)
                return qself * other

        if isinstance(other, Number) and other == 1:
            return self._REGISTRY.Quantity(other, self._units)

        return self._REGISTRY.Quantity(1, self._units) * other

    __rmul__ = __mul__

    def __truediv__(self, other):
        if self._check(other):
            if isinstance(other, self.__class__):
                return self.__class__(self._units/other._units)
            else:
                qself = 1.0 * self
                return qself / other

        return self._REGISTRY.Quantity(1/other, self._units)

    def __rtruediv__(self, other):
        # As Unit and Quantity both handle truediv with each other rtruediv can
        # only be called for something different.
        if isinstance(other, NUMERIC_TYPES):
            return self._REGISTRY.Quantity(other, 1/self._units)
        elif isinstance(other, UnitsContainer):
            return self.__class__(other/self._units)
        else:
            return NotImplemented

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __pow__(self, other):
        if isinstance(other, NUMERIC_TYPES):
            return self.__class__(self._units**other)

        else:
            mess = 'Cannot power Unit by {}'.format(type(other))
            raise TypeError(mess)

    def __hash__(self):
        return self._units.__hash__()

    def __eq__(self, other):
        # We compare to the base class of Unit because each Unit class is
        # unique.
        if self._check(other):
            if isinstance(other, self.__class__):
                return self._units == other._units
            else:
                return other == self._REGISTRY.Quantity(1, self._units)

        elif isinstance(other, NUMERIC_TYPES):
            return other == self._REGISTRY.Quantity(1, self._units)

        else:
            return self._units == other

    def __ne__(self, other):
        return not (self == other)

    def compare(self, other, op):
        self_q = self._REGISTRY.Quantity(1, self)

        if isinstance(other, NUMERIC_TYPES):
            return self_q.compare(other, op)
        elif isinstance(other, (_Unit, UnitsContainer, dict)):
            return self_q.compare(self._REGISTRY.Quantity(1, other), op)
        else:
            return NotImplemented

    __lt__ = lambda self, other: self.compare(other, op=operator.lt)
    __le__ = lambda self, other: self.compare(other, op=operator.le)
    __ge__ = lambda self, other: self.compare(other, op=operator.ge)
    __gt__ = lambda self, other: self.compare(other, op=operator.gt)

    def __int__(self):
        return int(self._REGISTRY.Quantity(1, self._units))

    def __long__(self):
        return long_type(self._REGISTRY.Quantity(1, self._units))

    def __float__(self):
        return float(self._REGISTRY.Quantity(1, self._units))

    def __complex__(self):
        return complex(self._REGISTRY.Quantity(1, self._units))

    __array_priority__ = 17

    def __array_prepare__(self, array, context=None):
        return 1

    def __array_wrap__(self, array, context=None):
        uf, objs, huh = context

        if uf.__name__ in ('true_divide', 'divide', 'floor_divide'):
            return self._REGISTRY.Quantity(array, 1/self._units)
        elif uf.__name__ in ('multiply',):
            return self._REGISTRY.Quantity(array, self._units)
        else:
            raise ValueError('Unsupproted operation for Unit')

    @property
    def systems(self):
        out = set()
        for uname in self._units.keys():
            for sname, sys in self._REGISTRY._systems.items():
                if uname in sys.members:
                    out.add(sname)
        return frozenset(out)

    def from_(self, value, strict=True, name='value'):
        """Converts a numerical value or quantity to this unit

        :param value: a Quantity (or numerical value if strict=False) to convert
        :param strict: boolean to indicate that only quanities are accepted
        :param name: descriptive name to use if an exception occurs
        :return: The converted value as this unit
        :raises:
            :class:`ValueError` if strict and one of the arguments is not a Quantity.
        """
        if self._check(value):
            if not isinstance(value, self._REGISTRY.Quantity):
                value = self._REGISTRY.Quantity(1, value)
            return value.to(self)
        elif strict:
            raise ValueError("%s must be a Quantity" % value)
        else:
            return value * self

    def m_from(self, value, strict=True, name='value'):
        """Converts a numerical value or quantity to this unit, then returns
        the magnitude of the converted value

        :param value: a Quantity (or numerical value if strict=False) to convert
        :param strict: boolean to indicate that only quanities are accepted
        :param name: descriptive name to use if an exception occurs
        :return: The magnitude of the converted value
        :raises:
            :class:`ValueError` if strict and one of the arguments is not a Quantity.
        """
        return self.from_(value, strict=strict, name=name).magnitude

def build_unit_class(registry):

    class Unit(_Unit):
        pass

    Unit._REGISTRY = registry
    return Unit
