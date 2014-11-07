# -*- coding: utf-8 -*-
"""
    pint.quantity
    ~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import math
import operator
import functools

from .formatting import remove_custom_flags
from .unit import (DimensionalityError, OffsetUnitCalculusError,
                   UnitsContainer, UnitDefinition, UndefinedUnitError)
from .compat import string_types, ndarray, np, _to_magnitude, long_type
from .util import logger


def _eq(first, second, check_all):
    """Comparison of scalars and arrays
    """
    out = first == second
    if check_all and isinstance(out, ndarray):
        return np.all(out)
    return out


class _Exception(Exception):            # pragma: no cover

    def __init__(self, internal):
        self.internal = internal


def _check(q1, other):
    """Check Quantities before math operations.

    Return True if q1 and other are from the same class.
    Raise a ValueError if other has a different _REGISTRY than q1.

    In other case, return False.
    """

    if isinstance(other, q1.__class__):
        # Both quantities are the same class and therefore from the same registry.
        # (Each registry has its own Quantity class)
        return True
    elif q1._REGISTRY is getattr(other, '_REGISTRY', None):
        return True
    elif isinstance(other, _Quantity):
        # The other object is a Quantity but from another registry.
        raise ValueError('Cannot operate between quantities of different registries')

    return False


class _Quantity(object):
    """Implements a class to describe a physical quantity:
    the product of a numerical value and a unit of measurement.

    :param value: value of the physical quantity to be created.
    :type value: str, Quantity or any numeric type.
    :param units: units of the physical quantity to be created.
    :type units: UnitsContainer, str or Quantity.
    """

    #: Default formatting string.
    default_format = ''

    def __reduce__(self):
        from . import _build_quantity
        return _build_quantity, (self.magnitude, self.units)

    def __new__(cls, value, units=None):
        if units is None:
            if isinstance(value, string_types):
                if value == '':
                    raise ValueError('Expression to parse as Quantity cannot be an empty string.')
                inst = cls._REGISTRY.parse_expression(value)
                return cls.__new__(cls, inst)
            elif isinstance(value, cls):
                inst = copy.copy(value)
            else:
                inst = object.__new__(cls)
                inst._magnitude = _to_magnitude(value, inst.force_ndarray)
                inst._units = UnitsContainer()
        elif isinstance(units, (UnitsContainer, UnitDefinition)):
            inst = object.__new__(cls)
            inst._magnitude = _to_magnitude(value, inst.force_ndarray)
            inst._units = units
        elif isinstance(units, string_types):
            inst = object.__new__(cls)
            inst._magnitude = _to_magnitude(value, inst.force_ndarray)
            inst._units = inst._REGISTRY.parse_units(units)
        elif isinstance(units, cls):
            if units.magnitude != 1:
                logger.warning('Creating new Quantity using a non unity Quantity as units.')
            inst = copy.copy(units)
            inst._magnitude = _to_magnitude(value, inst.force_ndarray)
        else:
            raise TypeError('units must be of type str, Quantity or '
                            'UnitsContainer; not {0}.'.format(type(units)))

        inst.__used = False
        inst.__handling = None
        return inst

    @property
    def debug_used(self):
        return self.__used

    def __copy__(self):
        ret = self.__class__(copy.copy(self._magnitude), copy.copy(self._units))
        ret.__used = self.__used
        return ret

    def __str__(self):
        return format(self)

    def __repr__(self):
        return "<Quantity({0}, '{1}')>".format(self._magnitude, self._units)

    def __format__(self, spec):
        spec = spec or self.default_format

        if '~' in spec:
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key), value)
                                   for key, value in self.units.items()))
            spec = spec.replace('~', '')
        else:
            units = self.units

        return '%s %s' % (format(self.magnitude, remove_custom_flags(spec)),
                          format(units, spec))

    # IPython related code
    def _repr_html_(self):
        return self.__format__('H')

    def _repr_latex_(self):
        return "$" + self.__format__('L') + "$"

    @property
    def magnitude(self):
        """Quantity's magnitude.
        """
        return self._magnitude

    @property
    def units(self):
        """Quantity's units.

        :rtype: UnitContainer
        """
        return self._units

    @property
    def unitless(self):
        """Return true if the quantity does not have units.
        """
        return not bool(self.to_base_units().units)

    @property
    def dimensionless(self):
        """Return true if the quantity is dimensionless.
        """
        tmp = copy.copy(self).to_base_units()

        return not bool(tmp.dimensionality)

    @property
    def dimensionality(self):
        """Quantity's dimensionality (e.g. {length: 1, time: -1})
        """
        try:
            return self._dimensionality
        except AttributeError:
            self._dimensionality = self._REGISTRY.get_dimensionality(self.units)

        return self._dimensionality

    def compatible_units(self, *contexts):
        if contexts:
            with self._REGISTRY.context(*contexts):
                return self._REGISTRY.get_compatible_units(self._units)

        return self._REGISTRY.get_compatible_units(self._units)

    def _convert_magnitude_not_inplace(self, other, *contexts, **ctx_kwargs):
        if contexts:
            with self._REGISTRY.context(*contexts, **ctx_kwargs):
                return self._REGISTRY.convert(self._magnitude, self._units, other)

        return self._REGISTRY.convert(self._magnitude, self._units, other)

    def _convert_magnitude(self, other, *contexts, **ctx_kwargs):
        if contexts:
            with self._REGISTRY.context(*contexts, **ctx_kwargs):
                return self._REGISTRY.convert(self._magnitude, self._units, other)

        return self._REGISTRY.convert(self._magnitude, self._units, other,
                                      inplace=isinstance(self._magnitude, ndarray))

    def ito(self, other=None, *contexts, **ctx_kwargs):
        """Inplace rescale to different units.

        :param other: destination units.
        :type other: Quantity, str or dict
        """
        if isinstance(other, string_types):
            other = self._REGISTRY.parse_units(other)
        elif isinstance(other, self.__class__):
            other = copy.copy(other.units)
        elif isinstance(other, UnitsContainer):
            pass
        else:
            other = UnitsContainer(other)

        self._magnitude = self._convert_magnitude(other, *contexts, **ctx_kwargs)
        self._units = other

        return None

    def to(self, other=None, *contexts, **ctx_kwargs):
        """Return Quantity rescaled to different units.

        :param other: destination units.
        :type other: Quantity, str or dict
        """
        if isinstance(other, string_types):
            other = self._REGISTRY.parse_units(other)
        elif isinstance(other, self.__class__):
            other = copy.copy(other.units)
        elif isinstance(other, UnitsContainer):
            pass
        else:
            other = UnitsContainer(other)

        magnitude = self._convert_magnitude_not_inplace(other, *contexts, **ctx_kwargs)

        return self.__class__(magnitude, other)

    def ito_base_units(self):
        """Return Quantity rescaled to base units
        """

        _, other = self._REGISTRY.get_base_units(self.units)

        self._magnitude = self._convert_magnitude(other)
        self._units = other

        return None

    def to_base_units(self):
        """Return Quantity rescaled to base units
        """

        _, other = self._REGISTRY.get_base_units(self.units)

        magnitude = self._convert_magnitude_not_inplace(other)

        return self.__class__(magnitude, other)

    # Mathematical operations
    def __int__(self):
        if self.dimensionless:
            return int(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def __long__(self):
        if self.dimensionless:
            return long_type(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def __float__(self):
        if self.dimensionless:
            return float(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def __complex__(self):
        if self.dimensionless:
            return complex(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def _iadd_sub(self, other, op):
        """Perform addition or subtraction operation in-place and return the result.

        :param other: object to be added to / subtracted from self
        :type other: Quantity or any type accepted by :func:`_to_magnitude`
        :param op: operator function (e.g. operator.add, operator.isub)
        :type op: function
        """
        if not _check(self, other):
            # other not from same Registry or not a Quantity
            try:
                other_magnitude = _to_magnitude(other, self.force_ndarray)
            except TypeError:
                return NotImplemented
            if _eq(other, 0, True):
                # If the other value is 0 (but not Quantity 0)
                # do the operation without checking units.
                # We do the calculation instead of just returning the same value to
                # enforce any shape checking and type casting due to the operation.
                self._magnitude = op(self._magnitude, other_magnitude)
            elif self.dimensionless:
                self.ito(UnitsContainer())
                self._magnitude = op(self._magnitude, other_magnitude)
            else:
                raise DimensionalityError(self.units, 'dimensionless')
            return self

        if not self.dimensionality == other.dimensionality:
            raise DimensionalityError(self.units, other.units,
                                      self.dimensionality,
                                      other.dimensionality)

        # Next we define some variables to make if-clauses more readable.
        self_non_mul_units = self._get_non_multiplicative_units()
        is_self_multiplicative = len(self_non_mul_units) == 0
        if len(self_non_mul_units) == 1:
            self_non_mul_unit = self_non_mul_units[0]
        other_non_mul_units = other._get_non_multiplicative_units()
        is_other_multiplicative = len(other_non_mul_units) == 0
        if len(other_non_mul_units) == 1:
            other_non_mul_unit = other_non_mul_units[0]

        # Presence of non-multiplicative units gives rise to several cases.
        if is_self_multiplicative and is_other_multiplicative:
            if self._units == other._units:
                self._magnitude = op(self._magnitude, other._magnitude)
            # If only self has a delta unit, other determines unit of result.
            elif self._get_delta_units() and not other._get_delta_units():
                self._magnitude = op(self._convert_magnitude(other.units),
                                     other._magnitude)
                self._units = copy.copy(other.units)
            else:
                self._magnitude = op(self._magnitude,
                                     other.to(self.units)._magnitude)

        elif (op == operator.isub and len(self_non_mul_units) == 1
                and self.units[self_non_mul_unit] == 1
                and not other._has_compatible_delta(self_non_mul_unit)):
            if self.units == other.units:
                self._magnitude = op(self._magnitude, other._magnitude)
            else:
                self._magnitude = op(self._magnitude,
                                     other.to(self.units)._magnitude)
            self.units['delta_' + self_non_mul_unit
                       ] = self.units.pop(self_non_mul_unit)

        elif (op == operator.isub and len(other_non_mul_units) == 1
                and other.units[other_non_mul_unit] == 1
                and not self._has_compatible_delta(other_non_mul_unit)):
            # we convert to self directly since it is multiplicative
            self._magnitude = op(self._magnitude,
                                other.to(self.units)._magnitude)

        elif (len(self_non_mul_units) == 1
                # order of the dimension of offset unit == 1 ?
                and self._units[self_non_mul_unit] == 1
                and other._has_compatible_delta(self_non_mul_unit)):
            tu = copy.copy(self.units)
            # Replace offset unit in self by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu['delta_' + self_non_mul_unit] = tu.pop(self_non_mul_unit)
            self._magnitude = op(self._magnitude, other.to(tu)._magnitude)
        elif (len(other_non_mul_units) == 1
                # order of the dimension of offset unit == 1 ?
                and other._units[other_non_mul_unit] == 1
                and self._has_compatible_delta(other_non_mul_unit)):
            tu = copy.copy(other.units)
            # Replace offset unit in other by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu['delta_' + other_non_mul_unit] = tu.pop(other_non_mul_unit)
            self._magnitude = op(self._convert_magnitude(tu), other._magnitude)
            self._units = copy.copy(other.units)
        else:
            raise OffsetUnitCalculusError(self.units, other.units)

        return self

    def _add_sub(self, other, op):
        """Perform addition or subtraction operation and return the result.

        :param other: object to be added to / subtracted from self
        :type other: Quantity or any type accepted by :func:`_to_magnitude`
        :param op: operator function (e.g. operator.add, operator.isub)
        :type op: function
        """
        if not _check(self, other):
            # other not from same Registry or not a Quantity
            if _eq(other, 0, True):
                # If the other value is 0 (but not Quantity 0)
                # do the operation without checking units.
                # We do the calculation instead of just returning the same value to
                # enforce any shape checking and type casting due to the operation.
                units = self.units
                magnitude = op(self._magnitude,
                               _to_magnitude(other, self.force_ndarray))
            elif self.dimensionless:
                units = UnitsContainer()
                magnitude = op(self.to(units)._magnitude,
                               _to_magnitude(other, self.force_ndarray))
            else:
                raise DimensionalityError(self.units, 'dimensionless')
            return self.__class__(magnitude, units)

        if not self.dimensionality == other.dimensionality:
            raise DimensionalityError(self.units, other.units,
                                      self.dimensionality,
                                      other.dimensionality)

        # Next we define some variables to make if-clauses more readable.
        self_non_mul_units = self._get_non_multiplicative_units()
        is_self_multiplicative = len(self_non_mul_units) == 0
        if len(self_non_mul_units) == 1:
            self_non_mul_unit = self_non_mul_units[0]
        other_non_mul_units = other._get_non_multiplicative_units()
        is_other_multiplicative = len(other_non_mul_units) == 0
        if len(other_non_mul_units) == 1:
            other_non_mul_unit = other_non_mul_units[0]

        # Presence of non-multiplicative units gives rise to several cases.
        if is_self_multiplicative and is_other_multiplicative:
            if self._units == other._units:
                magnitude = op(self._magnitude, other._magnitude)
                units = copy.copy(self.units)
            # If only self has a delta unit, other determines unit of result.
            elif self._get_delta_units() and not other._get_delta_units():
                magnitude = op(self._convert_magnitude(other.units),
                               other._magnitude)
                units = copy.copy(other.units)
            else:
                units = copy.copy(self.units)
                magnitude = op(self._magnitude,
                               other.to(self.units).magnitude)

        elif (op == operator.sub and len(self_non_mul_units) == 1
                and self.units[self_non_mul_unit] == 1
                and not other._has_compatible_delta(self_non_mul_unit)):
            if self.units == other.units:
                magnitude = op(self._magnitude, other._magnitude)
            else:
                magnitude = op(self._magnitude,
                               other.to(self.units)._magnitude)
            units = copy.copy(self.units)
            units['delta_' + self_non_mul_unit] = units.pop(self_non_mul_unit)

        elif (op == operator.sub and len(other_non_mul_units) == 1
                and other.units[other_non_mul_unit] == 1
                and not self._has_compatible_delta(other_non_mul_unit)):
            # we convert to self directly since it is multiplicative
            magnitude = op(self._magnitude,
                           other.to(self.units)._magnitude)
            units = copy.copy(self.units)

        elif (len(self_non_mul_units) == 1
                # order of the dimension of offset unit == 1 ?
                and self._units[self_non_mul_unit] == 1
                and other._has_compatible_delta(self_non_mul_unit)):
            tu = copy.copy(self.units)
            # Replace offset unit in self by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu['delta_' + self_non_mul_unit] = tu.pop(self_non_mul_unit)
            magnitude = op(self._magnitude, other.to(tu).magnitude)
            units = copy.copy(self.units)
        elif (len(other_non_mul_units) == 1
                # order of the dimension of offset unit == 1 ?
                and other._units[other_non_mul_unit] == 1
                and self._has_compatible_delta(other_non_mul_unit)):
            tu = copy.copy(other.units)
            # Replace offset unit in other by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu['delta_' + other_non_mul_unit] = tu.pop(other_non_mul_unit)
            magnitude = op(self._convert_magnitude(tu), other._magnitude)
            units = copy.copy(other.units)
        else:
            raise OffsetUnitCalculusError(self.units, other.units)

        return self.__class__(magnitude, units)

    def __iadd__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self._add_sub(other, operator.add)
        else:
            return self._iadd_sub(other, operator.iadd)

    def __add__(self, other):
        return self._add_sub(other, operator.add)

    __radd__ = __add__

    def __isub__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self._add_sub(other, operator.sub)
        else:
            return self._iadd_sub(other, operator.isub)

    def __sub__(self, other):
        return self._add_sub(other, operator.sub)

    def __rsub__(self, other):
        return -self._add_sub(other, operator.sub)

    def _imul_div(self, other, magnitude_op, units_op=None):
        """Perform multiplication or division operation in-place and return the result.

        :param other: object to be multiplied/divided with self
        :type other: Quantity or any type accepted by :func:`_to_magnitude`
        :param magnitude_op: operator function to perform on the magnitudes (e.g. operator.mul)
        :type magnitude_op: function
        :param units_op: operator function to perform on the units; if None, *magnitude_op* is used
        :type units_op: function or None
        """
        if units_op is None:
            units_op = magnitude_op

        offset_units_self = self._get_non_multiplicative_units()
        no_offset_units_self = len(offset_units_self)

        if not _check(self, other):
            if not self._ok_for_muldiv(no_offset_units_self):
                raise OffsetUnitCalculusError(self.units,
                                              getattr(other, 'units', ''))
            if len(offset_units_self) == 1:
                if (self.units[offset_units_self[0]] != 1
                        or magnitude_op not in [operator.mul, operator.imul]):
                    raise OffsetUnitCalculusError(self.units,
                                                  getattr(other, 'units', ''))
            try:
                other_magnitude = _to_magnitude(other, self.force_ndarray)
            except TypeError:
                return NotImplemented
            self._magnitude = magnitude_op(self._magnitude, other_magnitude)
            self._units = units_op(self._units, UnitsContainer())
            return self

        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self.units, other.units)
        elif no_offset_units_self == 1 and len(self.units) == 1:
                self.ito_base_units()

        no_offset_units_other = len(other._get_non_multiplicative_units())

        if not other._ok_for_muldiv(no_offset_units_other):
            raise OffsetUnitCalculusError(self.units, other.units)
        elif no_offset_units_other == 1 and len(other.units) == 1:
            other.ito_base_units()

        self._magnitude = magnitude_op(self._magnitude, other._magnitude)
        self._units = units_op(self._units, other._units)

        return self

    def _mul_div(self, other, magnitude_op, units_op=None):
        """Perform multiplication or division operation and return the result.

        :param other: object to be multiplied/divided with self
        :type other: Quantity or any type accepted by :func:`_to_magnitude`
        :param magnitude_op: operator function to perform on the magnitudes (e.g. operator.mul)
        :type magnitude_op: function
        :param units_op: operator function to perform on the units; if None, *magnitude_op* is used
        :type units_op: function or None
        """
        if units_op is None:
            units_op = magnitude_op

        offset_units_self = self._get_non_multiplicative_units()
        no_offset_units_self = len(offset_units_self)

        if not _check(self, other):
            if not self._ok_for_muldiv(no_offset_units_self):
                raise OffsetUnitCalculusError(self.units,
                                              getattr(other, 'units', ''))
            if len(offset_units_self) == 1:
                if (self.units[offset_units_self[0]] != 1
                        or magnitude_op not in [operator.mul, operator.imul]):
                    raise OffsetUnitCalculusError(self.units,
                                                  getattr(other, 'units', ''))
            try:
                other_magnitude = _to_magnitude(other, self.force_ndarray)
            except TypeError:
                return NotImplemented

            magnitude = magnitude_op(self._magnitude, other_magnitude)
            units = units_op(self._units, UnitsContainer())

            return self.__class__(magnitude, units)

        new_self = self

        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self.units, other.units)
        elif no_offset_units_self == 1 and len(self.units) == 1:
            new_self = self.to_base_units()

        no_offset_units_other = len(other._get_non_multiplicative_units())

        if not other._ok_for_muldiv(no_offset_units_other):
            raise OffsetUnitCalculusError(self.units, other.units)
        elif no_offset_units_other == 1 and len(other.units) == 1:
            other = other.to_base_units()

        magnitude = magnitude_op(new_self._magnitude, other._magnitude)
        units = units_op(new_self._units, other._units)

        return self.__class__(magnitude, units)

    def __imul__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self._mul_div(other, operator.mul)
        else:
            return self._imul_div(other, operator.imul)

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    __rmul__ = __mul__

    def __itruediv__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self._mul_div(other, operator.truediv)
        else:
            return self._imul_div(other, operator.itruediv)

    def __truediv__(self, other):
        return self._mul_div(other, operator.truediv)

    def __ifloordiv__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self._mul_div(other, operator.floordiv, units_op=operator.itruediv)
        else:
            return self._imul_div(other, operator.ifloordiv, units_op=operator.itruediv)

    def __floordiv__(self, other):
        return self._mul_div(other, operator.floordiv, units_op=operator.truediv)

    def __rtruediv__(self, other):
        try:
            other_magnitude = _to_magnitude(other, self.force_ndarray)
        except TypeError:
            return NotImplemented

        no_offset_units_self = len(self._get_non_multiplicative_units())
        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self.units, '')
        elif no_offset_units_self == 1 and len(self.units) == 1:
            self = self.to_base_units()

        return self.__class__(other_magnitude / self._magnitude, 1 / self._units)

    def __rfloordiv__(self, other):
        try:
            other_magnitude = _to_magnitude(other, self.force_ndarray)
        except TypeError:
            return NotImplemented

        no_offset_units_self = len(self._get_non_multiplicative_units())
        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self.units, '')
        elif no_offset_units_self == 1 and len(self.units) == 1:
            self = self.to_base_units()

        return self.__class__(other_magnitude // self._magnitude, 1 / self._units)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __idiv__ = __itruediv__

    def __ipow__(self, other):
        if not isinstance(self._magnitude, ndarray):
            return self.__pow__(other)

        try:
            other_magnitude = _to_magnitude(other, self.force_ndarray)
        except TypeError:
            return NotImplemented
        else:
            if not self._ok_for_muldiv:
                raise OffsetUnitCalculusError(self.units)

            if isinstance(getattr(other, '_magnitude', other), ndarray):
                # arrays are refused as exponent, because they would create
                #  len(array) quanitites of len(set(array)) different units
                if np.size(other) > 1:
                    raise DimensionalityError(self.units, 'dimensionless')

            new_self = self
            if other == 1:
                return self
            elif other == 0:
                self._units = UnitsContainer()
            else:
                if not self._is_multiplicative:
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        self.ito_base_units()
                    else:
                        raise OffsetUnitCalculusError(self.units)

                if getattr(other, 'dimensionless', False):
                    other = other.to_base_units()
                    self._units **= other.magnitude
                elif not getattr(other, 'dimensionless', True):
                    raise DimensionalityError(self.units, 'dimensionless')
                else:
                    self._units **= other

            self._magnitude **= _to_magnitude(other, self.force_ndarray)
            return self

    def __pow__(self, other):
        try:
            other_magnitude = _to_magnitude(other, self.force_ndarray)
        except TypeError:
            return NotImplemented
        else:
            if not self._ok_for_muldiv:
                raise OffsetUnitCalculusError(self.units)

            if isinstance(getattr(other, '_magnitude', other), ndarray):
                # arrays are refused as exponent, because they would create
                #  len(array) quantities of len(set(array)) different units
                if np.size(other) > 1:
                    raise DimensionalityError(self.units, 'dimensionless')

            new_self = self
            if other == 1:
                return self
            elif other == 0:
                units = UnitsContainer()
            else:
                if not self._is_multiplicative:
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        new_self = self.to_base_units()
                    else:
                        raise OffsetUnitCalculusError(self.units)

                if getattr(other, 'dimensionless', False):
                    units = new_self._units ** other.to_base_units().magnitude
                elif not getattr(other, 'dimensionless', True):
                    raise DimensionalityError(self.units, 'dimensionless')
                else:
                    units = new_self._units ** other

            magnitude = new_self._magnitude ** _to_magnitude(other, self.force_ndarray)
            return self.__class__(magnitude, units)

    def __rpow__(self, other):
        try:
            other_magnitude = _to_magnitude(other, self.force_ndarray)
        except TypeError:
            return NotImplemented
        else:
            if not self.dimensionless:
                raise DimensionalityError(self.units, 'dimensionless')
            if isinstance(self._magnitude, ndarray):
                if np.size(self._magnitude) > 1:
                    raise DimensionalityError(self.units, 'dimensionless')
            new_self = self.to_base_units()
            return other**new_self._magnitude

    def __abs__(self):
        return self.__class__(abs(self._magnitude), self._units)

    def __round__(self, ndigits=0):
        return self.__class__(round(self._magnitude, ndigits=ndigits), self._units)

    def __pos__(self):
        return self.__class__(operator.pos(self._magnitude), self._units)

    def __neg__(self):
        return self.__class__(operator.neg(self._magnitude), self._units)

    def __eq__(self, other):
        # We compare to the base class of Quantity because
        # each Quantity class is unique.
        if not isinstance(other, _Quantity):
            return (self.dimensionless and
                    _eq(self._convert_magnitude(UnitsContainer()), other, False))

        if _eq(self._magnitude, 0, True) and _eq(other._magnitude, 0, True):
            return self.dimensionality == other.dimensionality

        if self._units == other._units:
            return _eq(self._magnitude, other._magnitude, False)

        try:
            return _eq(self.to(other).magnitude, other._magnitude, False)
        except DimensionalityError:
            return False

    def __ne__(self, other):
        out = self.__eq__(other)
        if isinstance(out, ndarray):
            return np.logical_not(out)
        return not out

    def compare(self, other, op):
        if not isinstance(other, self.__class__):
            if self.dimensionless:
                return op(self._convert_magnitude_not_inplace(UnitsContainer()), other)
            else:
                raise ValueError('Cannot compare Quantity and {0}'.format(type(other)))

        if self.units == other.units:
            return op(self._magnitude, other._magnitude)
        if self.dimensionality != other.dimensionality:
            raise DimensionalityError(self.units, other.units,
                                      self.dimensionality, other.dimensionality)
        return op(self.to_base_units().magnitude,
                  other.to_base_units().magnitude)

    __lt__ = lambda self, other: self.compare(other, op=operator.lt)
    __le__ = lambda self, other: self.compare(other, op=operator.le)
    __ge__ = lambda self, other: self.compare(other, op=operator.ge)
    __gt__ = lambda self, other: self.compare(other, op=operator.gt)

    def __bool__(self):
        return bool(self._magnitude)

    __nonzero__ = __bool__

    # NumPy Support
    __radian = 'radian'
    __same_units = 'equal greater greater_equal less less_equal not_equal arctan2'.split()
    #: Dictionary mapping ufunc/attributes names to the units that they
    #: require (conversion will be tried).
    __require_units = {'cumprod': '',
                       'arccos': '', 'arcsin': '', 'arctan': '',
                       'arccosh': '', 'arcsinh': '', 'arctanh': '',
                       'exp': '', 'expm1': '', 'exp2': '',
                       'log': '', 'log10': '', 'log1p': '', 'log2': '',
                       'sin': __radian, 'cos': __radian, 'tan': __radian,
                       'sinh': __radian, 'cosh': __radian, 'tanh': __radian,
                       'radians': 'degree', 'degrees': __radian,
                       'deg2rad': 'degree', 'rad2deg': __radian,
                       'logaddexp': '', 'logaddexp2': ''}

    #: Dictionary mapping ufunc/attributes names to the units that they
    #: will set on output.
    __set_units = {'cos': '', 'sin': '', 'tan': '',
                   'cosh': '', 'sinh': '', 'tanh': '',
                   'arccos': __radian, 'arcsin': __radian,
                   'arctan': __radian, 'arctan2': __radian,
                   'arccosh': __radian, 'arcsinh': __radian,
                   'arctanh': __radian,
                   'degrees': 'degree', 'radians': __radian,
                   'expm1': '', 'cumprod': '',
                   'rad2deg': 'degree', 'deg2rad': __radian}

    #: List of ufunc/attributes names in which units are copied from the
    #: original.
    __copy_units = 'compress conj conjugate copy cumsum diagonal flatten ' \
                   'max mean min ptp ravel repeat reshape round ' \
                   'squeeze std sum take trace transpose ' \
                   'ceil floor hypot rint ' \
                   'add subtract ' \
                   'copysign nextafter trunc ' \
                   'frexp ldexp modf modf__1 ' \
                   'absolute negative remainder fmod mod'.split()

    #: Dictionary mapping ufunc/attributes names to the units that they will
    #: set on output. The value is interpreted as the power to which the unit
    #: will be raised.
    __prod_units = {'var': 2, 'prod': 'size', 'multiply': 'mul',
                    'true_divide': 'div', 'divide': 'div', 'floor_divide': 'div',
                    'remainder': 'div',
                    'sqrt': .5, 'square': 2, 'reciprocal': -1}

    __skip_other_args = 'ldexp multiply ' \
                        'true_divide divide floor_divide fmod mod ' \
                        'remainder'.split()

    __handled = tuple(__same_units) + \
                tuple(__require_units.keys()) + \
                tuple(__prod_units.keys()) + \
                tuple(__copy_units) + tuple(__skip_other_args)

    def clip(self, first=None, second=None, out=None, **kwargs):
        min = kwargs.get('min', first)
        max = kwargs.get('max', second)

        if min is None and max is None:
            raise TypeError('clip() takes at least 3 arguments (2 given)')

        if max is None and 'min' not in kwargs:
            min, max = max, min

        kwargs = {'out': out}

        if min is not None:
            if isinstance(min, self.__class__):
                kwargs['min'] = min.to(self).magnitude
            elif self.dimensionless:
                kwargs['min'] = min
            else:
                raise DimensionalityError('dimensionless', self.units)

        if max is not None:
            if isinstance(max, self.__class__):
                kwargs['max'] = max.to(self).magnitude
            elif self.dimensionless:
                kwargs['max'] = max
            else:
                raise DimensionalityError('dimensionless', self.units)

        return self.__class__(self.magnitude.clip(**kwargs), self._units)

    def fill(self, value):
        self._units = value.units
        return self.magnitude.fill(value.magnitude)

    def put(self, indices, values, mode='raise'):
        if isinstance(values, self.__class__):
            values = values.to(self).magnitude
        elif self.dimensionless:
            values = self.__class__(values, '').to(self)
        else:
            raise DimensionalityError('dimensionless', self.units)
        self.magnitude.put(indices, values, mode)

    @property
    def real(self):
        return self.__class__(self._magnitude.real, self.units)

    @property
    def imag(self):
        return self.__class__(self._magnitude.imag, self.units)

    @property
    def T(self):
        return self.__class__(self._magnitude.T, self.units)

    def searchsorted(self, v, side='left'):
        if isinstance(v, self.__class__):
            v = v.to(self).magnitude
        elif self.dimensionless:
            v = self.__class__(v, '').to(self)
        else:
            raise DimensionalityError('dimensionless', self.units)
        return self.magnitude.searchsorted(v, side)

    def __ito_if_needed(self, to_units):
        if self.unitless and to_units == 'radian':
            return

        self.ito(to_units)

    def __numpy_method_wrap(self, func, *args, **kwargs):
        """Convenience method to wrap on the fly numpy method taking
        care of the units.
        """
        if func.__name__ in self.__require_units:
            self.__ito_if_needed(self.__require_units[func.__name__])

        value = func(*args, **kwargs)

        if func.__name__ in self.__copy_units:
            return self.__class__(value, self._units)

        if func.__name__ in self.__prod_units:
            tmp = self.__prod_units[func.__name__]
            if tmp == 'size':
                return self.__class__(value, self._units ** self._magnitude.size)
            return self.__class__(value, self._units ** tmp)

        return value

    def __len__(self):
        return len(self._magnitude)

    def __iter__(self):
        # Allow exception to propagate in case of non-iterable magnitude
        it_mag = iter(self.magnitude)
        return iter((self.__class__(mag, self._units) for mag in it_mag))

    def __getattr__(self, item):
        # Attributes starting with `__array_` are common attributes of NumPy ndarray.
        # They are requested by numpy functions.
        if item.startswith('__array_'):
            if isinstance(self._magnitude, ndarray):
                return getattr(self._magnitude, item)
            else:
                # If an `__array_` attributes is requested but the magnitude is not an ndarray,
                # we convert the magnitude to a numpy ndarray.
                self._magnitude = _to_magnitude(self._magnitude, force_ndarray=True)
                return getattr(self._magnitude, item)
        elif item in self.__handled:
            if not isinstance(self._magnitude, ndarray):
                self._magnitude = _to_magnitude(self._magnitude, True)
            attr = getattr(self._magnitude, item)
            if callable(attr):
                return functools.partial(self.__numpy_method_wrap, attr)
            return attr
        try:
            return getattr(self._magnitude, item)
        except AttributeError as ex:
            raise AttributeError("Neither Quantity object nor its magnitude ({0})"
                                 "has attribute '{1}'".format(self._magnitude, item))

    def __getitem__(self, key):
        try:
            value = self._magnitude[key]
            return self.__class__(value, self._units)
        except TypeError:
            raise TypeError("Neither Quantity object nor its magnitude ({0})"
                            "supports indexing".format(self._magnitude))

    def __setitem__(self, key, value):
        try:
            if math.isnan(value):
                self._magnitude[key] = value
                return
        except (TypeError, DimensionalityError):
            pass

        try:
            if isinstance(value, self.__class__):
                factor = self.__class__(value.magnitude, value.units / self.units).to_base_units()
            else:
                factor = self.__class__(value, self._units ** (-1)).to_base_units()

            if isinstance(factor, self.__class__):
                if not factor.dimensionless:
                    raise DimensionalityError(value, self.units,
                                              extra_msg='. Assign a quantity with the same dimensionality or '
                                                        'access the magnitude directly as '
                                                        '`obj.magnitude[%s] = %s`' % (key, value))
                self._magnitude[key] = factor.magnitude
            else:
                self._magnitude[key] = factor

        except TypeError:
            raise TypeError("Neither Quantity object nor its magnitude ({0})"
                            "supports indexing".format(self._magnitude))

    def tolist(self):
        units = self._units
        return [self.__class__(value, units).tolist() if isinstance(value, list) else self.__class__(value, units)
                for value in self._magnitude.tolist()]

    __array_priority__ = 17

    def __array_prepare__(self, obj, context=None):
        # If this uf is handled by Pint, write it down in the handling dictionary.

        # name of the ufunc, argument of the ufunc, domain of the ufunc
        # In ufuncs with multiple outputs, domain indicates which output
        # is currently being prepared (eg. see modf).
        # In ufuncs with a single output, domain is 0
        uf, objs, huh = context

        if uf.__name__ in self.__handled and huh == 0:
            # Only one ufunc should be handled at a time.
            # If a ufunc is already being handled (and this is not another domain),
            # something is wrong..
            if self.__handling:
                raise Exception('Cannot handled nested ufuncs.\n'
                                'Current: {0}\n'
                                'New: {1}'.format(context, self.__handling))
            self.__handling = context

        return obj

    def __array_wrap__(self, obj, context=None):
        uf, objs, huh = context

        # if this ufunc is not handled by Pint, pass it to the magnitude.
        if uf.__name__ not in self.__handled:
            return self.magnitude.__array_wrap__(obj, context)

        try:
            ufname = uf.__name__ if huh == 0 else '{0}__{1}'.format(uf.__name__, huh)

            # First, we check the units of the input arguments.

            if huh == 0:
                # Do this only when the wrap is called for the first ouput.

                # Store the destination units
                dst_units = None
                # List of magnitudes of Quantities with the right units
                # to be used as argument of the ufunc
                mobjs = None

                if uf.__name__ in self.__require_units:
                    # ufuncs in __require_units
                    # require specific units
                    # This is more complex that it should be due to automatic
                    # conversion between radians/dimensionless
                    # TODO: maybe could be simplified using Contexts
                    dst_units = self.__require_units[uf.__name__]
                    if dst_units == 'radian':
                        mobjs = []
                        for other in objs:
                            unt = getattr(other, 'units', '')
                            if unt == 'radian':
                                mobjs.append(getattr(other, 'magnitude', other))
                            else:
                                factor, units = self._REGISTRY.get_base_units(unt)
                                if units and units != UnitsContainer({'radian': 1}):
                                    raise DimensionalityError(units, dst_units)
                                mobjs.append(getattr(other, 'magnitude', other) * factor)
                        mobjs = tuple(mobjs)
                    else:
                        dst_units = self._REGISTRY.parse_expression(dst_units).units

                elif len(objs) > 1 and uf.__name__ not in self.__skip_other_args:
                    # ufunc with multiple arguments require that all inputs have
                    # the same arguments unless they are in __skip_other_args
                    dst_units = objs[0].units

                # Do the conversion (if needed) and extract the magnitude for each input.
                if mobjs is None:
                    if dst_units is not None:
                        mobjs = tuple(self._REGISTRY.convert(getattr(other, 'magnitude', other),
                                                             getattr(other, 'units', ''),
                                                             dst_units)
                                      for other in objs)
                    else:
                        mobjs = tuple(getattr(other, 'magnitude', other)
                                      for other in objs)

                # call the ufunc
                out = uf(*mobjs)

                # If there are multiple outputs,
                # store them in __handling (uf, objs, huh, out0, out1, ...)
                # and return the first
                if uf.nout > 1:
                    self.__handling += out
                    out = out[0]
            else:
                # If this is not the first output,
                # just grab the result that was previously calculated.
                out = self.__handling[3 + huh]


            # Second, we set the units of the output value.
            if ufname in self.__set_units:
                try:
                    out = self.__class__(out, self.__set_units[ufname])
                except:
                    raise _Exception(ValueError)
            elif ufname in self.__copy_units:
                try:
                    out = self.__class__(out, self.units)
                except:
                    raise _Exception(ValueError)
            elif ufname in self.__prod_units:
                tmp = self.__prod_units[ufname]
                if tmp == 'size':
                    out = self.__class__(out, self.units ** self._magnitude.size)
                elif tmp == 'div':
                    units1 = objs[0].units if isinstance(objs[0], self.__class__) else UnitsContainer()
                    units2 = objs[1].units if isinstance(objs[1], self.__class__) else UnitsContainer()
                    out = self.__class__(out, units1 / units2)
                elif tmp == 'mul':
                    units1 = objs[0].units if isinstance(objs[0], self.__class__) else UnitsContainer()
                    units2 = objs[1].units if isinstance(objs[1], self.__class__) else UnitsContainer()
                    out = self.__class__(out, units1 * units2)
                else:
                    out = self.__class__(out, self.units ** tmp)

            return out
        except (DimensionalityError, UndefinedUnitError) as ex:
            raise ex
        except _Exception as ex:
            raise ex.internal
        except Exception as ex:
            print(ex)
        finally:
            # If this is the last output argument for the ufunc,
            # we are done handling this ufunc.
            if uf.nout == huh + 1:
                self.__handling = None

        return self.magnitude.__array_wrap__(obj, context)

    # Measurement support
    def plus_minus(self, error, relative=False):
        if isinstance(error, self.__class__):
            if relative:
                raise ValueError('{} is not a valid relative error.'.format(error))
            error = error.to(self.units).magnitude
        else:
            if relative:
                error = error * abs(self.magnitude)

        return self._REGISTRY.Measurement(copy.copy(self.magnitude), error, self.units)

    # methods/properties that help for math operations with offset units
    @property
    def _is_multiplicative(self):
        """Check if the Quantity object has only multiplicative units.
        """
        # XXX Turn this into a method/property of _Quantity?
        return not self._get_non_multiplicative_units()

    def _get_non_multiplicative_units(self):
        """Return a list of the of non-multiplicative units of the Quantity object
        """
        offset_units = [unit for unit in self.units.keys()
                        if not self._REGISTRY._units[unit].is_multiplicative]
        return offset_units

    def _get_delta_units(self):
        """Return list of delta units ot the Quantity object
        """
        delta_units = [u for u in self.units.keys() if u.startswith("delta_")]
        return delta_units

    def _has_compatible_delta(self, unit):
        """"Check if Quantity object has a delta_unit that is compatible with unit
        """
        deltas = self._get_delta_units()
        if 'delta_' + unit in deltas:
            return True
        else:  # Look for delta units with same dimension as the offset unit
            offset_unit_dim = self._REGISTRY._units[unit].reference
            for d in deltas:
                if self._REGISTRY._units[d].reference == offset_unit_dim:
                    return True
        return False

    def _ok_for_muldiv(self, no_offset_units=None):
        """Checks if Quantity object can be multiplied or divided

        :q: quantity object that is checked
        :no_offset_units: number of offset units in q
        """
        is_ok = True
        if no_offset_units is None:
            no_offset_units = len(self._get_non_multiplicative_units())
        if no_offset_units > 1:
            is_ok = False
        if no_offset_units == 1:
            if len(self._units) > 1:
                is_ok = False
            if (len(self._units) == 1
                    and not self._REGISTRY.autoconvert_offset_to_baseunit):
                is_ok = False
            if next(iter(self._units.values())) != 1:
                is_ok = False
        return is_ok
