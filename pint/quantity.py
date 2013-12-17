# -*- coding: utf-8 -*-
"""
    pint.quantity
    ~~~~~~~~~~~~~

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import copy
import operator
import functools
from collections import Iterable

from .unit import DimensionalityError, UnitsContainer, UnitDefinition, UndefinedUnitError
from .measurement import Measurement
from .util import string_types, NUMERIC_TYPES, ndarray

try:
    import numpy as np

    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise ValueError('Invalid magnitude for Quantity: {!r}'.format(value))
        elif value == '':
            raise ValueError('Quantity magnitude cannot be an empty string.')
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
        if force_ndarray:
            return np.asarray(value)
        return value

except ImportError:
    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise ValueError('Invalid magnitude for Quantity: {!r}'.format(value))
        elif value == '':
            raise ValueError('Quantity magnitude cannot be an empty string.')
        elif isinstance(value, (list, tuple)):
            raise ValueError('lists and tuples are valid magnitudes for '
                             'Quantity only when NumPy is present.')
        return value


def _eq(first, second, check_all):
    """Comparison of scalars and arrays
    """
    out = first == second
    if check_all and isinstance(out, Iterable):
        if isinstance(out, ndarray):
            return np.all(out)
        else:
            return all(out)
    return out


class _Exception(Exception):

    def __init__(self, internal):
        self.internal = internal


def _check(q1, other):
    """Check Quantities before math operations.

    Return True if q1 and other are from the same class.
    Raise a ValueError if other has a different _REGISTRY than q1.

    In other case, return False.
    """
    if isinstance(other, q1.__class__):
        return True
    try:
        reg = other._REGISTRY
    except AttributeError:
        return False

    if q1._REGISTRY is reg:
        return True

    raise ValueError('Cannot operate between quantities of different registries')


class _Quantity(object):
    """Implements a class to describe a physical quantities:
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
                    raise ValueError('Quantity magnitude cannot be an empty string.')
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
            inst = copy.copy(units)
            inst._magnitude = _to_magnitude(value, inst.force_ndarray)
        else:
            raise TypeError('units must be of type str, Quantity or '
                            'UnitsContainer; not {}.'.format(type(units)))

        inst.__handling = None
        return inst

    def __copy__(self):
        return self.__class__(copy.copy(self._magnitude), copy.copy(self._units))

    def __str__(self):
        return '{} {}'.format(self._magnitude, self._units)

    def __repr__(self):
        return "<Quantity({}, '{}')>".format(self._magnitude, self._units)

    def __format__(self, spec):
        spec = spec or self.default_format

        if '~' in spec:
            units = UnitsContainer({self._REGISTRY.get_symbol(key): value
                                   for key, value in self.units.items()})
            spec = spec.replace('~', '')
        else:
            units = self.units

        return format(self.magnitude, spec.replace('L', '').replace('P', '').replace('H', '')) \
               + ' ' + format(units, spec)

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

    def _convert_magnitude(self, other, *contexts, **ctx_kwargs):
        if contexts:
            with self._REGISTRY.context(*contexts, **ctx_kwargs):
                return self._REGISTRY.convert(self._magnitude, self._units, other)

        return self._REGISTRY.convert(self._magnitude, self._units, other)

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
        return self

    def to(self, other=None, *contexts, **ctx_kwargs):
        """Return Quantity rescaled to different units.

        :param other: destination units.
        :type other: Quantity, str or dict
        """
        ret = copy.copy(self)
        ret.ito(other, *contexts, **ctx_kwargs)
        return ret

    def ito_base_units(self):
        """Return Quantity rescaled to base units
        """

        _, other = self._REGISTRY.get_base_units(self.units)

        self._magnitude = self._convert_magnitude(other)
        self._units = other
        return self

    def to_base_units(self):
        """Return Quantity rescaled to base units
        """

        ret = copy.copy(self)
        ret.ito_base_units()
        return ret

    # Mathematical operations
    def __float__(self):
        if self.dimensionless:
            return float(self._convert_magnitude(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def __complex__(self):
        if self.dimensionless:
            return complex(self._convert_magnitude(UnitsContainer()))
        raise DimensionalityError(self.units, 'dimensionless')

    def iadd_sub(self, other, op):
        if _check(self, other):
            if not self.dimensionality == other.dimensionality:
                raise DimensionalityError(self.units, other.units,
                                          self.dimensionality, other.dimensionality)
            if self._units == other._units:
                self._magnitude = op(self._magnitude, other._magnitude)
            else:
                self._magnitude = op(self._magnitude, other.to(self)._magnitude)
        else:
            if self.dimensionless:
                self.ito(UnitsContainer())
                self._magnitude = op(self._magnitude, _to_magnitude(other, self.force_ndarray))
            else:
                raise DimensionalityError(self.units, 'dimensionless')

        return self

    def add_sub(self, other, op):
        ret = copy.copy(self)
        op(ret, other)
        return ret

    def __iadd__(self, other):
        return self.iadd_sub(other, operator.iadd)

    def __add__(self, other):
        return self.add_sub(other, operator.iadd)

    __radd__ = __add__

    def __isub__(self, other):
        return self.iadd_sub(other, operator.isub)

    def __sub__(self, other):
        return self.add_sub(other, operator.isub)

    def __rsub__(self, other):
        return -self.add_sub(other, operator.isub)

    def __imul__(self, other):
        if _check(self, other):
            self._magnitude *= other._magnitude
            self._units *= other._units
        else:
            self._magnitude *= _to_magnitude(other, self.force_ndarray)

        return self

    def __mul__(self, other):
        if _check(self, other):
            return self.__class__(self._magnitude * other._magnitude, self._units * other._units)
        else:
            return self.__class__(self._magnitude * other, self._units)

    __rmul__ = __mul__

    def __itruediv__(self, other):
        if _check(self, other):
            self._magnitude /= other._magnitude
            self._units /= other._units
        else:
            self._magnitude /= _to_magnitude(other, self.force_ndarray)

        return self

    def __truediv__(self, other):
        ret = copy.copy(self)
        ret /= other
        return ret

    def __rtruediv__(self, other):
        if isinstance(other, NUMERIC_TYPES):
            return self.__class__(other / self._magnitude, 1 / self._units)
        raise NotImplementedError

    def __ifloordiv__(self, other):
        if _check(self, other):
            self._magnitude //= other._magnitude
            self._units /= other._units
        else:
            self._magnitude //= _to_magnitude(other, self.force_ndarray)

        return self

    def __floordiv__(self, other):
        ret = copy.copy(self)
        ret //= other
        return ret

    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __idiv__ = __itruediv__

    def __rfloordiv__(self, other):
        if _check(self, other):
            return self.__class__(other._magnitude // self._magnitude, other._units / self._units)
        else:
            return self.__class__(other // self._magnitude, 1.0 / self._units)

    def __ipow__(self, other):
        self._magnitude **= _to_magnitude(other, self.force_ndarray)
        self._units **= other
        return self

    def __pow__(self, other):
        ret = copy.copy(self)
        ret **= other
        return ret

    def __abs__(self):
        return self.__class__(abs(self._magnitude), self._units)

    def __round__(self, ndigits=0):
        return self.__class__(round(self._magnitude, ndigits=ndigits), self._units)

    def __pos__(self):
        return self.__class__(operator.pos(self._magnitude), self._units)

    def __neg__(self):
        return self.__class__(operator.neg(self._magnitude), self._units)

    def __eq__(self, other):
        # This is class comparison by name is to bypass that
        # each Quantity class is unique.
        if other.__class__.__name__ != self.__class__.__name__:
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
                return op(self._convert_magnitude(UnitsContainer()), other)
            else:
                raise ValueError('Cannot compare Quantity and {}'.format(type(other)))

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
        if item.startswith('__array_'):
            if isinstance(self._magnitude, ndarray):
                try:
                    return getattr(self._magnitude, item)
                except AttributeError:
                    return getattr(_to_magnitude(self._magnitude, True), item)
            else:
                return getattr(_to_magnitude(self._magnitude, True), item)
        try:
            try:
                attr = getattr(self._magnitude, item)
            except AttributeError:
                attr = getattr(_to_magnitude(self._magnitude, True), item)
            if callable(attr):
                return functools.partial(self.__numpy_method_wrap, attr)
            return attr
        except AttributeError as ex:
            raise AttributeError("Neither Quantity object nor its magnitude ({})"
                                 "has attribute '{}'".format(self._magnitude, item))

    def __getitem__(self, key):
        try:
            value = self._magnitude[key]
            return self.__class__(value, self._units)
        except TypeError:
            raise TypeError("Neither Quantity object nor its magnitude ({})"
                            "supports indexing".format(self._magnitude))

    def __setitem__(self, key, value):
        try:
            if isinstance(value, self.__class__):
                factor = self.__class__(value.magnitude, value.units / self.units).to_base_units()
            else:
                factor = self.__class__(value, self._units ** (-1)).to_base_units()

            if isinstance(factor, self.__class__):
                if not factor.dimensionless:
                    raise ValueError
                self._magnitude[key] = factor.magnitude
            else:
                self._magnitude[key] = factor

        except TypeError:
            raise TypeError("Neither Quantity object nor its magnitude ({})"
                            "supports indexing".format(self._magnitude))

    def tolist(self):
        units = self._units
        return [self.__class__(value, units).tolist() if isinstance(value, list) else self.__class__(value, units)
                for value in self._magnitude.tolist()]

    __array_priority__ = 17

    def __array_prepare__(self, obj, context=None):
        # If this uf is handled by Pint, write it down in the handling dictionary.

        uf, objs, huh = context
        ufname = uf.__name__ if huh == 0 else '{}__{}'.format(uf.__name__, huh)
        if uf.__name__ in self.__handled and huh == 0:
            if self.__handling:
                raise Exception('Cannot handled nested ufuncs.\n'
                                'Current: {}\n'
                                'New: {}'.format(context, self.__handling))
            self.__handling = context

        return obj

    def __array_wrap__(self, obj, context=None):
        uf, objs, huh = context

        if uf.__name__ not in self.__handled:
            return self.magnitude.__array_wrap__(obj, context)

        try:
            ufname = uf.__name__ if huh == 0 else '{}__{}'.format(uf.__name__, huh)

            if huh == 0:
                dst_units = None
                mobjs = None
                if uf.__name__ in self.__require_units:
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
                    dst_units = objs[0].units

                if mobjs is None:
                    if dst_units is not None:
                        mobjs = tuple(self._REGISTRY.convert(getattr(other, 'magnitude', other),
                                                             getattr(other, 'units', ''),
                                                             dst_units)
                                      for other in objs)
                    else:
                        mobjs = tuple(getattr(other, 'magnitude', other)
                                      for other in objs)

                out = uf(*mobjs)

                if uf.nout > 1:
                    self.__handling += out
                    out = out[0]
            else:
                out = self.__handling[3 + huh]

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
            if uf.nout == huh + 1:
                self.__handling = None

        return self.magnitude.__array_wrap__(obj, context)

    # Measurement support
    def plus_minus(self, error, relative=False):
        if isinstance(error, self.__class__):
            if relative:
                raise ValueError('{} is not a valid relative error.'.format(error))
        else:
            if relative:
                error = error * abs(self)
            else:
                error = self.__class__(error, self.units)

        return Measurement(copy.copy(self), error)
