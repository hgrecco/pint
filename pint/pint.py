# -*- coding: utf-8 -*-
"""
    pint
    ~~~~

    Pint is Python module/package to define, operate and manipulate
    **physical quantities**: the product of a numerical value and a
    unit of measurement. It allows arithmetic operations between them
    and conversions from and to different units.

    :copyright: 2012 by Hernan E. Grecco.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

__version__ = '0.1'

import os
import sys
import copy
import math
import logging
import operator
import functools

from io import BytesIO
from numbers import Number
import tokenize
from tokenize import untokenize, NUMBER, STRING, NAME, OP

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if sys.version < '3':
    import codecs
    from io import open
    from StringIO import StringIO
    string_types = basestring
    _tokenize = lambda input: tokenize.generate_tokens(StringIO(input).readline)
else:
    string_types = str
    _tokenize = lambda input: tokenize.tokenize(BytesIO(input.encode('utf-8')).readline)

PRETTY = '⁰¹²³⁴⁵⁶⁷⁸⁹·⁻'

def _definitions_from_file(filename):
    with open(filename, encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                if ';' in line:
                    [definition, modifiers] = line.split(';', 2)
                    modifiers = (modifier.split('=') for modifier in modifiers.split(';'))
                    modifiers = dict((key.strip(), float(value.strip())) for key, value in modifiers)
                else:
                    definition = line
                    modifiers = {}
                result = [res.strip() for res in definition.split('=')]
                name, value, aliases = result[0], result[1], result[2:]
            except Exception as ex:
                logger.error("Exception: Cannot parse '{}' {}".format(line, ex))
                continue
            yield name, value, aliases, modifiers


def _solve_dependencies(dependencies):
    """Solve a dependency graph.

    :param dependencies: dependency dictionary. For each key, the value is
                         an iterable indicating its dependencies.
    :return: list of sets, each containing keys of independents tasks dependent

    """
    d = dict((key, set(dependencies[key])) for key in dependencies)
    r = []
    while d:
        # values not in keys (items without dep)
        t = set(i for v in d.values() for i in v) - set(d.keys())
        # and keys without value (items without dep)
        t.update(k for k, v in d.items() if not v)
        # can be done right away
        r.append(t)
        # and cleaned up
        d = dict(((k, v - t) for k, v in d.items() if v))
    return r


class UndefinedUnitError(ValueError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, unit_names):
        super(ValueError, self).__init__()
        self.unit_names = unit_names

    def __str__(self):
        if isinstance(self.unit_names, string_types):
            return "'{}' is not defined in the unit registry.".format(self.unit_names)
        else:
            return '{} are not defined in the unit registry.'.format(self.unit_names)


class DimensionalityError(ValueError):
    """Raised when trying to convert between incompatible units.
    """

    def __init__(self, units1, units2, dim1=None, dim2=None):
        super(DimensionalityError, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2

    def __str__(self):
        if self.dim1 or self.dim2:
            dim1 = ' ({})'.format(self.dim1)
            dim2 = ' ({})'.format(self.dim2)
        else:
            dim1 = ''
            dim2 = ''
        return "Cannot convert from '{}'{} to '{}'{}.".format(self.units1, dim1, self.units2, dim2)


class AliasDict(dict):

    def add_alias(self, key, value):
        if value not in self:
            raise IndexError("The aliased value '{}' is not present in the dictionary".format(value))
        self[key] = self.get_aliased(value)

    def get_aliased(self, key):
        value = self[key]
        if isinstance(value, string_types):
            return self.get_aliased(value)
        return key


class UnitsContainer(dict):
    """The UnitsContainer stores the product of units and their respective
    exponent and implements the corresponding operations
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        for key, value in self.items():
            if not isinstance(key, string_types):
                raise TypeError('key must be a str, not {}'.format(type(key)))
            if not isinstance(value, Number):
                raise TypeError('value must be a Number, not {}'.format(type(value)))
            if not isinstance(value, float):
                self[key] = float(value)

    def __missing__(self, key):
        return 0.0

    def __setitem__(self, key, value):
        if not isinstance(key, string_types):
            raise TypeError('key must be a str, not {}'.format(type(key)))
        if not isinstance(value, Number):
            raise TypeError('value must be a Number, not {}'.format(type(value)))
        dict.__setitem__(self, key, float(value))

    def _formatter(self, product_sign=' * ', superscript_format=' ** {:n}',
                  as_ratio=True, single_denominator=False):
        if not self:
            return 'dimensionless'

        if as_ratio:
            fun = abs
        else:
            fun = lambda x: x

        tmp_plus = []
        tmp_minus = []
        for key, value in sorted(self.items()):
            if value == 1:
                tmp_plus.append(key)
            elif value > 1:
                tmp_plus.append(key + superscript_format.format(value))
            elif value == -1:
                tmp_minus.append(key)
            else:
                tmp_minus.append(key + superscript_format.format(fun(value)))

        if tmp_plus:
            ret = product_sign.join(tmp_plus)
        elif as_ratio:
            ret = '1'
        else:
            ret = ''

        if tmp_minus:
            if as_ratio:
                ret += ' / '
                if single_denominator:
                   ret += ' / '.join(tmp_minus)
                else:
                   ret += product_sign.join(tmp_minus)
            else:
                ret += product_sign.join(tmp_minus)

        return ret

    def __str__(self):
        return self._formatter()

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value) for key, value in sorted(self.items())])
        return '<UnitsContainer({})>'.format(tmp)

    def __format__(self, spec):
        if spec == '!s' or spec == '':
            return str(self)
        elif spec == '!r':
            return repr(self)
        elif spec == '!l':
            tmp = self._formatter(r' \cdot ', '^[{:n}]', True, True).replace('[', '{').replace(']', '}')
            if '/' in tmp:
                return r'\frac{%s}' % tmp.replace(' / ', '}{')
        elif spec == '!p':
            pretty = '{}'.format(self).replace(' ** ', '').replace(' * ', PRETTY[10]).replace('-', PRETTY[11]).replace(' / ', '/')
            for n in range(10):
                pretty = pretty.replace(str(n), PRETTY[n])
            return pretty
        else:
            raise ValueError('{} is not a valid format for UnitsContainer'.format(spec))

    def __copy__(self):
        ret = self.__class__()
        ret.update(self)
        return ret

    def __imul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot multiply UnitsContainer by {}'.format(type(other)))
        for key, value in other.items():
            self[key] += value
        keys = [key for key, value in self.items() if value == 0]
        for key in keys:
            del self[key]

        return self

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot multiply UnitsContainer by {}'.format(type(other)))
        ret = copy.copy(self)
        ret *= other
        return ret

    __rmul__ = __mul__

    def __ipow__(self, other):
        if not isinstance(other, Number):
            raise TypeError('Cannot power UnitsContainer by {}'.format(type(other)))
        for key, value in self.items():
            self[key] *= other
        return self

    def __pow__(self, other):
        if not isinstance(other, Number):
            raise TypeError('Cannot power UnitsContainer by {}'.format(type(other)))
        ret = copy.copy(self)
        ret **= other
        return ret

    def __itruediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot divide UnitsContainer by {}'.format(type(other)))

        for key, value in other.items():
            self[key] -= value

        keys = [key for key, value in self.items() if value == 0]
        for key in keys:
            del self[key]

        return self

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot divide UnitsContainer by {}'.format(type(other)))

        ret = copy.copy(self)
        ret /= other
        return ret

    def __rtruediv__(self, other):
        if not isinstance(other, self.__class__) and other != 1:
            raise TypeError('Cannot divide {} by UnitsContainer'.format(type(other)))

        ret = copy.copy(self)
        ret **= -1
        return ret


def converter_to_reference(scale, offset, log_base):
    def _inner(value):
        if log_base:
            return log_base ** (value / scale + offset)
        else:
            return value * scale + offset
    return _inner

def converter_from_reference(scale, offset, log_base):
    def _inner(value):
        if log_base:
            return (math.log10(value) / math.log10(log_base) - offset) / scale
        else:
            return (value - offset) / scale
    return _inner


class UnitRegistry(object):
    """The unit registry stores the definitions and relationships between
    units.

    :param filename: path of the units definition file to load.
                     Empty to load the default definition file.
                     None to leave the UnitRegistry empty.
    """

    #: Map unit name (string) to unit value (Quantity), and unit alias to canonical unit name
    _UNITS = AliasDict()

    #: Map prefix name (string) to prefix value (float), and unit alias to canonical prefix name
    _PREFIXES = AliasDict({'': 1})

    #: Map suffix name (string) to canonical , and unit alias to canonical unit name
    _SUFFIXES = AliasDict({'': None, 's': ''})

    def __init__(self, filename=''):
        self.Quantity._REGISTRY = self
        self._definition_files = []
        if filename == '':
            self.add_from_file(os.path.join(os.path.dirname(__file__), 'default_en.txt'))
        elif filename is not None:
            self.add_from_file(filename)

    def __getattr__(self, item):
        return self.Quantity(1, self._to_canonical(item))

    def __getitem__(self, item):
        return self.Quantity(1, self._parse_expression(item))

    def add_unit(self, name, value, aliases=tuple(), **modifiers):
        """Add unit to the registry.
        """
        if not isinstance(value, self.Quantity):
            value = self.Quantity(value, **modifiers)

        self._UNITS[name] = value

        for alias in aliases:
            if ' ' in alias:
                logger.warn('Alias cannot contain a space ' + alias)
            self._UNITS.add_alias(alias.strip(), name)

    def add_prefix(self, name, value, aliases=tuple()):
        """Add prefix to the registry.
        """

        if not isinstance(value, Number):
            value = eval(value, {'__builtins__': None}, {})
        self._PREFIXES[name] = float(value)

        for alias in aliases:
            self._PREFIXES.add_alias(alias.strip(), name)

    def add_from_file(self, filename):
        """Add units and prefixes defined in a definition text file.
        """
        self._definition_files.append(filename)
        pending = dict()
        dependencies = dict()
        conv = dict()
        for name, value, aliases, modifiers in _definitions_from_file(filename):
            try:
                if name.endswith('-'):
                    # Prefix
                    self.add_prefix(name[:-1], value, [alias[:-1] for alias in aliases])
                    continue
                if '[' in value:
                    # Reference units, indicates dimensionality
                    value = value.strip('[]')
                    if value:
                        value = self.Quantity(None, UnitsContainer({value: 1}))
                    else:
                        value = self.Quantity(None, None)

                conv[name] = name
                conv.update({alias: name for alias in aliases})
                self.add_unit(name, value, aliases, **modifiers)
                if modifiers:
                    self.add_unit('delta_' + name, value, tuple('delta_' + item for item in aliases))
            except UndefinedUnitError as ex:
                pending[name] = (value, aliases)
                dependencies[name] = ex.unit_names
            except Exception as ex:
                logger.error("Exception: Cannot add '{}' {}".format(line, ex))

        dep2 = {}
        for unit_name, deps in dependencies.items():
            dep2[unit_name] = set(conv[dep_name] for dep_name in deps)

        for unit_names in _solve_dependencies(dep2):
            for unit_name in unit_names:
                if not unit_name in self._UNITS:
                    self.add_unit(unit_name, *pending[unit_name])

    def _to_canonical(self, candidate):
        try:
            return self._UNITS.get_aliased(candidate)
        except KeyError:
            pass

        candidates = tuple(self._parse_candidate(candidate))
        if not candidates:
            raise UndefinedUnitError(candidate)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning('Parsing {} yield multiple results. Options are: {}'.format(candidate, candidates))
            prefix, unit_name, _ = candidates[0]

        if prefix:
            self._UNITS[prefix + unit_name] = self.Quantity(self._PREFIXES[prefix], unit_name)
            return prefix + unit_name

        return unit_name

    def _parse_candidate(self, candidate):

        for unit_name in self._UNITS.keys():
            if unit_name in candidate:
                try:
                    [prefix, suffix] = candidate.split(unit_name)
                    if len(unit_name) == 1 and len(suffix) == 1:
                        continue
                except ValueError: # too many values to unpack
                    continue
                if prefix in self._PREFIXES and suffix in self._SUFFIXES:
                    yield (self._PREFIXES.get_aliased(prefix),
                           self._UNITS.get_aliased(unit_name),
                           self._SUFFIXES.get_aliased(suffix))

    def _parse_expression(self, input):
        gen = _tokenize(input)
        result = []
        unknown = set()
        for toknum, tokval, _, _, _  in gen:
            if toknum in (STRING, NAME):  # replace NUMBER tokens
                # TODO: Integrate math better
                if tokval == 'pi':
                    result.append((toknum, str(math.pi)))
                    continue
                try:
                    tokval = self._to_canonical(tokval)
                except UndefinedUnitError as ex:
                    unknown.add(ex.unit_names)

                result.extend([
                    (NAME, 'Quantity'),
                    (OP, '('),
                    (NUMBER, '1'),
                    (OP, ','),
                    (NAME, 'U_'),
                    (OP, '('),
                    (STRING, tokval),
                    (OP, '='),
                    (NUMBER, '1'),
                    (OP, ')'),
                    (OP, ')')
                ])
            else:
                result.append((toknum, tokval))

        if unknown:
            raise UndefinedUnitError(unknown)

        return eval(untokenize(result), {'__builtins__': None},
                                        {'REGISTRY': self._UNITS,
                                         'Quantity': self.Quantity,
                                         'U_': UnitsContainer})


    @functools.total_ordering
    class Quantity(object):
        """Quantity object constituted by magnitude and units.

        :param value: value of the physical quantity to be created.
        :type value: str, Quantity or any numeric type.
        :param units: units of the physical quantity to be created.
        :type units: UnitsContainer, str or Quantity.

        """

        #: Unit registry containing this class.
        _REGISTRY = None

        def __new__(cls, value, units=None, offset=0, log_base=0):
            if units is None:
                if isinstance(value, string_types):
                    inst = cls._REGISTRY._parse_expression(value)
                elif isinstance(value, cls):
                    inst = copy.copy(value)
                else:
                    inst = object.__new__(cls)
                    inst._magnitude = value
                    inst._units = UnitsContainer()
            elif isinstance(units, UnitsContainer):
                inst = object.__new__(cls)
                inst._magnitude = value
                inst._units = units
            elif isinstance(units, string_types):
                inst = cls._REGISTRY._parse_expression(units)
                inst._magnitude = value
            elif isinstance(units, cls):
                inst = copy.copy(units)
                inst._magnitude = value
            else:
                raise TypeError('units must be of type str, Quantity or UnitsContainer; not {}.'.format(type(units)))

            # This only works if expressed as reference
            inst.multiplicative = offset or log_base
            scale = 1 / inst.magnitude if inst.magnitude else 1
            inst._convert_to_reference = converter_to_reference(scale, offset, log_base)
            inst._convert_from_reference = converter_from_reference(scale, offset, log_base)

            return inst

        def __copy__(self):
            return self.__class__(self._magnitude, self._units)

        def __str__(self):
            return '{} {}'.format(self._magnitude, self._units)

        def __repr__(self):
            return "<Quantity({}, '{}')>".format(self._magnitude, self._units)

        def __format__(self, spec):
            if not spec:
                return str(self)
            if '!' in spec:
                fmt, conv = spec.split('!')
                conv = '!' + conv
            else:
                fmt, conv = spec, ''
            return format(self.magnitude, fmt) + ' ' + format(self.units, conv)

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
            return not bool(self.convert_to_reference().units)

        @property
        def dimensionless(self):
            """Return true if the quantity is dimensionless.
            """
            tmp = self.convert_to_reference()

            return not bool(self.convert_to_reference().dimensionality)

        @property
        def dimensionality(self):
            """Quantity's dimensionality (e.g. {length: 1, time: -1})
            """
            try:
                return self._dimensionality
            except AttributeError:
                if self._magnitude is None:
                    return UnitsContainer(self.units)

                tmp = UnitsContainer()
                for key, value in self.units.items():
                    reg = self._REGISTRY._UNITS[key]
                    tmp = tmp * reg.dimensionality ** value

                self._dimensionality = tmp

            return self._dimensionality

        def ito(self, other=None):
            """Inplace rescale to different units.

            :param other: destination units.
            :type other: Quantity or str.
            """
            if isinstance(other, string_types):
                other = self._REGISTRY._parse_expression(other)

            if self._units == other._units:
                return self.__class__(self._magnitude, other)

            if len(self.units) == 1:
                unit, power = tuple(self.units.items())[0]
                ounit, opower = tuple(other.units.items())[0]
                if self.dimensionality != other.dimensionality:
                    raise DimensionalityError(self.units, other.units,
                                              self.dimensionality, other.dimensionality)

                unit = self._REGISTRY._UNITS[unit]
                ounit = self._REGISTRY._UNITS[ounit]
                if unit.multiplicative or ounit.multiplicative:
                    value = unit._convert_to_reference(self.magnitude)
                    value = ounit._convert_from_reference(value)

                    self._magnitude = value
                    self._units = copy.copy(other._units)
                    return self

            factor = self.__class__(1, self.units / other.units)
            factor =  factor.convert_to_reference()
            if not factor.unitless:
                raise DimensionalityError(self.units, other.units,
                                          self.dimensionality, other.dimensionality)

            self._magnitude *= factor.magnitude
            self._units = copy.copy(other._units)
            return self

        def to(self, other=None):
            """Return Quantity rescaled to different units.

            :param other: destination units.
            :type other: Quantity or str.
            """
            ret = copy.copy(self)
            ret.ito(other)
            return ret

        def convert_to_reference(self):
            """Return Quantity rescaled to reference units.

            """

            tmp = self._REGISTRY.Quantity(self.magnitude)
            for key, value in self.units.items():
                reg = self._REGISTRY._UNITS[key]
                if reg._magnitude is None:
                    factor = self.__class__(1, key) ** value
                else:
                    factor = reg.convert_to_reference() ** value

                tmp *= factor

            return tmp

        # Mathematical operations
        def __float__(self):
            if self.dimensionless:
                return float(self._magnitude)
            raise DimensionalityError(self.units, 'dimensionless')

        def __complex__(self):
            if self.dimensionless:
                return complex(self._magnitude)
            raise DimensionalityError(self.units, 'dimensionless')

        def iadd_sub(self, other, fun):
            if isinstance(other, self.__class__):
                if not self.dimensionality == other.dimensionality:
                    raise DimensionalityError(self.units, other.units,
                                              self.dimensionality, other.dimensionality)
                if self._units == other._units:
                    self._magnitude = fun(self._magnitude, other._magnitude)
                else:
                    self._magnitude = fun(self._magnitude, other.to(self)._magnitude)
            else:
                if self.unitless:
                    self._magnitude = fun(self._magnitude, other)
                else:
                    raise DimensionalityError(self.units, 'dimensionless')

            return self

        def add_sub(self, other, fun):
            ret = copy.copy(self)
            fun(ret, other)
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

        __rsub__ = __sub__

        def __imul__(self, other):
            if isinstance(other, self.__class__):
                self._magnitude *= other._magnitude
                self._units *= other._units
            else:
                self._magnitude *= other

            return self

        def __mul__(self, other):
            ret = copy.copy(self)
            ret *= other
            return ret

        __rmul__ = __mul__

        def __itruediv__(self, other):
            if isinstance(other, self.__class__):
                self._magnitude /= other._magnitude
                self._units /= other._units
            else:
                self._magnitude /= other

            return self

        def __truediv__(self, other):
            ret = copy.copy(self)
            ret /= other
            return ret

        def __rtruediv__(self, other):
            if isinstance(other, Number):
                return self.__class__(other / self._magnitude, 1 / self._units)
            raise NotImplementedError

        def __ifloordiv__(self, other):
            if isinstance(other, self.__class__):
                self._magnitude //= other._magnitude
                self._units /= other._units
            else:
                self._magnitude //= other

            return self

        def __floordiv__(self, other):
            ret = copy.copy(self)
            ret //= other
            return ret

        __div__ = __floordiv__
        __idiv__ = __ifloordiv__

        def __rfloordiv__(self, other):
            if isinstance(other, self.__class__):
                return self.__class__(other._magnitude // self._magnitude, other._units / self._units)
            else:
                return self.__class__(other // self._magnitude, 1.0 / self._units)

        def __ipow__(self, other):
            self._magnitude **= other
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
            if not isinstance(other, self.__class__):
                return self.dimensionless and self.magnitude == other

            if self._magnitude == 0 and other._magnitude == 0:
                return self.dimensionality == other.dimensionality

            if self._units == other._units:
                return self._magnitude == other._magnitude

            try:
                return self.to(other).magnitude == other._magnitude
            except DimensionalityError:
                return False

        def __lt__(self, other):
            if not isinstance(other, self.__class__):
                if self.dimensionless:
                    return operator.lt(self.magnitude, other)
                else:
                    raise ValueError('Cannot compare Quantity and {}'.format(type(other)))

            if self.units == other.units:
                return operator.lt(self._magnitude, other._magnitude)
            if self.dimensionality != other.dimensionality:
                raise DimensionalityError(self.units, other.units,
                                          self.dimensionality, other.dimensionality)
            return operator.lt(self.convert_to_reference().magnitude,
                               self.convert_to_reference().magnitude)

        def __bool__(self):
            return bool(self._magnitude)

        __nonzero__ = __bool__
