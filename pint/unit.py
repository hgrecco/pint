# -*- coding: utf-8 -*-
"""
    pint.unit
    ~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import copy
import math
import itertools
import functools
import pkg_resources

from io import open
from numbers import Number

from tokenize import untokenize, NUMBER, STRING, NAME, OP

from .util import (formatter, logger, NUMERIC_TYPES, pi_theorem, solve_dependencies,
                   ParserHelper, string_types, ptok, string_preprocessor)

from decimal import Decimal

class UndefinedUnitError(ValueError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, unit_names):
        super(ValueError, self).__init__()
        self.unit_names = unit_names

    def __str__(self):
        if isinstance(self.unit_names, string_types):
            return "'{}' is not defined in the unit registry".format(self.unit_names)
        elif isinstance(self.unit_names, (list, tuple)) and len(self.unit_names) == 1:
            return "'{}' is not defined in the unit registry".format(self.unit_names[0])
        elif isinstance(self.unit_names, set) and len(self.unit_names) == 1:
            uname = list(self.unit_names)[0]
            return "'{}' is not defined in the unit registry".format(uname)
        else:
            return '{} are not defined in the unit registry'.format(self.unit_names)


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
        return "Cannot convert from '{}'{} to '{}'{}".format(self.units1, dim1, self.units2, dim2)


class Converter(object):
    """Base class for value converters.
    """

    def to_reference(self, value):
        return value

    def from_reference(self, value):
        return value


class ScaleConverter(Converter):
    """A linear transformation
    """

    def __init__(self, scale):
        self.scale = scale

    def to_reference(self, value):
        return value / self.scale

    def from_reference(self, value):
        return value * self.scale


class OffsetConverter(Converter):
    """An affine transformation
    """

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def to_reference(self, value):
        return value / self.scale + self.offset

    def from_reference(self, value):
        return (value - self.offset) * self.scale


class Definition(object):
    """Base class for definitions.

    :param name: name.
    :param symbol: a short name or symbol for the definition
    :param aliases: iterable of other names.
    :param converter: an instance of Converter.
    """

    def __init__(self, name, symbol, aliases, converter):
        self._name = name
        self._symbol = symbol
        self._aliases = aliases
        self._converter = converter

    @classmethod
    def from_string(cls, definition):
        """Parse a definition
        """
        name, definition = definition.split('=', 1)
        name = name.strip()

        result = [res.strip() for res in definition.split('=')]
        value, aliases = result[0], tuple(result[1:])
        symbol, aliases = (aliases[0], aliases[1:]) if aliases else (None, aliases)

        if name.startswith('['):
            return DimensionDefinition(name, symbol, aliases, value)
        elif name.endswith('-'):
            name = name.rstrip('-')
            return PrefixDefinition(name, symbol, aliases, value)
        else:
            return UnitDefinition(name, symbol, aliases, value)

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol or self._name

    @property
    def aliases(self):
        return self._aliases

    @property
    def converter(self):
        return self._converter

    def __str__(self):
        return self.name


def _is_dim(name):
    return name.startswith('[') and name.endswith(']')


class PrefixDefinition(Definition):
    """Definition of a prefix.
    """

    def __init__(self, name, symbol, aliases, converter):
        if isinstance(converter, string_types):
            converter = ScaleConverter(eval(converter))
        aliases = tuple(alias.strip('-') for alias in aliases)
        if symbol:
            symbol = symbol.strip('-')
        super(PrefixDefinition, self).__init__(name, symbol, aliases, converter)


class UnitDefinition(Definition):
    """Definition of a unit.

    :param reference: Units container with reference units.
    :param is_base: indicates if it is a base unit.
    """

    def __init__(self, name, symbol, aliases, converter,
                 reference=None, is_base=False):
        self.reference = reference
        self.is_base = is_base
        if isinstance(converter, string_types):
            if ';' in converter:
                [converter, modifiers] = converter.split(';', 2)
                modifiers = {key.strip(): eval(value) for key, value in
                             (part.split(':') for part in modifiers.split(';'))}
            else:
                modifiers = {}

            converter = ParserHelper.from_string(converter)
            if all(_is_dim(key) for key in converter.keys()):
                self.is_base = True
            elif not any(_is_dim(key) for key in converter.keys()):
                self.is_base = False
            else:
                raise ValueError('Base units must be referenced only to dimensions. '
                                 'Derived units must not be referenced to dimensions.')
            self.reference = UnitsContainer(converter.items())
            if 'offset' in modifiers:
                converter = OffsetConverter(converter.scale, modifiers['offset'])
            else:
                converter = ScaleConverter(converter.scale)

        super(UnitDefinition, self).__init__(name, symbol, aliases, converter)


class DimensionDefinition(Definition):
    """Definition of a dimension.
    """

    def __init__(self, name, symbol, aliases, converter,
                 reference=None, is_base=False):
        self.reference = reference
        self.is_base = is_base
        if isinstance(converter, string_types):
            converter = ParserHelper.from_string(converter)
            if not converter:
                self.is_base = True
            elif all(_is_dim(key) for key in converter.keys()):
                self.is_base = False
            else:
                raise ValueError('Base dimensions must be referenced to None. '
                                 'Derived dimensions must only be referenced to dimensions.')
            self.reference = UnitsContainer(converter.items())

        super(DimensionDefinition, self).__init__(name, symbol, aliases, converter=None)


class UnitsContainer(dict):
    """The UnitsContainer stores the product of units and their respective
    exponent and implements the corresponding operations
    """
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        for key, value in self.items():
            if not isinstance(key, string_types):
                raise TypeError('key must be a str, not {}'.format(type(key)))
            if not isinstance(value, Number):
                raise TypeError('value must be a number, not {}'.format(type(value)))
            if not isinstance(value, float):
                self[key] = float(value)

    def __missing__(self, key):
        return 0.0

    def __setitem__(self, key, value):
        if not isinstance(key, string_types):
            raise TypeError('key must be a str, not {}'.format(type(key)))
        if not isinstance(value, NUMERIC_TYPES):
            raise TypeError('value must be a NUMERIC_TYPES, not {}'.format(type(value)))
        dict.__setitem__(self, key, float(value))

    def add(self, key, value):
        newval = self.__getitem__(key) + value
        if newval:
            self.__setitem__(key, newval)
        else:
            del self[key]

    def __eq__(self, other):
        if isinstance(other, string_types):
            other = ParserHelper.from_string(other)
            other = dict(other.items())
        return dict.__eq__(self, other)

    def __str__(self):
        if not self:
            return 'dimensionless'
        return formatter(self.items())

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value) for key, value in sorted(self.items())])
        return '<UnitsContainer({})>'.format(tmp)

    def __format__(self, spec):
        if 'L' in spec:
            tmp = formatter(self.items(), True, True,
                            r' \cdot ', r'\frac[{}][{}]', '{}^[{}]',
                            r'\left( {} \right)')
            tmp = tmp.replace('[', '{').replace(']', '}')
            return tmp
        elif 'P' in spec:
            def fmt_exponent(num):
                PRETTY = '⁰¹²³⁴⁵⁶⁷⁸⁹'
                ret = '{:n}'.format(num).replace('-', '⁻')
                for n in range(10):
                    ret = ret.replace(str(n), PRETTY[n])
                return ret
            tmp = formatter(self.items(), True, False,
                            '·', '/', '{}{}',
                            '({})', fmt_exponent)
            return tmp
        elif 'H' in spec:
            tmp = formatter(self.items(), True, True,
                            r' ', r'{}/{}', '{}<sup>{}</sup>',
                            r'({})')
            return tmp
        else:
            return str(self)

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
        if not isinstance(other, NUMERIC_TYPES):
            raise TypeError('Cannot power UnitsContainer by {}'.format(type(other)))
        for key, value in self.items():
            self[key] *= other
        return self

    def __pow__(self, other):
        if not isinstance(other, NUMERIC_TYPES):
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


class UnitRegistry(object):
    """The unit registry stores the definitions and relationships between
    units.

    :param filename: path of the units definition file to load.
                     Empty to load the default definition file.
                     None to leave the UnitRegistry empty.
    :param force_ndarray: convert any input, scalar or not to a numpy.ndarray.
    :param default_to_delta: In the context of a multiplication of units, interpret
                             non-multiplicative units as their *delta* counterparts.
    """

    def __init__(self, filename='', force_ndarray=False, default_to_delta=True):
        self.Quantity = build_quantity_class(self, force_ndarray)

        #: Map dimension name (string) to its definition (DimensionDefinition).
        self._dimensions = {}

        #: Map unit name (string) to its definition (UnitDefinition).
        self._units = {}

        #: Map prefix name (string) to its definition (PrefixDefinition).
        self._prefixes = {'': PrefixDefinition('', '', (), 1)}

        #: Map suffix name (string) to canonical , and unit alias to canonical unit name
        self._suffixes = {'': None, 's': ''}

        #: In the context of a multiplication of units, interpret
        #: non-multiplicative units as their *delta* counterparts.
        self.default_to_delta = default_to_delta

        if filename == '':
            data = pkg_resources.resource_filename(__name__, 'default_en.txt')
            self.load_definitions(data, True)
        elif filename is not None:
            self.load_definitions(filename)

        self.define(UnitDefinition('pi', 'π', (), ScaleConverter(math.pi)))

    def __getattr__(self, item):
        return self.Quantity(1, item)

    def __getitem__(self, item):
        return self.parse_expression(item)

    def __dir__(self):
        return list(self._units.keys()) + \
               ['define', 'load_definitions', 'get_name', 'get_symbol',
                'get_dimensionality', 'Quantity', 'wraps', 'parse_unit',
                'parse_units', 'parse_expression', 'pi_theorem',
                'convert', 'get_base_units']

    def define(self, definition):
        """Add unit to the registry.
        """
        if isinstance(definition, string_types):
            definition = Definition.from_string(definition)

        if isinstance(definition, DimensionDefinition):
            d = self._dimensions
        elif isinstance(definition, UnitDefinition):
            d = self._units
            if definition.is_base:
                for dimension in definition.reference.keys():
                    if dimension != '[]' and dimension in self._dimensions:
                        raise ValueError('Only one unit per dimension can be a base unit.')
                    self.define(DimensionDefinition(dimension, '', (), None, is_base=True))

        elif isinstance(definition, PrefixDefinition):
            d = self._prefixes
        else:
            raise TypeError('{} is not a valid definition.'.format(definition))

        d[definition.name] = definition

        if definition.symbol:
            d[definition.symbol] = definition

        for alias in definition.aliases:
            if ' ' in alias:
                logger.warn('Alias cannot contain a space: ' + alias)
            d[alias] = definition

        if isinstance(definition.converter, OffsetConverter):
            d_name = 'delta_' + definition.name
            if definition.symbol:
                d_symbol = 'Δ' + definition.symbol
            else:
                d_symbol = None
            d_aliases = tuple('Δ' + alias for alias in definition.aliases)

            def prep(_name):
                if _name.startswith('['):
                    return '[delta_' + _name[1:]
                return 'delta_' + _name

            d_reference = UnitsContainer({prep(ref): value
                                          for ref, value in definition.reference.items()})
            self.define(UnitDefinition(d_name, d_symbol, d_aliases,
                                       ScaleConverter(definition.converter.scale),
                                       d_reference, definition.is_base))

    def load_definitions(self, file, is_resource=False):
        """Add units and prefixes defined in a definition text file.
        """
        # Permit both filenames and line-iterables
        if isinstance(file, string_types):
            with open(file, encoding='utf-8') as fp:
                return self.load_definitions(fp, is_resource)

        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@import'):
                if is_resource:
                    path = pkg_resources.resource_filename(__name__, line[7:].strip())
                else:
                    try:
                        path = os.path.dirname(file.name)
                    except AttributeError:
                        path = os.getcwd()
                    path = os.path.join(path, os.path.normpath(line[7:].strip()))
                self.load_definitions(path, is_resource)
                continue
            try:
                self.define(Definition.from_string(line))
            except Exception as ex:
                logger.error("Exception: Cannot add '{}' {}".format(line, ex))

    def validate(self):
        """Walk the registry and calculate for each unit definition
        the corresponding base units and dimensionality.
        """

        deps = {name: set(definition.reference.keys())
                for name, definition in self._units.items()}

        for unit_names in solve_dependencies(deps):
            for unit_name in unit_names:
                bu = self.get_base_units(unit_name)
                di = self.get_dimensionality(bu)

    def get_name(self, name_or_alias):
        """Return the canonical name of a unit.
        """

        if name_or_alias == 'dimensionless':
            return ''

        try:
            return self._units[name_or_alias].name
        except KeyError:
            pass

        candidates = self._dedup_candidates(self.parse_unit_name(name_or_alias))
        if not candidates:
            raise UndefinedUnitError(name_or_alias)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning('Parsing {} yield multiple results. '
                           'Options are: {}'.format(name_or_alias, candidates))
            prefix, unit_name, _ = candidates[0]

        if prefix:
            name = prefix + unit_name
            symbol = self.get_symbol(name)
            prefix_def = self._prefixes[prefix]
            self._units[name] = UnitDefinition(name, symbol, (), prefix_def.converter,
                                               UnitsContainer({unit_name: 1}))
            return prefix + unit_name

        return unit_name

    def get_symbol(self, name_or_alias):
        """Return the preferred alias for a unit
        """
        candidates = self._dedup_candidates(self.parse_unit_name(name_or_alias))
        if not candidates:
            raise UndefinedUnitError(name_or_alias)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning('Parsing {} yield multiple results. '
                           'Options are: {!r}'.format(name_or_alias, candidates))
            prefix, unit_name, _ = candidates[0]

        return self._prefixes[prefix].symbol + self._units[unit_name].symbol

    def get_dimensionality(self, input_units):
        """Convert unit or dict of units or dimensions to a dict of base dimensions

        :param input_units:
        :return: dimensionality
        """
        dims = UnitsContainer()
        if not input_units:
            return dims

        if isinstance(input_units, string_types):
            input_units = ParserHelper.from_string(input_units)

        if len(input_units) == 1:
            key, value = list(input_units.items())[0]
            if _is_dim(key):
                reg = self._dimensions[key]
                if reg.is_base:
                    dims.add(key, value)
                else:
                    dims *= self.get_dimensionality(reg.reference) ** value
            else:
                reg = self._units[self.get_name(key)]
                if reg.is_base:
                    dims *= reg.reference ** value
                else:
                    dims *= self.get_dimensionality(reg.reference) ** value
            if '[]' in dims:
                del dims['[]']
            return dims

        for key, value in input_units.items():
            if _is_dim(key):
                reg = self._dimensions[key]
                if reg.is_base:
                    dims.add(key, value)
                else:
                    if reg.converter and not isinstance(reg.converter, ScaleConverter):
                        raise ValueError('{} is not a multiplicative unit'.format(reg))
                    dims *= self.get_dimensionality(reg.reference) ** value
            else:
                reg = self._units[self.get_name(key)]
                if reg.is_base:
                    dims *= reg.reference ** value
                else:
                    if not isinstance(reg.converter, ScaleConverter):
                        raise ValueError('{} is not a multiplicative unit'.format(reg))
                    dims *= self.get_dimensionality(reg.reference) ** value

        if '[]' in dims:
            del dims['[]']

        return dims

    def get_base_units(self, input_units):
        """Convert unit or dict of units to the base units

        :param input_units:
        :return: multiplicative factor, base units
        """
        if not input_units:
            return 1., UnitsContainer()

        if isinstance(input_units, string_types):
            input_units = ParserHelper.from_string(input_units)

        factor = 1.
        units = UnitsContainer()
        for key, value in input_units.items():
            key = self.get_name(key)
            reg = self._units[key]
            if reg.is_base:
                units.add(key, value)
            else:
                if not isinstance(reg.converter, ScaleConverter):
                    raise ValueError('{} is not a multiplicative unit'.format(reg))
                fac, uni = self.get_base_units(reg.reference)
                factor *= (reg.converter.scale * fac) ** value
                units *= uni ** value

        return factor, units

    def convert(self, value, src, dst):
        """Convert value from some source to destination units.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer or str
        :param dst: destination units.
        :type dst: UnitsContainer or str

        :return: converted value
        """
        if isinstance(src, string_types):
            src = ParserHelper.from_string(src)
        if isinstance(dst, string_types):
            dst = ParserHelper.from_string(dst)
        if src == dst:
            return value
        if len(src) == 1:
            src_unit, src_value = list(src.items())[0]
            src_unit = self._units[src_unit]
            if not isinstance(src_unit.converter, ScaleConverter):
                if not len(dst) == 1:
                    raise DimensionalityError(src, dst,
                                              self.get_dimensionality(src),
                                              self.get_dimensionality(dst))
                dst_unit, dst_value = list(dst.items())[0]
                dst_unit = self._units[dst_unit]
                if not type(src_unit.converter) is type(dst_unit.converter):
                    raise DimensionalityError(src, dst,
                                              self.get_dimensionality(src),
                                              self.get_dimensionality(dst))

                return dst_unit.converter.from_reference(src_unit.converter.to_reference(value))

        factor, units = self.get_base_units(src / dst)
        if len(units):
            raise DimensionalityError(src, dst,
                                      self.get_dimensionality(src),
                                      self.get_dimensionality(dst))

        # factor is type float and if our magintude is type Decimal then
        # must first convert to Decimal before we can '*' the values
        if isinstance(value, Decimal):
            return Decimal(str(factor)) * value
        
        return factor * value

    def pi_theorem(self, quantities):
        """Builds dimensionless quantities using the Buckingham π theorem

        :param quantities: mapping between variable name and units
        :type quantities: dict
        :return: a list of dimensionless quantities expressed as dicts
        """
        return pi_theorem(quantities, self)

    def _dedup_candidates(self, candidates):
        """Given a list of units, remove those with different names but equal value.
        """
        candidates = tuple(candidates)
        if len(candidates) < 2:
            return candidates

        unique = [candidates[0]]
        for c in candidates[2:]:
            for u in unique:
                if c == u:
                    break
            else:
                unique.append(c)

        return tuple(unique)

    def parse_unit_name(self, unit_name):
        """Parse a unit to identify prefix, unit name and suffix
        by walking the list of prefix and suffix.
        """
        for suffix, prefix in itertools.product(self._suffixes.keys(), self._prefixes.keys()):
            if unit_name.startswith(prefix) and unit_name.endswith(suffix):
                name = unit_name[len(prefix):]
                if suffix:
                    name = name[:-len(suffix)]
                    if len(name) == 1:
                        continue
                if name in self._units:
                    yield (self._prefixes[prefix].name,
                           self._units[name].name,
                           self._suffixes[suffix])

    def parse_units(self, input_string, to_delta=None):
        """Parse a units expression and returns a UnitContainer with
        the canonical names.

        The expression can only contain products, ratios and powers of units.

        :param to_delta: if the expression has multiple units, the parser will
                         interpret non multiplicative units as their `delta_` counterparts.

        :raises:
            :class:`pint.UndefinedUnitError` if a unit is not in the registry
            :class:`ValueError` if the expression is invalid.
        """
        if to_delta is None:
            to_delta = self.default_to_delta

        if not input_string:
            return UnitsContainer()

        units = ParserHelper.from_string(input_string)
        if units.scale != 1:
            raise ValueError('Unit expression cannot have a scaling factor.')

        ret = UnitsContainer()
        many = len(units) > 1
        for name, value in units.items():
            cname = self.get_name(name)
            if not cname:
                continue
            if to_delta and (many or (not many and abs(value) != 1)):
                definition = self._units[cname]
                if not isinstance(definition.converter, ScaleConverter):
                    cname = 'delta_' + cname
            ret[cname] = value

        return ret

    def parse_expression(self, input_string):
        """Parse a mathematical expression including units and return a quantity object.
        """

        if not input_string:
            return self.Quantity(1)

        input_string = string_preprocessor(input_string)
        gen = ptok(input_string)
        result = []
        unknown = set()
        for toknum, tokval, _, _, _ in gen:
            if toknum in (STRING, NAME):  # replace NUMBER tokens
                # TODO: Integrate math better, Replace eval
                if tokval == 'pi':
                    result.append((toknum, str(math.pi)))
                    continue
                try:
                    tokval = self.get_name(tokval)
                except UndefinedUnitError as ex:
                    unknown.add(ex.unit_names)
                if tokval:
                    result.extend([
                        (NAME, 'Q_'), (OP, '('), (NUMBER, '1'), (OP, ','),
                        (NAME, 'U_'),  (OP, '('), (STRING, tokval), (OP, '='), (NUMBER, '1'), (OP, ')'),
                        (OP, ')')
                    ])
                else:
                    result.extend([
                        (NAME, 'Q_'), (OP, '('), (NUMBER, '1'), (OP, ','),
                        (NAME, 'U_'), (OP, '('), (OP, ')'),
                        (OP, ')')
                    ])
            else:
                result.append((toknum, tokval))

        if unknown:
            raise UndefinedUnitError(unknown)
        return eval(untokenize(result), {'__builtins__': None},
                                        {'REGISTRY': self._units,
                                         'Q_': self.Quantity,
                                         'U_': UnitsContainer})

    def wraps(self, ret, args, strict=True):
        """Wraps a function to become pint-aware.

        Use it when a function requires a numerical value but in some specific
        units. The wrapper function will take a pint quantity, convert to the units
        specified in `args` and then call the wrapped function with the resulting
        magnitude.

        The value returned by the wrapped function will be converted to the units
        specified in `ret`.

        Use None to skip argument conversion.
        Set strict to False, to accept also numerical values.

        :param ret: output units.
        :param args: iterable of input units.
        :param strict: boolean to indicate that only quantities are accepted.
        :return: the wrapped function.
        :raises:
            :class:`ValueError` if strict and one of the arguments is not a Quantity.
        """

        Q_ = self.Quantity

        if not isinstance(args, (list, tuple)):
            args = (args, )

        def to_units(x):
            if isinstance(x, string_types):
                return self.parse_units(x)
            elif isinstance(x, Q_):
                return x.units
            return x

        units = [to_units(arg) for arg in args]

        if isinstance(ret, (list, tuple)):
            ret = ret.__class__([to_units(arg) for arg in ret])
        elif isinstance(ret, string_types):
            ret = self.parse_units(ret)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*values, **kw):
                new_args = []
                for unit, value in zip(units, values):
                    if unit is None:
                        new_args.append(value)
                    elif isinstance(value, Q_):
                        new_args.append(self.convert(value.magnitude, value.units, unit))
                    elif not strict:
                        new_args.append(value)
                    else:
                        raise ValueError('A wrapped function using strict=True requires '
                                         'quantity for all arguments with not None units. '
                                         '(error found for {}, {})'.format(unit, value))

                result = func(*new_args, **kw)

                if isinstance(ret, (list, tuple)):
                    return ret.__class__(res if unit is None else Q_(res, unit)
                                         for unit, res in zip(ret, result))
                elif ret is not None:
                    return Q_(result, ret)

                return result
            return wrapper
        return decorator


def build_quantity_class(registry, force_ndarray=False):
    from .quantity import _Quantity

    class Quantity(_Quantity):
        pass

    Quantity._REGISTRY = registry
    Quantity.force_ndarray = force_ndarray

    return Quantity
