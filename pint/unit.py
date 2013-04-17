# -*- coding: utf-8 -*-
"""
    pint.unit
    ~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2012 by Hernan E. Grecco.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import copy
import math
import operator
import tokenize
import itertools

from io import open
from numbers import Number

from tokenize import untokenize, NUMBER, STRING, NAME, OP

if sys.version < '3':
    from StringIO import StringIO
    string_types = basestring
    _tokenize = lambda input_string: tokenize.generate_tokens(StringIO(input_string).readline)
else:
    from io import BytesIO
    string_types = str
    _tokenize = lambda input_string: tokenize.tokenize(BytesIO(input_string.encode('utf-8')).readline)

from .util import formatter, logger, NUMERIC_TYPES, pi_theorem, solve_dependencies

PRETTY = '⁰¹²³⁴⁵⁶⁷⁸⁹·⁻'


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


class ParserHelper(dict):
    """The ParserHelper stores in place the product of variables and
    their respective exponent and implements the corresponding operations.
    """

    __slots__ = ('scale', )

    def __init__(self, scale=1, *args, **kwargs):
        self.scale = scale
        dict.__init__(self, *args, **kwargs)

    @classmethod
    def from_string(cls, input_string):
        """Parse linear expression mathematical units and return a quantity object.
        """

        if not input_string:
            return cls()

        if '[' in input_string:
            input_string = input_string.replace('[', '__obra__').replace(']', '__cbra__')
            brackets = True
        else:
            brackets = False

        gen = _tokenize(input_string)
        result = []
        for toknum, tokval, _, _, _ in gen:
            if toknum in (STRING, NAME):
                if not tokval:
                    continue
                result.extend([
                    (NAME, 'L_'),
                    (OP, '('),
                    (STRING, tokval), (OP, '='), (NUMBER, '1'),
                    (OP, ')')
                ])
            else:
                result.append((toknum, tokval))

        ret = eval(untokenize(result),
                   {'__builtins__': None},
                   {'L_': cls})

        if not brackets:
            return ret

        return ParserHelper(ret.scale,
                            {key.replace('__obra__', '[').replace('__cbra__', ']'): value
                            for key, value in ret.items()})

    def __missing__(self, key):
        return 0.0

    def add(self, key, value):
        newval = self.__getitem__(key) + value
        if newval:
            self.__setitem__(key, newval)
        else:
            del self[key]

    def operate(self, items, op=operator.iadd, cleanup=True):
        for key, value in items:
            self[key] = op(self[key], value)

        if cleanup:
            keys = [key for key, value in self.items() if value == 0]
            for key in keys:
                del self[key]

    def __str__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value) for key, value in sorted(self.items())])
        return '{} {}'.format(self.scale, tmp)

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value) for key, value in sorted(self.items())])
        return '<ParserHelper({}, {})>'.format(self.scale, tmp)

    def __mul__(self, other):
        if isinstance(other, string_types):
            self.add(other, 1)
        elif isinstance(other, Number):
            self.scale *= other
        else:
            self.operate(other.items())
        return self

    __imul__ = __mul__
    __rmul__ = __mul__

    def __pow__(self, other):
        self.scale **= other
        for key in self.keys():
            self[key] *= other
        return self

    __ipow__ = __pow__

    def __truediv__(self, other):
        if isinstance(other, string_types):
            self.add(other, -1)
        elif isinstance(other, Number):
            self.scale /= other
        else:
            self.operate(other.items(), operator.sub)
        return self

    __itruediv__ = __truediv__
    __floordiv__ = __truediv__

    def __rtruediv__(self, other):
        self.__pow__(-1)
        if isinstance(other, string_types):
            self.add(other, 1)
        elif isinstance(other, Number):
            self.scale *= other
        else:
            self.operate(other.items(), operator.add)
        return self


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
        return self._symbol

    @property
    def aliases(self):
        return self._aliases

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol

    @property
    def converter(self):
        return self._converter

    def __str__(self):
        return self.name


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

            if '[' in converter:
                self.is_base = True
            else:
                self.is_base = False
            converter = ParserHelper.from_string(converter)
            self.reference = UnitsContainer(converter.items())
            if 'offset' in modifiers:
                converter = OffsetConverter(converter.scale, modifiers['offset'])
            else:
                converter = ScaleConverter(converter.scale)

        super(UnitDefinition, self).__init__(name, symbol, aliases, converter)


class DimensionDefinition(UnitDefinition):
    """Definition of a dimension.
    """


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

    def __str__(self):
        if not self:
            return 'dimensionless'
        return formatter(self.items())

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value) for key, value in sorted(self.items())])
        return '<UnitsContainer({})>'.format(tmp)

    def __format__(self, spec):
        if spec == '!s' or spec == '':
            return str(self)
        elif spec == '!r':
            return repr(self)
        elif spec == '!l':
            tmp = formatter(self.items(), r' \cdot ', '^[{:n}]', True, True, (r'\left(', r'\right)'))
            tmp = tmp.replace('[', '{').replace(']', '}')
            if '/' in tmp:
                return r'\frac{%s}' % tmp.replace(' / ', '}{')
            return tmp
        elif spec == '!p':
            pretty = '{}'.format(self)
            for bef, aft in ((' ** ', ''), (' / ', '/'), (' * ', PRETTY[10]), ('-', PRETTY[11])):
                pretty = pretty.replace(bef, aft)
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
    """

    def __init__(self, filename='', force_ndarray=False):
        self.Quantity = build_quantity_class(self, force_ndarray)

        #: Map dimension name (string) to its definition (DimensionDefinition).
        self._dimensions = {}

        #: Map unit name (string) to its definition (UnitDefinition).
        self._units = {}

        #: Map prefix name (string) to its definition (PrefixDefinition).
        self._prefixes = {'': PrefixDefinition('', '', (), 1)}

        #: Map suffix name (string) to canonical , and unit alias to canonical unit name
        self._suffixes = {'': None, 's': ''}

        self._definition_files = []
        if filename == '':
            self.load_definitions(os.path.join(os.path.dirname(__file__), 'default_en.txt'))
        elif filename is not None:
            self.load_definitions(filename)

        self.define(UnitDefinition('pi', 'π', (), ScaleConverter(math.pi)))

    def __getattr__(self, item):
        return self.Quantity(1, item)

    def __getitem__(self, item):
        return self.parse_expression(item)

    def define(self, definition):
        """Add unit to the registry.
        """
        if isinstance(definition, str):
            definition = Definition.from_string(definition)

        if isinstance(definition, UnitDefinition):
            d = self._units
            if definition.is_base:
                for dimension in definition.reference.keys():
                    if dimension and dimension in self._dimensions:
                        raise ValueError('Only one unit per dimension can be a base unit.')
                    self.define(DimensionDefinition(dimension, '', (), None))

        elif isinstance(definition, PrefixDefinition):
            d = self._prefixes
        elif isinstance(definition, DimensionDefinition):
            d = self._dimensions
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
            d_reference = UnitsContainer({'delta_' + ref: value
                                          for ref, value in definition.reference.items()})
            self.define(UnitDefinition(d_name, d_symbol, d_aliases,
                                                ScaleConverter(definition.converter.scale),
                                                d_reference, definition.is_base))

    def load_definitions(self, filename):
        """Add units and prefixes defined in a definition text file.
        """
        with open(filename, encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    self.define(Definition.from_string(line))
                except Exception as ex:
                    logger.error("Exception: Cannot add '{}' {}".format(line, ex))
        self._definition_files.append(filename)

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
        """Convert unit or dict of units to a dict of dimensions

        :param input_units:
        :return: dimensionality
        """
        units = UnitsContainer()
        if not input_units:
            return units

        for key, value in (input_units.items() if isinstance(input_units, dict)
                           else ((input_units, 1),)):
            reg = self._units[self.get_name(key)]
            if reg.is_base:
                units.update(reg.reference ** value)
            else:
                if not isinstance(reg.converter, ScaleConverter):
                    raise ValueError('{} is not a multiplicative unit'.format(reg))
                units *= self.get_dimensionality(reg.reference) ** value
        if '[]' in units:
            del units['[]']
        return units

    def get_base_units(self, input_units):
        """Convert unit or dict of units to the base units

        :param input_units:
        :return: multiplicative factor, base units
        """
        if not input_units:
            return 1., UnitsContainer()

        factor = 1.
        units = UnitsContainer()
        for key, value in (input_units.items() if isinstance(input_units, dict)
                           else ((input_units, 1),)):
            reg = self._units[self.get_name(key)]
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
            if not len(dst) == 1:
                raise DimensionalityError(src, dst,
                                          self.get_dimensionality(src),
                                          self.get_dimensionality(dst))
            src_unit, src_value = list(src.items())[0]
            src_unit = self._units[src_unit]
            if not isinstance(src_unit.converter, ScaleConverter):
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

    def parse_units(self, input_string):
        """Parse a units expression and returns a UnitContainer with
        the canonical names.

        The expression can only contain products, ratios and powers of units.

        :raises:
            :class:`pint.UndefinedUnitError` if a unit is not in the registry
            :class:`ValueError` if the expression is invalid.
        """
        if not input_string:
            return UnitsContainer()

        units = ParserHelper.from_string(input_string)
        if units.scale != 1:
            raise ValueError('Unit expression cannot have a scaling factor.')

        ret = UnitsContainer()
        for name, value in units.items():
            cname = self.get_name(name)
            if cname:
                ret[cname] = value

        return ret

    def parse_expression(self, input_string):
        """Parse a mathematical expression including units and return a quantity object.
        """

        if not input_string:
            return self.Quantity(1)

        gen = _tokenize(input_string)
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


def build_quantity_class(registry, force_ndarray=False):
    from .quantity import _Quantity

    class Quantity(_Quantity):
        pass

    Quantity._REGISTRY = registry
    Quantity.force_ndarray = force_ndarray

    return Quantity
