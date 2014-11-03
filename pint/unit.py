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
from decimal import Decimal
from contextlib import contextmanager, closing
from io import open, StringIO
from numbers import Number
from collections import defaultdict
from tokenize import untokenize, NUMBER, STRING, NAME, OP

from .context import Context, ContextChain, _freeze
from .util import (logger, pi_theorem, solve_dependencies, ParserHelper,
                   string_preprocessor, find_connected_nodes, find_shortest_path)
from .compat import tokenizer, string_types, NUMERIC_TYPES, TransformDict
from .formatting import format_unit


class DefinitionSyntaxError(ValueError):
    """Raised when a textual definition has a syntax error.
    """

    def __init__(self, msg, filename=None, lineno=None):
        super(ValueError, self).__init__()
        self.msg = msg
        self.filename = None
        self.lineno = None

    def __str__(self):
        return "While opening {0}, in line {1}: ".format(self.filename, self.lineno) + self.msg


class RedefinitionError(ValueError):
    """Raised when a unit or prefix is redefined.
    """

    def __init__(self, name, definition_type):
        super(ValueError, self).__init__()
        self.name = name
        self.definition_type = definition_type
        self.filename = None
        self.lineno = None

    def __str__(self):
        msg = "cannot redefine '{0}' ({1})".format(self.name, self.definition_type)
        if self.filename:
            return "While opening {0}, in line {1}: ".format(self.filename, self.lineno) + msg
        return msg


class UndefinedUnitError(ValueError):
    """Raised when the units are not defined in the unit registry.
    """

    def __init__(self, unit_names):
        super(ValueError, self).__init__()
        self.unit_names = unit_names

    def __str__(self):
        if isinstance(self.unit_names, string_types):
            return "'{0}' is not defined in the unit registry".format(self.unit_names)
        elif isinstance(self.unit_names, (list, tuple)) and len(self.unit_names) == 1:
            return "'{0}' is not defined in the unit registry".format(self.unit_names[0])
        elif isinstance(self.unit_names, set) and len(self.unit_names) == 1:
            uname = list(self.unit_names)[0]
            return "'{0}' is not defined in the unit registry".format(uname)
        else:
            return '{0} are not defined in the unit registry'.format(self.unit_names)


class DimensionalityError(ValueError):
    """Raised when trying to convert between incompatible units.
    """

    def __init__(self, units1, units2, dim1=None, dim2=None, extra_msg=''):
        super(DimensionalityError, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2
        self.extra_msg = extra_msg

    def __str__(self):
        if self.dim1 or self.dim2:
            dim1 = ' ({0})'.format(self.dim1)
            dim2 = ' ({0})'.format(self.dim2)
        else:
            dim1 = ''
            dim2 = ''

        msg = "Cannot convert from '{0}'{1} to '{2}'{3}" + self.extra_msg

        return msg.format(self.units1, dim1, self.units2, dim2)


class OffsetUnitCalculusError(ValueError):
    """Raised on ambiguous operations with offset units.
    """
    def __init__(self, units1, units2='', extra_msg=''):
        super(ValueError, self).__init__()
        self.units1 = units1
        self.units2 = units2
        self.extra_msg = extra_msg

    def __str__(self):
        msg = ("Ambiguous operation with offset unit (%s)." %
               ', '.join(['%s' % u for u in [self.units1, self.units2] if u])
               + self.extra_msg)
        return msg.format(self.units1, self.units2)


class Converter(object):
    """Base class for value converters.
    """

    is_multiplicative = True

    def to_reference(self, value, inplace=False):
        return value

    def from_reference(self, value, inplace=False):
        return value


class ScaleConverter(Converter):
    """A linear transformation
    """

    is_multiplicative = True

    def __init__(self, scale):
        self.scale = scale

    def to_reference(self, value, inplace=False):
        if inplace:
            value *= self.scale
        else:
            value = value * self.scale

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value /= self.scale
        else:
            value = value / self.scale

        return value


class OffsetConverter(Converter):
    """An affine transformation
    """

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    @property
    def is_multiplicative(self):
        return self.offset == 0

    def to_reference(self, value, inplace=False):
        if inplace:
            value *= self.scale
            value += self.offset
        else:
            value = value * self.scale + self.offset

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value -= self.offset
            value /= self.scale
        else:
            value = (value - self.offset) / self.scale

        return value


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

    @property
    def is_multiplicative(self):
        return self._converter.is_multiplicative

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
    def has_symbol(self):
        return bool(self._symbol)

    @property
    def aliases(self):
        return self._aliases

    @property
    def converter(self):
        return self._converter

    def __str__(self):
        return self.name


def _is_dim(name):
    return name[0] == '[' and name[-1] == ']'


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
                modifiers = dict((key.strip(), eval(value)) for key, value in
                                 (part.split(':') for part in modifiers.split(';')))
            else:
                modifiers = {}

            converter = ParserHelper.from_string(converter)
            if all(_is_dim(key) for key in converter.keys()):
                self.is_base = True
            elif not any(_is_dim(key) for key in converter.keys()):
                self.is_base = False
            else:
                raise ValueError('Cannot mix dimensions and units in the same definition. '
                                 'Base units must be referenced only to dimensions. '
                                 'Derived units must be referenced only to units.')
            self.reference = UnitsContainer(converter.items())
            if modifiers.get('offset', 0.) != 0.:
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
                raise TypeError('key must be a str, not {0}'.format(type(key)))
            if not isinstance(value, Number):
                raise TypeError('value must be a number, not {0}'.format(type(value)))
            if not isinstance(value, float):
                self[key] = float(value)

    def __missing__(self, key):
        return 0.0

    def __setitem__(self, key, value):
        if not isinstance(key, string_types):
            raise TypeError('key must be a str, not {0}'.format(type(key)))
        if not isinstance(value, NUMERIC_TYPES):
            raise TypeError('value must be a NUMERIC_TYPES, not {0}'.format(type(value)))
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
      return self.__format__('')

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{0}': {1}".format(key, value) for key, value in sorted(self.items())])
        return '<UnitsContainer({0})>'.format(tmp)

    def __format__(self, spec):
        return format_unit(self, spec)

    def __copy__(self):
        ret = self.__class__()
        ret.update(self)
        return ret

    def __imul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot multiply UnitsContainer by {0}'.format(type(other)))
        for key, value in other.items():
            self[key] += value
        keys = [key for key, value in self.items() if value == 0]
        for key in keys:
            del self[key]

        return self

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot multiply UnitsContainer by {0}'.format(type(other)))
        ret = copy.copy(self)
        ret *= other
        return ret

    __rmul__ = __mul__

    def __ipow__(self, other):
        if not isinstance(other, NUMERIC_TYPES):
            raise TypeError('Cannot power UnitsContainer by {0}'.format(type(other)))
        for key, value in self.items():
            self[key] *= other
        return self

    def __pow__(self, other):
        if not isinstance(other, NUMERIC_TYPES):
            raise TypeError('Cannot power UnitsContainer by {0}'.format(type(other)))
        ret = copy.copy(self)
        ret **= other
        return ret

    def __itruediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot divide UnitsContainer by {0}'.format(type(other)))

        for key, value in other.items():
            self[key] -= value

        keys = [key for key, value in self.items() if value == 0]
        for key in keys:
            del self[key]

        return self

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Cannot divide UnitsContainer by {0}'.format(type(other)))

        ret = copy.copy(self)
        ret /= other
        return ret

    def __rtruediv__(self, other):
        if not isinstance(other, self.__class__) and other != 1:
            raise TypeError('Cannot divide {0} by UnitsContainer'.format(type(other)))

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
    :param default_as_delta: In the context of a multiplication of units, interpret
                             non-multiplicative units as their *delta* counterparts.
    :autoconvert_offset_to_baseunit: If True converts offset units in quantites are
                                     converted to their base units in multiplicative
                                     context. If False no conversion happens.
    :param on_redefinition: action to take in case a unit is redefined.
                            'warn', 'raise', 'ignore'
    :type on_redefintion: str
    """

    def __init__(self, filename='', force_ndarray=False, default_as_delta=True,
                 autoconvert_offset_to_baseunit=False,
                 on_redefinition='warn'):
        self.Quantity = build_quantity_class(self, force_ndarray)
        self.Measurement = build_measurement_class(self, force_ndarray)

        #: Action to take in case a unit is redefined. 'warn', 'raise', 'ignore'
        self._on_redefinition = on_redefinition

        #: Map dimension name (string) to its definition (DimensionDefinition).
        self._dimensions = {}

        #: Map unit name (string) to its definition (UnitDefinition).
        #: Might contain prefixed units.
        self._units = {}

        #: Map unit name in lower case (string) to a set of unit names with the right case.
        #: Does not contain prefixed units.
        #: e.g: 'hz' - > set('Hz', )
        self._units_casei = defaultdict(set)

        #: Map prefix name (string) to its definition (PrefixDefinition).
        self._prefixes = {'': PrefixDefinition('', '', (), 1)}

        #: Map suffix name (string) to canonical , and unit alias to canonical unit name
        self._suffixes = {'': None, 's': ''}

        #: Map context name (string) or abbreviation to context.
        self._contexts = {}

        #: Stores active contexts.
        self._active_ctx = ContextChain()

        #: Maps dimensionality (_freeze(UnitsContainer)) to Units (str)
        self._dimensional_equivalents = TransformDict(_freeze)

        #: Maps dimensionality (_freeze(UnitsContainer)) to Dimensionality (_freeze(UnitsContainer))
        self._base_units_cache = TransformDict(_freeze)
        #: Maps dimensionality (_freeze(UnitsContainer)) to Units (_freeze(UnitsContainer))
        self._dimensionality_cache = TransformDict(_freeze)

        #: Cache the unit name associated to user input. ('mV' -> 'millivolt')
        self._parse_unit_cache = dict()

        #: When performing a multiplication of units, interpret
        #: non-multiplicative units as their *delta* counterparts.
        self.default_as_delta = default_as_delta

        # Determines if quantities with offset units are converted to their
        # base units on multiplication and division.
        self.autoconvert_offset_to_baseunit = autoconvert_offset_to_baseunit

        if filename == '':
            self.load_definitions('default_en.txt', True)
        elif filename is not None:
            self.load_definitions(filename)

        self.define(UnitDefinition('pi', 'π', (), ScaleConverter(math.pi)))

        self._build_cache()

    def __name__(self):
        return 'UnitRegistry'

    def __getattr__(self, item):
        return self.Quantity(1, item)

    def __getitem__(self, item):
        logger.warning('Calling the getitem method from a UnitRegistry is deprecated. '
                       'use `parse_expression` method or use the registry as a callable.')
        return self.parse_expression(item)

    def __dir__(self):
        return list(self._units.keys()) + \
               ['define', 'load_definitions', 'get_name', 'get_symbol',
                'get_dimensionality', 'Quantity', 'wraps', 'parse_unit',
                'parse_units', 'parse_expression', 'pi_theorem',
                'convert', 'get_base_units']

    @property
    def default_format(self):
        """Default formatting string for quantities.
        """
        return self.Quantity.default_format

    @default_format.setter
    def default_format(self, value):
        self.Quantity.default_format = value

    def add_context(self, context):
        """Add a context object to the registry.

        The context will be accessible by its name and aliases.

        Notice that this method will NOT enable the context. Use `enable_contexts`.
        """
        if context.name in self._contexts:
            logger.warning('The name %s was already registered for another context.',
                           context.name)
        self._contexts[context.name] = context
        for alias in context.aliases:
            if alias in self._contexts:
                logger.warning('The name %s was already registered for another context',
                               context.name)
            self._contexts[alias] = context

    def remove_context(self, name_or_alias):
        """Remove a context from the registry and return it.

        Notice that this methods will not disable the context. Use `disable_contexts`.
        """
        context = self._contexts[name_or_alias]

        del self._contexts[context.name]
        for alias in context.aliases:
            del self._contexts[alias]

        return context

    def enable_contexts(self, *names_or_contexts, **kwargs):
        """Enable contexts provided by name or by object.

        :param names_or_contexts: sequence of the contexts or contexts names/alias
        :param kwargs: keyword arguments for the context
        """

        # If present, copy the defaults from the containing contexts
        if self._active_ctx.defaults:
            kwargs = dict(self._active_ctx.defaults, **kwargs)

        # For each name, we first find the corresponding context
        ctxs = tuple((self._contexts[name] if isinstance(name, string_types) else name)
                     for name in names_or_contexts)

        # Check if the contexts have been checked first, if not we make sure
        # that dimensions are expressed in terms of base dimensions.
        for ctx in ctxs:
            if getattr(ctx, '_checked', False):
                continue
            for (src, dst), func in ctx.funcs.items():
                src_ = self.get_dimensionality(dict(src))
                dst_ = self.get_dimensionality(dict(dst))
                if src != src_ or dst != dst_:
                    ctx.remove_transformation(src, dst)
                    ctx.add_transformation(src_, dst_, func)
            ctx._checked = True

        # and create a new one with the new defaults.
        ctxs = tuple(Context.from_context(ctx, **kwargs)
                     for ctx in ctxs)

        # Finally we add them to the active context.
        self._active_ctx.insert_contexts(*ctxs)

    def disable_contexts(self, n=None):
        """Disable the last n enabled contexts.
        """
        if n is None:
            n = len(self._contexts)
        self._active_ctx.remove_contexts(n)

    @contextmanager
    def context(self, *names, **kwargs):
        """Used as a context manager, this function enables to activate a context
        which is removed after usage.

        :param names: name of the context.
        :param kwargs: keyword arguments for the contexts.

        Context are called by their name::

            >>> with ureg.context('one'):
            ...     pass

        If the context has an argument, you can specify its value as a keyword
        argument::

            >>> with ureg.context('one', n=1):
            ...     pass

        Multiple contexts can be entered in single call:

            >>> with ureg.context('one', 'two', n=1):
            ...     pass

        or nested allowing you to give different values to the same keyword argument::

            >>> with ureg.context('one', n=1):
            ...     with ureg.context('two', n=2):
            ...         pass

        A nested context inherits the defaults from the containing context::

            >>> with ureg.context('one', n=1):
            ...     with ureg.context('two'): # Here n takes the value of the upper context
            ...         pass

        """

        # Enable the contexts.
        self.enable_contexts(*names, **kwargs)

        try:
            # After adding the context and rebuilding the graph, the registry
            # is ready to use.
            yield self
        finally:
            # Upon leaving the with statement,
            # the added contexts are removed from the active one.
            self.disable_contexts(len(names))

    def define(self, definition):
        """Add unit to the registry.
        """
        if isinstance(definition, string_types):
            definition = Definition.from_string(definition)

        if isinstance(definition, DimensionDefinition):
            d, di = self._dimensions, None
        elif isinstance(definition, UnitDefinition):
            d, di = self._units, self._units_casei
            if definition.is_base:
                for dimension in definition.reference.keys():
                    if dimension in self._dimensions:
                        if dimension != '[]':
                            raise DefinitionSyntaxError('only one unit per dimension can be a base unit.')
                        continue

                    self.define(DimensionDefinition(dimension, '', (), None, is_base=True))

        elif isinstance(definition, PrefixDefinition):
            d, di = self._prefixes, None
        else:
            raise TypeError('{0} is not a valid definition.'.format(definition))

        def _adder(key, value, action=self._on_redefinition, selected_dict=d, casei_dict=di):
            if key in selected_dict:
                if action == 'raise':
                    raise RedefinitionError(key, type(value))
                elif action == 'warn':
                    logger.warning("Redefining '%s' (%s)", key, type(value))

            selected_dict[key] = value
            if casei_dict is not None:
                casei_dict[key.lower()].add(key)

        _adder(definition.name, definition)

        if definition.has_symbol:
            _adder(definition.symbol, definition)

        for alias in definition.aliases:
            if ' ' in alias:
                logger.warn('Alias cannot contain a space: ' + alias)

            _adder(alias, definition)

        # define additional "delta_" units for units with an offset
        if getattr(definition.converter, "offset", 0.0) != 0.0:
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

            d_reference = UnitsContainer(dict((ref, value)
                                         for ref, value in definition.reference.items()))

            self.define(UnitDefinition(d_name, d_symbol, d_aliases,
                                       ScaleConverter(definition.converter.scale),
                                       d_reference, definition.is_base))

    def load_definitions(self, file, is_resource=False):
        """Add units and prefixes defined in a definition text file.
        """
        # Permit both filenames and line-iterables
        if isinstance(file, string_types):
            try:
                if is_resource:
                    with closing(pkg_resources.resource_stream(__name__, file)) as fp:
                        rbytes = fp.read()
                    return self.load_definitions(StringIO(rbytes.decode('utf-8')), is_resource)
                else:
                    with open(file, encoding='utf-8') as fp:
                        return self.load_definitions(fp, is_resource)
            except (RedefinitionError, DefinitionSyntaxError) as e:
                if e.filename is None:
                    e.filename = file
                raise e
            except Exception as e:
                msg = getattr(e, 'message', '') or str(e)
                raise ValueError('While opening {0}\n{1}'.format(file, msg))

        ifile = enumerate(file, 1)
        for no, line in ifile:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@import'):
                if is_resource:
                    path = line[7:].strip()
                else:
                    try:
                        path = os.path.dirname(file.name)
                    except AttributeError:
                        path = os.getcwd()
                    path = os.path.join(path, os.path.normpath(line[7:].strip()))
                self.load_definitions(path, is_resource)
            elif line.startswith('@context'):
                context = [line, ]
                for no, line in ifile:
                    line = line.strip()
                    if line.startswith('@end'):
                        try:
                            self.add_context(Context.from_lines(context, self.get_dimensionality))
                        except KeyError as e:
                            raise DefinitionSyntaxError('unknown dimension {0} in context'.format(str(e)), lineno=no)
                        break
                    elif line.startswith('@'):
                        raise DefinitionSyntaxError('cannot nest @ directives', lineno=no)
                    context.append(line)
            else:
                try:
                    self.define(Definition.from_string(line))
                except (RedefinitionError, DefinitionSyntaxError) as ex:
                    if ex.lineno is None:
                        ex.lineno = no
                    raise ex
                except Exception as ex:
                    logger.error("In line {0}, cannot add '{1}' {2}".format(no, line, ex))

    def _build_cache(self):
        """Build a cache of dimensionality and base units.
        """

        deps = dict((name, set(definition.reference.keys() if definition.reference else {}))
                    for name, definition in self._units.items())

        for unit_names in solve_dependencies(deps):
            for unit_name in unit_names:
                prefixed = False
                for p in self._prefixes.keys():
                    if p and unit_name.startswith(p):
                        prefixed = True
                        break
                if '[' in unit_name:
                    continue
                try:
                    uc = ParserHelper.from_word(unit_name)

                    bu = self.get_base_units(uc)
                    di = self.get_dimensionality(uc)

                    self._base_units_cache[uc] = bu
                    self._dimensionality_cache[uc] = di

                    if not prefixed:
                        if di not in self._dimensional_equivalents:
                            self._dimensional_equivalents[di] = set()

                        self._dimensional_equivalents[di].add(self._units[unit_name].name)

                except Exception as e:
                    logger.warning('Could not resolve {0}: {1!r}'.format(unit_name, e))

    def get_name(self, name_or_alias, case_sensitive=True):
        """Return the canonical name of a unit.
        """

        if name_or_alias == 'dimensionless':
            return ''

        try:
            return self._units[name_or_alias]._name
        except KeyError:
            pass

        candidates = self._dedup_candidates(self.parse_unit_name(name_or_alias, case_sensitive))
        if not candidates:
            raise UndefinedUnitError(name_or_alias)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning('Parsing {0} yield multiple results. '
                           'Options are: {1}'.format(name_or_alias, candidates))
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
            logger.warning('Parsing {0} yield multiple results. '
                           'Options are: {1!r}'.format(name_or_alias, candidates))
            prefix, unit_name, _ = candidates[0]

        return self._prefixes[prefix].symbol + self._units[unit_name].symbol

    def _get_symbol(self, name):
        return self._units[name].symbol

    def get_dimensionality(self, input_units):
        """Convert unit or dict of units or dimensions to a dict of base dimensions

        :param input_units:
        :return: dimensionality
        """
        if not input_units:
            return UnitsContainer()

        if isinstance(input_units, string_types):
            input_units = ParserHelper.from_string(input_units)

        if input_units in self._dimensionality_cache:
            return copy.copy(self._dimensionality_cache[input_units])

        accumulator = defaultdict(float)
        self._get_dimensionality_recurse(input_units, 1.0, accumulator)

        dims = UnitsContainer(dict((k, v) for k, v in accumulator.items() if v != 0.))

        if '[]' in dims:
            del dims['[]']

        self._dimensionality_cache[input_units] = copy.copy(dims)

        return dims

    def _get_dimensionality_recurse(self, ref, exp, accumulator):
        for key in ref:
            exp2 = exp*ref[key]
            if _is_dim(key):
                reg = self._dimensions[key]
                if reg.is_base:
                    accumulator[key] += exp2
                elif reg.reference is not None:
                    self._get_dimensionality_recurse(reg.reference, exp2, accumulator)
            else:
                reg = self._units[self.get_name(key)]
                if reg.reference is not None:
                    self._get_dimensionality_recurse(reg.reference, exp2, accumulator)

    def get_base_units(self, input_units, check_nonmult=True):
        """Convert unit or dict of units to the base units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        :param input_units: units
        :type input_units: UnitsContainer or str
        :param check_nonmult: if True, None will be returned as the multiplicative factor
                              is a non-multiplicative units is found in the final Units.
        :return: multiplicative factor, base units
        """
        if not input_units:
            return 1., UnitsContainer()

        if isinstance(input_units, string_types):
            input_units = ParserHelper.from_string(input_units)

        # The cache is only done for check_nonmult=True
        if check_nonmult and input_units in self._base_units_cache:
            return copy.deepcopy(self._base_units_cache[input_units])

        accumulators = [1., defaultdict(float)]
        self._get_base_units(input_units, 1.0, accumulators)

        factor = accumulators[0]
        units = UnitsContainer(dict((k, v) for k, v in accumulators[1].items() if v != 0.))

        # Check if any of the final units is non multiplicative and return None instead.
        if check_nonmult:
            for unit in units.keys():
                if not self._units[unit].converter.is_multiplicative:
                    return None, units

        return factor, units

    def _get_base_units(self, ref, exp, accumulators):
        for key in ref:
            exp2 = exp*ref[key]
            key = self.get_name(key)
            reg = self._units[key]
            if reg.is_base:
                accumulators[1][key] += exp2
            else:
                accumulators[0] *= reg._converter.scale ** exp2
                if reg.reference is not None:
                    self._get_base_units(reg.reference, exp2, accumulators)

    def get_compatible_units(self, input_units):
        if not input_units:
            return 1., UnitsContainer()

        if isinstance(input_units, string_types):
            input_units = ParserHelper.from_string(input_units)

        src_dim = self.get_dimensionality(input_units)

        ret = self._dimensional_equivalents[src_dim]

        if self._active_ctx:
            nodes = find_connected_nodes(self._active_ctx.graph, _freeze(src_dim))
            ret = set()
            if nodes:
                for node in nodes:
                    ret |= self._dimensional_equivalents[node]

        return frozenset(ret)

    def convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer or str
        :param dst: destination units.
        :type dst: UnitsContainer or str

        :return: converted value
        """
        if isinstance(src, string_types):
            src = self.parse_units(src)
        if isinstance(dst, string_types):
            dst = self.parse_units(dst)
        if src == dst:
            return value

        src_dim = self.get_dimensionality(src)
        dst_dim = self.get_dimensionality(dst)

        # If there is an active context, we look for a path connecting source and
        # destination dimensionality. If it exists, we transform the source value
        # by applying sequentially each transformation of the path.
        if self._active_ctx:
            path = find_shortest_path(self._active_ctx.graph,
                                      *Context.__keytransform__(src_dim, dst_dim))
            if path:
                src = self.Quantity(value, src)
                for a, b in zip(path[:-1], path[1:]):
                    src = self._active_ctx.transform(a, b, self, src)

                value, src = src.magnitude, src.units

                src_dim = self.get_dimensionality(src)

        # If the source and destination dimensionality are different,
        # then the conversion cannot be performed.

        if src_dim != dst_dim:
            raise DimensionalityError(src, dst, src_dim, dst_dim)

        # Conversion needs to consider if non-multiplicative (AKA offset
        # units) are involved. Conversion is only possible if src and dst
        # have at most one offset unit per dimension.
        src_offset_units = [(u, e) for u, e in src.items()
                            if not self._units[u].is_multiplicative]
        dst_offset_units = [(u, e) for u, e in dst.items()
                            if not self._units[u].is_multiplicative]

        # For offset units we need to check if the conversion is allowed.
        if src_offset_units or dst_offset_units:

            # Validate that not more than one offset unit is present
            if len(src_offset_units) > 1 or len(dst_offset_units) > 1:
                raise DimensionalityError(
                    src, dst, src_dim, dst_dim,
                    extra_msg=' - more than one offset unit.')

            # validate that offset unit is not used in multiplicative context
            if ((len(src_offset_units) == 1 and len(src) > 1)
                    or (len(dst_offset_units) == 1 and len(dst) > 1)
                    and not self.autoconvert_offset_to_baseunit):
                raise DimensionalityError(
                    src, dst, src_dim, dst_dim,
                    extra_msg=' - offset unit used in multiplicative context.')

            # Validate that order of offset unit is exactly one.
            if src_offset_units:
                if src_offset_units[0][1] != 1:
                    raise DimensionalityError(
                        src, dst, src_dim, dst_dim,
                        extra_msg=' - offset units in higher order.')
            else:
                if dst_offset_units[0][1] != 1:
                    raise DimensionalityError(
                        src, dst, src_dim, dst_dim,
                        extra_msg=' - offset units in higher order.')

        # Here we convert only the offset quantities. Any remaining scaled
        # quantities will be converted later.

        # clean src from offset units by converting to reference
        for u, e in src_offset_units:
            value = self._units[u].converter.to_reference(value, inplace)
            src.pop(u)

        # clean dst units from offset units
        for u, e in dst_offset_units:
            dst.pop(u)

        # Here src and dst have only multiplicative units left. Thus we can
        # convert with a factor.
        factor, units = self.get_base_units(src / dst)

        # factor is type float and if our magnitude is type Decimal then
        # must first convert to Decimal before we can '*' the values
        if isinstance(value, Decimal):
            factor = Decimal(str(factor))

        if inplace:
            value *= factor
        else:
            value = value * factor

        # Finally convert to offset units specified in destination
        for u, e in dst_offset_units:
            value = self._units[u].converter.from_reference(value, inplace)
            # add back offset units to dst
            dst[u] = e

        # restore offset conversion of src units
        for u, e in src_offset_units:
            src[u] = e

        return value

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

    def parse_unit_name(self, unit_name, case_sensitive=True):
        """Parse a unit to identify prefix, unit name and suffix
        by walking the list of prefix and suffix.
        """
        stw = unit_name.startswith
        edw = unit_name.endswith
        for suffix, prefix in itertools.product(self._suffixes, self._prefixes):
            if stw(prefix) and edw(suffix):
                name = unit_name[len(prefix):]
                if suffix:
                    name = name[:-len(suffix)]
                    if len(name) == 1:
                        continue
                if case_sensitive:
                    if name in self._units:
                        yield (self._prefixes[prefix]._name,
                               self._units[name]._name,
                               self._suffixes[suffix])
                else:
                    for real_name in self._units_casei.get(name.lower(), ()):
                        yield (self._prefixes[prefix]._name,
                               self._units[real_name]._name,
                               self._suffixes[suffix])

    def parse_units(self, input_string, as_delta=None):
        """Parse a units expression and returns a UnitContainer with
        the canonical names.

        The expression can only contain products, ratios and powers of units.

        :param as_delta: if the expression has multiple units, the parser will
                         interpret non multiplicative units as their `delta_` counterparts.

        :raises:
            :class:`pint.UndefinedUnitError` if a unit is not in the registry
            :class:`ValueError` if the expression is invalid.
        """
        if input_string in self._parse_unit_cache:
            return self._parse_unit_cache[input_string]

        if as_delta is None:
            as_delta = self.default_as_delta

        if not input_string:
            return UnitsContainer()

        units = ParserHelper.from_string(input_string)
        if units.scale != 1:
            raise ValueError('Unit expression cannot have a scaling factor.')

        ret = UnitsContainer()
        many = len(units) > 1
        for name in units:
            cname = self.get_name(name)
            value = units[name]
            if not cname:
                continue
            if as_delta and (many or (not many and value != 1)):
                definition = self._units[cname]
                if not definition.is_multiplicative:
                    cname = 'delta_' + cname
            ret[cname] = value

        self._parse_unit_cache[input_string] = ret

        return ret

    def parse_expression(self, input_string, case_sensitive=True, **values):
        """Parse a mathematical expression including units and return a quantity object.

        Numerical constants can be specified as keyword arguments and will take precedence
        over the names defined in the registry.
        """

        if not input_string:
            return self.Quantity(1)

        input_string = string_preprocessor(input_string)
        gen = tokenizer(input_string)
        result = []
        unknown = set()
        for toknum, tokval, _, _, _ in gen:
            if toknum == NAME:
                # TODO: Integrate math better, Replace eval, make as_delta-aware
                if tokval == 'pi' or tokval in values:
                    result.append((toknum, tokval))
                    continue
                try:
                    tokval = self.get_name(tokval, case_sensitive)
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
        return eval(untokenize(result),
                    {'__builtins__': None,
                     'REGISTRY': self._units,
                     'Q_': self.Quantity,
                     'U_': UnitsContainer,
                     'pi': math.pi},
                    values
                    )

    __call__ = parse_expression

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
            assigned = tuple(attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
            updated = tuple(attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))
            @functools.wraps(func, assigned=assigned, updated=updated)
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
                                         '(error found for {0}, {1})'.format(unit, value))

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


def build_measurement_class(registry, force_ndarray=False):
    from .measurement import _Measurement, ufloat

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


class LazyRegistry(object):

    def __init__(self, args=None, kwargs=None):
        self.__dict__['params'] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__['params']
        kwargs['on_redefinition'] = 'raise'
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item == '_on_redefinition':
            return 'raise'
        self.__init()
        return getattr(self, item)

    def __setattr__(self, key, value):
        if key == '__class__':
            super(LazyRegistry, self).__setattr__(key, value)
        else:
            self.__init()
            setattr(self, key, value)

    def __getitem__(self, item):
        self.__init()
        return self[item]

    def __call__(self, *args, **kwargs):
        self.__init()
        return self(*args, **kwargs)
