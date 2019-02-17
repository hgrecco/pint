# -*- coding: utf-8 -*-
"""
    pint.registry
    ~~~~~~~~~~~~~

    Defines the Registry, a class to contain units and their relations.

    The module actually defines 5 registries with different capabilites:

    - BaseRegistry: Basic unit definition and querying.
                    Conversion between multiplicative units.

    - NonMultiplicativeRegistry: Conversion between non multiplicative (offset) units.
                                 (e.g. Temperature)
      * Inherits from BaseRegistry

    - ContextRegisty: Conversion between units with different dimenstions according
                      to previously established relations (contexts).
                      (e.g. in the spectroscopy, conversion between frequency and energy is possible)
      * Inherits from BaseRegistry

    - SystemRegistry: Group unit and changing of base units.
                      (e.g. in MKS, meter, kilogram and second are base units.)

      * Inherits from BaseRegistry

    - UnitRegistry: Combine all previous capabilities, it is exposed by Pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import re
import math
import functools
import itertools
import pkg_resources
from decimal import Decimal
from fractions import Fraction
from contextlib import contextmanager, closing
from io import open, StringIO
from collections import defaultdict
from tokenize import NUMBER, NAME

from . import registry_helpers
from .context import Context, ContextChain
from .util import (logger, pi_theorem, solve_dependencies, ParserHelper,
                   string_preprocessor, find_connected_nodes,
                   find_shortest_path, UnitsContainer, _is_dim,
                   to_units_container, SourceIterator)

from .compat import tokenizer, string_types, meta
from .definitions import (Definition, UnitDefinition, PrefixDefinition,
                          DimensionDefinition)
from .converters import ScaleConverter
from .errors import (DimensionalityError, UndefinedUnitError,
                     DefinitionSyntaxError, RedefinitionError)

from .pint_eval import build_eval_tree
from . import systems

_BLOCK_RE = re.compile(r' |\(')


class _Meta(type):
    """This is just to call after_init at the right time
    instead of asking the developer to do it when subclassing.
    """

    def __call__(self, *args, **kwargs):
        obj = super(_Meta, self).__call__(*args, **kwargs)
        obj._after_init()
        return obj


class BaseRegistry(meta.with_metaclass(_Meta)):
    """Base class for all registries.

    Capabilities:
    - Register units, prefixes, and dimensions, and their relations.
    - Convert between units.
    - Find dimensionality of a unit.
    - Parse units with prefix and/or suffix.
    - Parse expressions.
    - Parse a definition file.
    - Allow extending the definition file parser by registering @ directives.

    :param filename: path of the units definition file to load or line iterable object.
                     Empty to load the default definition file.
                     None to leave the UnitRegistry empty.
    :type filename: str | None
    :param force_ndarray: convert any input, scalar or not to a numpy.ndarray.
    :param on_redefinition: action to take in case a unit is redefined.
                            'warn', 'raise', 'ignore'
    :type on_redefinition: str
    :param auto_reduce_dimensions: If True, reduce dimensionality on appropriate operations.
    """

    #: Map context prefix to function
    #: type: Dict[str, (SourceIterator -> None)]
    _parsers = None

    #: List to be used in addition of units when dir(registry) is called.
    #: Also used for autocompletion in IPython.
    _dir = ['Quantity', 'Unit', 'Measurement',
            'define', 'load_definitions',
            'get_name', 'get_symbol', 'get_dimensionality',
            'get_base_units', 'get_root_units',
            'parse_unit_name', 'parse_units', 'parse_expression',
            'convert']

    def __init__(self, filename='', force_ndarray=False, on_redefinition='warn', auto_reduce_dimensions=False):

        self._register_parsers()

        from .unit import build_unit_class
        self.Unit = build_unit_class(self)

        from .quantity import build_quantity_class
        self.Quantity = build_quantity_class(self, force_ndarray)

        from .measurement import build_measurement_class
        self.Measurement = build_measurement_class(self, force_ndarray)

        self._filename = filename

        #: Action to take in case a unit is redefined. 'warn', 'raise', 'ignore'
        self._on_redefinition = on_redefinition

        #: Determines if dimensionality should be reduced on appropriate operations.
        self.auto_reduce_dimensions = auto_reduce_dimensions

        #: Map between name (string) and value (string) of defaults stored in the definitions file.
        self._defaults = {}

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

        #: Maps dimensionality (UnitsContainer) to Units (str)
        self._dimensional_equivalents = dict()

        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self._root_units_cache = dict()

        #: Maps dimensionality (UnitsContainer) to Units (UnitsContainer)
        self._dimensionality_cache = dict()

        #: Cache the unit name associated to user input. ('mV' -> 'millivolt')
        self._parse_unit_cache = dict()

        self._initialized = False

    def _after_init(self):
        """This should be called after all __init__
        """
        if self._filename == '':
            self.load_definitions('default_en.txt', True)
        elif self._filename is not None:
            self.load_definitions(self._filename)

        self.define(UnitDefinition('pi', 'π', (), ScaleConverter(math.pi)))

        self._build_cache()
        self._initialized = True

    def _register_parsers(self):
        self._register_parser('@defaults', self._parse_defaults)

    def _parse_defaults(self, ifile):
        """Loader for a @default section.

        :type ifile: SourceITerator
        """
        next(ifile)
        for lineno, part in ifile.block_iter():
            k, v = part.split('=')
            self._defaults[k.strip()] = v.strip()

    def __name__(self):
        return 'UnitRegistry'

    def __getattr__(self, item):
        if item[0] == '_':
            return super(BaseRegistry, self).__getattribute__(item)
        return self.Unit(item)

    def __getitem__(self, item):
        logger.warning('Calling the getitem method from a UnitRegistry is deprecated. '
                       'use `parse_expression` method or use the registry as a callable.')
        return self.parse_expression(item)

    def __dir__(self):
        return list(self._units.keys()) + self._dir

    @property
    def default_format(self):
        """Default formatting string for quantities.
        """
        return self.Quantity.default_format

    @default_format.setter
    def default_format(self, value):
        self.Unit.default_format = value
        self.Quantity.default_format = value

    def define(self, definition):
        """Add unit to the registry.

        :param definition: a dimension, unit or prefix definition.
        :type definition: str | Definition
        """

        if isinstance(definition, string_types):
            for line in definition.split('\n'):
                self._define(Definition.from_string(line))
        else:
            self._define(definition)

    def _define(self, definition):
        """Add unit to the registry.

        This method defines only multiplicative units, converting any other type
        to `delta_` units.

        :param definition: a dimension, unit or prefix definition.
        :type definition: Definition
        :return: Definition instance, case sensitive unit dict, case insensitive unit dict.
        :rtype: Definition, dict, dict
        """

        if isinstance(definition, DimensionDefinition):
            d, di = self._dimensions, None

        elif isinstance(definition, UnitDefinition):
            d, di = self._units, self._units_casei

            # For a base units, we need to define the related dimension
            # (making sure there is only one to define)
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
            raise TypeError('{} is not a valid definition.'.format(definition))

        # define "delta_" units for units with an offset
        if getattr(definition.converter, "offset", 0.0) != 0.0:

            if definition.name.startswith('['):
                d_name = '[delta_' + definition.name[1:]
            else:
                d_name = 'delta_' + definition.name

            if definition.symbol:
                d_symbol = 'Δ' + definition.symbol
            else:
                d_symbol = None

            d_aliases = tuple('Δ' + alias for alias in definition.aliases)

            d_reference = UnitsContainer(dict((ref, value)
                                         for ref, value in definition.reference.items()))

            d_def = UnitDefinition(d_name, d_symbol, d_aliases,
                                   ScaleConverter(definition.converter.scale),
                                   d_reference, definition.is_base)
        else:
            d_def = definition

        self._define_adder(d_def, d, di)

        return definition, d, di

    def _define_adder(self, definition, unit_dict, casei_unit_dict):
        """Helper function to store a definition in the internal dictionaries.
        It stores the definition under its name, symbol and aliases.
        """
        self._define_single_adder(definition.name, definition, unit_dict, casei_unit_dict)

        if definition.has_symbol:
            self._define_single_adder(definition.symbol, definition, unit_dict, casei_unit_dict)

        for alias in definition.aliases:
            if ' ' in alias:
                logger.warn('Alias cannot contain a space: ' + alias)

            self._define_single_adder(alias, definition, unit_dict, casei_unit_dict)

    def _define_single_adder(self, key, value, unit_dict, casei_unit_dict):
        """Helper function to store a definition in the internal dictionaries.

        It warns or raise error on redefinition.
        """
        if key in unit_dict:
            if self._on_redefinition == 'raise':
                raise RedefinitionError(key, type(value))
            elif self._on_redefinition == 'warn':
                logger.warning("Redefining '%s' (%s)", key, type(value))

        unit_dict[key] = value
        if casei_unit_dict is not None:
            casei_unit_dict[key.lower()].add(key)

    def _register_parser(self, prefix, parserfunc):
        """Register a loader for a given @ directive..

        :param prefix: string identifying the section (e.g. @context)
        :param parserfunc: A function that is able to parse a Definition section.
        :type parserfunc: SourceIterator -> None
        """
        if self._parsers is None:
            self._parsers = dict()

        if prefix and prefix[0] == '@':
            self._parsers[prefix] = parserfunc
        else:
            raise ValueError("Prefix directives must start with '@'")

    def load_definitions(self, file, is_resource=False):
        """Add units and prefixes defined in a definition text file.

        :param file: can be a filename or a line iterable.
        :param is_resource: used to indicate that the file is a resource file
                            and therefore should be loaded from the package.
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
                raise ValueError('While opening {}\n{}'.format(file, msg))

        ifile = SourceIterator(file)
        for no, line in ifile:
            if line and line[0] == '@':
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
                else:
                    parts = _BLOCK_RE.split(line)

                    loader = self._parsers.get(parts[0], None) if self._parsers else None

                    if loader is None:
                        raise DefinitionSyntaxError('Unknown directive %s' % line, lineno=no)

                    try:
                        loader(ifile)
                    except DefinitionSyntaxError as ex:
                        if ex.lineno is None:
                            ex.lineno = no
                        raise ex
            else:
                try:
                    self.define(Definition.from_string(line))
                except DefinitionSyntaxError as ex:
                    if ex.lineno is None:
                        ex.lineno = no
                    raise ex
                except Exception as ex:
                    logger.error("In line {}, cannot add '{}' {}".format(no, line, ex))

    def _build_cache(self):
        """Build a cache of dimensionality and base units.
        """
        self._dimensional_equivalents = dict()

        deps = dict((name, set(definition.reference.keys() if definition.reference else {}))
                    for name, definition in self._units.items())

        for unit_names in solve_dependencies(deps):
            for unit_name in unit_names:
                if '[' in unit_name:
                    continue
                parsed_names = tuple(self.parse_unit_name(unit_name))
                _prefix = None
                if parsed_names:
                    _prefix, base_name, _suffix = parsed_names[0]
                else:
                    base_name = unit_name
                prefixed = True if _prefix else False
                try:
                    uc = ParserHelper.from_word(base_name)

                    bu = self._get_root_units(uc)
                    di = self._get_dimensionality(uc)

                    self._root_units_cache[uc] = bu
                    self._dimensionality_cache[uc] = di

                    if not prefixed:
                        if di not in self._dimensional_equivalents:
                            self._dimensional_equivalents[di] = set()

                        self._dimensional_equivalents[di].add(self._units[base_name]._name)

                except Exception as e:
                    logger.warning('Could not resolve {0}: {1!r}'.format(unit_name, e))

    def _dedup_candidates(self, candidates):
        """Given a list of unit triplets (prefix, name, suffix),
        remove those with different names but equal value.

            e.g. ('kilo', 'gram', '') and ('', 'kilogram', '')
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
            logger.warning('Parsing {0} yield multiple results. '
                           'Options are: {1!r}'.format(name_or_alias, candidates))
            prefix, unit_name, _ = candidates[0]

        return self._prefixes[prefix].symbol + self._units[unit_name].symbol

    def _get_symbol(self, name):
        return self._units[name].symbol

    def get_dimensionality(self, input_units):
        """Convert unit or dict of units or dimensions to a dict of base dimensions
        dimensions

        :param input_units:
        :return: dimensionality
        """
        input_units = to_units_container(input_units)

        return self._get_dimensionality(input_units)

    def _get_dimensionality(self, input_units):
        """ Convert a UnitsContainer to base dimensions.

        :param input_units:
        :return: dimensionality
        """
        if not input_units:
            return UnitsContainer()

        if input_units in self._dimensionality_cache:
            return self._dimensionality_cache[input_units]

        accumulator = defaultdict(float)
        self._get_dimensionality_recurse(input_units, 1.0, accumulator)

        if '[]' in accumulator:
            del accumulator['[]']

        dims = UnitsContainer(dict((k, v) for k, v in accumulator.items()
                                   if v != 0.0))

        self._dimensionality_cache[input_units] = dims

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

    def _get_dimensionality_ratio(self, unit1, unit2):
        """ Get the exponential ratio between two units, i.e. solve unit2 = unit1**x for x.
        :param unit1: first unit
        :type unit1: UnitsContainer compatible (str, Unit, UnitsContainer, dict)
        :param unit2: second unit
        :type unit2: UnitsContainer compatible (str, Unit, UnitsContainer, dict)
        :returns: exponential proportionality or None if the units cannot be converted
        """
        #shortcut in case of equal units
        if unit1 == unit2:
            return 1

        dim1, dim2 = (self.get_dimensionality(unit) for unit in (unit1, unit2))
        if not dim1 or not dim2 or dim1.keys() != dim2.keys(): #not comparable
            return None

        ratios = (dim2[key]/val for key, val in dim1.items())
        first = next(ratios)
        if all(r == first for r in ratios): #all are same, we're good
            return first
        return None

    def get_root_units(self, input_units, check_nonmult=True):
        """Convert unit or dict of units to the root units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        :param input_units: units
        :type input_units: UnitsContainer or str
        :param check_nonmult: if True, None will be returned as the
                              multiplicative factor if a non-multiplicative
                              units is found in the final Units.
        :return: multiplicative factor, base units
        """
        input_units = to_units_container(input_units)

        f, units = self._get_root_units(input_units, check_nonmult)

        return f, self.Unit(units)

    def _get_root_units(self, input_units, check_nonmult=True):
        """Convert unit or dict of units to the root units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        :param input_units: units
        :type input_units: UnitsContainer or dict
        :param check_nonmult: if True, None will be returned as the
                              multiplicative factor if a non-multiplicative
                              units is found in the final Units.
        :return: multiplicative factor, base units
        """
        if not input_units:
            return 1., UnitsContainer()

        # The cache is only done for check_nonmult=True
        if check_nonmult and input_units in self._root_units_cache:
            return self._root_units_cache[input_units]

        accumulators = [1., defaultdict(float)]
        self._get_root_units_recurse(input_units, 1.0, accumulators)

        factor = accumulators[0]
        units = UnitsContainer(dict((k, v) for k, v in accumulators[1].items()
                                    if v != 0.))

        # Check if any of the final units is non multiplicative and return None instead.
        if check_nonmult:
            for unit in units.keys():
                if not self._units[unit].converter.is_multiplicative:
                    return None, units

        if check_nonmult:
            self._root_units_cache[input_units] = factor, units

        return factor, units

    def get_base_units(self, input_units, check_nonmult=True, system=None):
        """Convert unit or dict of units to the base units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        :param input_units: units
        :type input_units: UnitsContainer or str
        :param check_nonmult: if True, None will be returned as the
                              multiplicative factor if a non-multiplicative
                              units is found in the final Units.
        :return: multiplicative factor, base units
        """

        return self.get_root_units(input_units, check_nonmult)

    def _get_root_units_recurse(self, ref, exp, accumulators):
        for key in sorted(ref):
            exp2 = exp*ref[key]
            key = self.get_name(key)
            reg = self._units[key]
            if reg.is_base:
                accumulators[1][key] += exp2
            else:
                accumulators[0] *= reg._converter.scale ** exp2
                if reg.reference is not None:
                    self._get_root_units_recurse(reg.reference, exp2,
                                                 accumulators)

    def get_compatible_units(self, input_units, group_or_system=None):
        """
        """
        input_units = to_units_container(input_units)

        equiv = self._get_compatible_units(input_units, group_or_system)

        return frozenset(self.Unit(eq) for eq in equiv)

    def _get_compatible_units(self, input_units, group_or_system):
        """
        """
        if not input_units:
            return frozenset()

        src_dim = self._get_dimensionality(input_units)

        ret = self._dimensional_equivalents[src_dim]

        return ret

    def convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        :param value: value
        :param src: source units.
        :type src: Quantity or str
        :param dst: destination units.
        :type dst: Quantity or str

        :return: converted value
        """
        src = to_units_container(src, self)

        dst = to_units_container(dst, self)

        if src == dst:
            return value

        return self._convert(value, src, dst, inplace)

    def _convert(self, value, src, dst, inplace=False, check_dimensionality=True):
        """Convert value from some source to destination units.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer
        :param dst: destination units.
        :type dst: UnitsContainer

        :return: converted value
        """

        if check_dimensionality:

            src_dim = self._get_dimensionality(src)
            dst_dim = self._get_dimensionality(dst)

            # If the source and destination dimensionality are different,
            # then the conversion cannot be performed.
            if src_dim != dst_dim:
                raise DimensionalityError(src, dst, src_dim, dst_dim)

        # Here src and dst have only multiplicative units left. Thus we can
        # convert with a factor.
        factor, units = self._get_root_units(src / dst)

        # factor is type float and if our magnitude is type Decimal then
        # must first convert to Decimal before we can '*' the values
        if isinstance(value, Decimal):
            factor = Decimal(str(factor))
        elif isinstance(value, Fraction):
            factor = Fraction(str(factor))

        if inplace:
            value *= factor
        else:
            value = value * factor

        return value

    def parse_unit_name(self, unit_name, case_sensitive=True):
        """Parse a unit to identify prefix, unit name and suffix
        by walking the list of prefix and suffix.

        :rtype: (str, str, str)
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
                        yield (self._prefixes[prefix].name,
                               self._units[name].name,
                               self._suffixes[suffix])
                else:
                    for real_name in self._units_casei.get(name.lower(), ()):
                        yield (self._prefixes[prefix].name,
                               self._units[real_name].name,
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
        units = self._parse_units(input_string, as_delta)
        return self.Unit(units)

    def _parse_units(self, input_string, as_delta=None):
        """
        """
        if as_delta is None:
            as_delta = True

        if as_delta and input_string in self._parse_unit_cache:
            return self._parse_unit_cache[input_string]

        if not input_string:
            return UnitsContainer()

        # Sanitize input_string with whitespaces.
        input_string = input_string.strip()

        units = ParserHelper.from_string(input_string)
        if units.scale != 1:
            raise ValueError('Unit expression cannot have a scaling factor.')

        ret = {}
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

        ret = UnitsContainer(ret)

        if as_delta:
            self._parse_unit_cache[input_string] = ret

        return ret

    def _eval_token(self, token, case_sensitive=True, **values):
        token_type = token[0]
        token_text = token[1]
        if token_type == NAME:
            if token_text == 'pi':
                return self.Quantity(math.pi)
            elif token_text == 'dimensionless':
                return 1 * self.dimensionless
            elif token_text in values:
                return self.Quantity(values[token_text])
            else:
                return self.Quantity(1, UnitsContainer({self.get_name(token_text,
                                                                      case_sensitive=case_sensitive) : 1}))
        elif token_type == NUMBER:
            return ParserHelper.eval_token(token)
        else:
            raise Exception('unknown token type')

    def parse_expression(self, input_string, case_sensitive=True, **values):
        """Parse a mathematical expression including units and return a quantity object.

        Numerical constants can be specified as keyword arguments and will take precedence
        over the names defined in the registry.
        """

        if not input_string:
            return self.Quantity(1)

        input_string = string_preprocessor(input_string)
        gen = tokenizer(input_string)

        return build_eval_tree(gen).evaluate(lambda x: self._eval_token(x,
                                                                        case_sensitive=case_sensitive,
                                                                        **values))

    __call__ = parse_expression


class NonMultiplicativeRegistry(BaseRegistry):
    """Handle of non multiplicative units (e.g. Temperature).

    Capabilities:
    - Register non-multiplicative units and their relations.
    - Convert between non-multiplicative units.

    :param default_as_delta: If True, non-multiplicative units are interpreted as
                             their *delta* counterparts in multiplications.
    :param autoconvert_offset_to_baseunit: If True, non-multiplicative units are
                                           converted to base units in multiplications.
    """

    def __init__(self, default_as_delta=True, autoconvert_offset_to_baseunit=False, **kwargs):
        super(NonMultiplicativeRegistry, self).__init__(**kwargs)

        #: When performing a multiplication of units, interpret
        #: non-multiplicative units as their *delta* counterparts.
        self.default_as_delta = default_as_delta

        # Determines if quantities with offset units are converted to their
        # base units on multiplication and division.
        self.autoconvert_offset_to_baseunit = autoconvert_offset_to_baseunit

    def _parse_units(self, input_string, as_delta=None):
        """
        """
        if as_delta is None:
            as_delta = self.default_as_delta

        return super(NonMultiplicativeRegistry, self)._parse_units(input_string, as_delta)

    def _define(self, definition):
        """Add unit to the registry.

        In addition to what is done by the BaseRegistry,
        registers also non-multiplicative units.

        :param definition: a dimension, unit or prefix definition.
        :type definition: str | Definition
        :return: Definition instance, case sensitive unit dict, case insensitive unit dict.
        :rtype: Definition, dict, dict
        """

        definition, d, di = super(NonMultiplicativeRegistry, self)._define(definition)

        # define additional units for units with an offset
        if getattr(definition.converter, "offset", 0.0) != 0.0:
            self._define_adder(definition, d, di)

        return definition, d, di

    def _is_multiplicative(self, u):
        if u in self._units:
            return self._units[u].is_multiplicative

        # If the unit is not in the registry might be because it is not
        # registered with its prefixed version.
        # TODO: Might be better to register them.
        l = self._dedup_candidates(self.parse_unit_name(u))
        try:
            u = l[0][1]
            return self._units[u].is_multiplicative
        except KeyError:
            raise UndefinedUnitError(u)

    def _validate_and_extract(self, units):

        nonmult_units = [(u, e) for u, e in units.items()
                         if not self._is_multiplicative(u)]

        # Let's validate source offset units
        if len(nonmult_units) > 1:
            # More than one src offset unit is not allowed
            raise ValueError('more than one offset unit.')

        elif len(nonmult_units) == 1:
            # A single src offset unit is present. Extract it
            # But check that:
            # - the exponent is 1
            # - is not used in multiplicative context
            nonmult_unit, exponent = nonmult_units.pop()

            if exponent != 1:
                raise ValueError('offset units in higher order.')

            if len(units) > 1 and not self.autoconvert_offset_to_baseunit:
                raise ValueError('offset unit used in multiplicative context.')

            return nonmult_unit

        return None

    def _convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        In addition to what is done by the BaseRegistry,
        converts between non-multiplicative units.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer
        :param dst: destination units.
        :type dst: UnitsContainer

        :return: converted value
        """

        # Conversion needs to consider if non-multiplicative (AKA offset
        # units) are involved. Conversion is only possible if src and dst
        # have at most one offset unit per dimension. Other rules are applied
        # by validate and extract.
        try:
            src_offset_unit = self._validate_and_extract(src)
        except ValueError as ex:
            raise DimensionalityError(src, dst, extra_msg=' - In source units, %s ' % ex)

        try:
            dst_offset_unit = self._validate_and_extract(dst)
        except ValueError as ex:
            raise DimensionalityError(src, dst, extra_msg=' - In destination units, %s ' % ex)

        if not (src_offset_unit or dst_offset_unit):
            return super(NonMultiplicativeRegistry, self)._convert(value, src, dst, inplace)

        src_dim = self._get_dimensionality(src)
        dst_dim = self._get_dimensionality(dst)

        # If the source and destination dimensionality are different,
        # then the conversion cannot be performed.
        if src_dim != dst_dim:
            raise DimensionalityError(src, dst, src_dim, dst_dim)

        # clean src from offset units by converting to reference
        if src_offset_unit:
            value = self._units[src_offset_unit].converter.to_reference(value, inplace)

        src = src.remove([src_offset_unit])

        # clean dst units from offset units
        dst = dst.remove([dst_offset_unit])

        # Convert non multiplicative units to the dst.
        value = super(NonMultiplicativeRegistry, self)._convert(value, src, dst, inplace, False)

        # Finally convert to offset units specified in destination
        if dst_offset_unit:
            value = self._units[dst_offset_unit].converter.from_reference(value, inplace)

        return value


class ContextRegistry(BaseRegistry):
    """Handle of Contexts.

    Conversion between units with different dimenstions according
    to previously established relations (contexts).
    (e.g. in the spectroscopy, conversion between frequency and energy is possible)

    Capabilities:
    - Register contexts.
    - Enable and disable contexts.
    - Parse @context directive.

    """

    def __init__(self, **kwargs):
        super(ContextRegistry, self).__init__(**kwargs)

        #: Map context name (string) or abbreviation to context.
        self._contexts = {}

        #: Stores active contexts.
        self._active_ctx = ContextChain()

    def _register_parsers(self):
        super(ContextRegistry, self)._register_parsers()
        self._register_parser('@context', self._parse_context)

    def _parse_context(self, ifile):
        try:
            self.add_context(Context.from_lines(ifile.block_iter(),
                                                self.get_dimensionality))
        except KeyError as e:
            raise DefinitionSyntaxError('unknown dimension {} in context'.format(str(e)))

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
        ctxs = list((self._contexts[name] if isinstance(name, string_types) else name)
                    for name in names_or_contexts)

        # Check if the contexts have been checked first, if not we make sure
        # that dimensions are expressed in terms of base dimensions.
        for ctx in ctxs:
            if getattr(ctx, '_checked', False):
                continue
            for (src, dst), func in ctx.funcs.items():
                src_ = self._get_dimensionality(src)
                dst_ = self._get_dimensionality(dst)
                if src != src_ or dst != dst_:
                    ctx.remove_transformation(src, dst)
                    ctx.add_transformation(src_, dst_, func)
            ctx._checked = True

        # and create a new one with the new defaults.
        ctxs = tuple(Context.from_context(ctx, **kwargs)
                     for ctx in ctxs)

        # Finally we add them to the active context.
        self._active_ctx.insert_contexts(*ctxs)
        self._build_cache()

    def disable_contexts(self, n=None):
        """Disable the last n enabled contexts.
        """
        if n is None:
            n = len(self._contexts)
        self._active_ctx.remove_contexts(n)
        self._build_cache()

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

    def with_context(self, name, **kw):
        """Decorator to wrap a function call in a Pint context.

        Use it to ensure that a certain context is active when
        calling a function::

            >>> @ureg.with_context('sp')
            ... def my_cool_fun(wavelenght):
            ...     print('This wavelength is equivalent to: %s', wavelength.to('terahertz'))


        :param names: name of the context.
        :param kwargs: keyword arguments for the contexts.
        :return: the wrapped function.
        """
        def decorator(func):
            assigned = tuple(attr for attr in functools.WRAPPER_ASSIGNMENTS if hasattr(func, attr))
            updated = tuple(attr for attr in functools.WRAPPER_UPDATES if hasattr(func, attr))

            @functools.wraps(func, assigned=assigned, updated=updated)
            def wrapper(*values, **kwargs):
                with self.context(name, **kw):
                    return func(*values, **kwargs)

            return wrapper

        return decorator

    def _convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        In addition to what is done by the BaseRegistry,
        converts between units with different dimensions by following
        transformation rules defined in the context.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer
        :param dst: destination units.
        :type dst: UnitsContainer

        :return: converted value
        """

        # If there is an active context, we look for a path connecting source and
        # destination dimensionality. If it exists, we transform the source value
        # by applying sequentially each transformation of the path.
        if self._active_ctx:

            src_dim = self._get_dimensionality(src)
            dst_dim = self._get_dimensionality(dst)

            path = find_shortest_path(self._active_ctx.graph, src_dim, dst_dim)
            if path:
                src = self.Quantity(value, src)
                for a, b in zip(path[:-1], path[1:]):
                    src = self._active_ctx.transform(a, b, self, src)

                value, src = src._magnitude, src._units

        return super(ContextRegistry, self)._convert(value, src, dst, inplace)

    def _get_compatible_units(self, input_units, group_or_system):
        """
        """

        src_dim = self._get_dimensionality(input_units)

        ret = super(ContextRegistry, self)._get_compatible_units(input_units, group_or_system)

        if self._active_ctx:
            nodes = find_connected_nodes(self._active_ctx.graph, src_dim)
            if nodes:
                for node in nodes:
                    ret |= self._dimensional_equivalents[node]

        return ret


class SystemRegistry(BaseRegistry):
    """Handle of Systems and Groups.

    Conversion between units with different dimenstions according
    to previously established relations (contexts).
    (e.g. in the spectroscopy, conversion between frequency and energy is possible)

    Capabilities:
    - Register systems and groups.
    - List systems
    - Get or get the default system.
    - Parse @system and @group directive.

    """

    def __init__(self, system=None, **kwargs):
        super(SystemRegistry, self).__init__(**kwargs)

        #: Map system name to system.
        #: :type: dict[ str | System]
        self._systems = {}

        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self._base_units_cache = dict()

        #: Map group name to group.
        #: :type: dict[ str | Group]
        self._groups = {}
        self.Group = systems.build_group_class(self)
        self._groups['root'] = self.Group('root')
        self.System = systems.build_system_class(self)

        self._default_system = system

    def _after_init(self):
        super(SystemRegistry, self)._after_init()

        #: Copy units in root group to the default group
        if 'group' in self._defaults:
            grp = self.get_group(self._defaults['group'], True)
            grp.add_units(*self.get_group('root', False).non_inherited_unit_names)

        #: System name to be used by default.
        self._default_system = self._default_system or self._defaults.get('system', None)

    def _register_parsers(self):
        super(SystemRegistry, self)._register_parsers()
        self._register_parser('@group', self._parse_group)
        self._register_parser('@system', self._parse_system)

    def _parse_group(self, ifile):
        self.Group.from_lines(ifile.block_iter(), self.define)

    def _parse_system(self, ifile):
        self.System.from_lines(ifile.block_iter(), self.get_root_units)

    def get_group(self, name, create_if_needed=True):
        """Return a Group.

        :param name: Name of the group to be
        :param create_if_needed: Create a group if not Found. If False, raise an Exception.
        :return: Group
        """
        if name in self._groups:
            return self._groups[name]

        if not create_if_needed:
            raise ValueError('Unkown group %s' % name)

        return self.Group(name)

    @property
    def sys(self):
        return systems.Lister(self._systems)

    @property
    def default_system(self):
        return self._default_system

    @default_system.setter
    def default_system(self, name):
        if name:
            if name not in self._systems:
                raise ValueError('Unknown system %s' % name)

            self._base_units_cache = {}

        self._default_system = name

    def get_system(self, name, create_if_needed=True):
        """Return a Group.

        :param name: Name of the group to be
        :param create_if_needed: Create a group if not Found. If False, raise an Exception.
        :return: System
        """
        if name in self._systems:
            return self._systems[name]

        if not create_if_needed:
            raise ValueError('Unkown system %s' % name)

        return self.System(name)

    def _define(self, definition):

        # In addition to the what is done by the BaseRegistry,
        # this adds all units to the `root` group.

        definition, d, di = super(SystemRegistry, self)._define(definition)

        if isinstance(definition, UnitDefinition):
            # We add all units to the root group
            self.get_group('root').add_units(definition.name)

        return definition, d, di

    def get_base_units(self, input_units, check_nonmult=True, system=None):
        """Convert unit or dict of units to the base units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        Unlike BaseRegistry, in this registry root_units might be different
        from base_units

        :param input_units: units
        :type input_units: UnitsContainer or str
        :param check_nonmult: if True, None will be returned as the
                              multiplicative factor if a non-multiplicative
                              units is found in the final Units.
        :return: multiplicative factor, base units
        """

        input_units = to_units_container(input_units)

        f, units = self._get_base_units(input_units, check_nonmult, system)

        return f, self.Unit(units)

    def _get_base_units(self, input_units, check_nonmult=True, system=None):

        if system is None:
            system = self._default_system

        # The cache is only done for check_nonmult=True and the current system.
        if check_nonmult and system == self._default_system and input_units in self._base_units_cache:
            return self._base_units_cache[input_units]

        factor, units = self.get_root_units(input_units, check_nonmult)

        if not system:
            return factor, units

        # This will not be necessary after integration with the registry
        # as it has a UnitsContainer intermediate
        units = to_units_container(units, self)

        destination_units = UnitsContainer()

        bu = self.get_system(system, False).base_units

        for unit, value in units.items():
            if unit in bu:
                new_unit = bu[unit]
                new_unit = to_units_container(new_unit, self)
                destination_units *= new_unit ** value
            else:
                destination_units *= UnitsContainer({unit: value})

        base_factor = self.convert(factor, units, destination_units)

        if check_nonmult:
            self._base_units_cache[input_units] = base_factor, destination_units

        return base_factor, destination_units

    def _get_compatible_units(self, input_units, group_or_system):
        """
        """

        if group_or_system is None:
            group_or_system = self._default_system

        ret = super(SystemRegistry, self)._get_compatible_units(input_units, group_or_system)

        if group_or_system:
            if group_or_system in self._systems:
                members = self._systems[group_or_system].members
            elif group_or_system in self._groups:
                members = self._groups[group_or_system].members
            else:
                raise ValueError("Unknown Group o System with name '%s'" % group_or_system)
            return frozenset(ret.intersection(members))

        return ret


class UnitRegistry(SystemRegistry, ContextRegistry, NonMultiplicativeRegistry):
    """The unit registry stores the definitions and relationships between units.

    :param filename: path of the units definition file to load or line-iterable object.
                     Empty to load the default definition file.
                     None to leave the UnitRegistry empty.
    :param force_ndarray: convert any input, scalar or not to a numpy.ndarray.
    :param default_as_delta: In the context of a multiplication of units, interpret
                             non-multiplicative units as their *delta* counterparts.
    :param autoconvert_offset_to_baseunit: If True converts offset units in quantites are
                                           converted to their base units in multiplicative
                                           context. If False no conversion happens.
    :param on_redefinition: action to take in case a unit is redefined.
                            'warn', 'raise', 'ignore'
    :type on_redefinition: str
    :param auto_reduce_dimensions: If True, reduce dimensionality on appropriate operations.
    """

    def __init__(self, filename='', force_ndarray=False, default_as_delta=True,
                 autoconvert_offset_to_baseunit=False,
                 on_redefinition='warn', system=None,
                 auto_reduce_dimensions=False):

        super(UnitRegistry, self).__init__(filename=filename, force_ndarray=force_ndarray,
                                           on_redefinition=on_redefinition,
                                           default_as_delta=default_as_delta,
                                           autoconvert_offset_to_baseunit=autoconvert_offset_to_baseunit,
                                           system=system,
                                           auto_reduce_dimensions=auto_reduce_dimensions)

    def pi_theorem(self, quantities):
        """Builds dimensionless quantities using the Buckingham π theorem
        :param quantities: mapping between variable name and units
        :type quantities: dict
        :return: a list of dimensionless quantities expressed as dicts
        """
        return pi_theorem(quantities, self)

    def setup_matplotlib(self, enable=True):
        """Set up handlers for matplotlib's unit support.
        :param enable: whether support should be enabled or disabled
        :type enable: bool
        """
        # Delays importing matplotlib until it's actually requested
        from .matplotlib import setup_matplotlib_handlers
        setup_matplotlib_handlers(self, enable)

    wraps = registry_helpers.wraps

    check = registry_helpers.check


class LazyRegistry(object):

    def __init__(self, args=None, kwargs=None):
        self.__dict__['params'] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__['params']
        kwargs['on_redefinition'] = 'raise'
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()

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
