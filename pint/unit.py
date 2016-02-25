# -*- coding: utf-8 -*-
"""
    pint.unit
    ~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import math
import itertools
import operator
import pkg_resources
from decimal import Decimal
from fractions import Fraction
from contextlib import contextmanager, closing
from io import open, StringIO
from collections import defaultdict
from tokenize import untokenize, NUMBER, STRING, NAME, OP
from numbers import Number

from . import registry_helpers
from .context import Context, ContextChain
from .util import (logger, pi_theorem, solve_dependencies, ParserHelper,
                   string_preprocessor, find_connected_nodes,
                   find_shortest_path, UnitsContainer, _is_dim,
                   SharedRegistryObject, to_units_container,
                   fix_str_conversions, SourceIterator)

from .compat import tokenizer, string_types, NUMERIC_TYPES, long_type
from .formatting import siunitx_format_unit
from .definitions import (Definition, UnitDefinition, PrefixDefinition,
                          DimensionDefinition)
from .converters import ScaleConverter
from .errors import (DimensionalityError, UndefinedUnitError,
                     DefinitionSyntaxError, RedefinitionError)

from .pint_eval import build_eval_tree
from . import systems


@fix_str_conversions
class _Unit(SharedRegistryObject):
    """Implements a class to describe a unit supporting math operations.

    :type units: UnitsContainer, str, Unit or Quantity.

    """

    #: Default formatting string.
    default_format = ''

    def __reduce__(self):
        return self.Unit, (self._units)

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
                            'UnitsContainer; not {0}.'.format(type(units)))

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

    def __str__(self):
        return format(self)

    def __repr__(self):
        return "<Unit('{0}')>".format(self._units)

    def __format__(self, spec):
        spec = spec or self.default_format
        # special cases
        if 'Lx' in spec: # the LaTeX siunitx code
          opts = ''
          ustr = siunitx_format_unit(self)
          ret = r'\si[%s]{%s}'%( opts, ustr )
          return ret


        if '~' in spec:
            if self.dimensionless:
                return ''
            units = UnitsContainer(dict((self._REGISTRY._get_symbol(key),
                                         value)
                                   for key, value in self._units.items()))
            spec = spec.replace('~', '')
        else:
            units = self._units

        return '%s' % (format(units, spec))

    # IPython related code
    def _repr_html_(self):
        return self.__format__('H')

    def _repr_latex_(self):
        return "$" + self.__format__('L') + "$"

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
                 on_redefinition='warn', system=None):

        self.Unit = build_unit_class(self)
        self.Quantity = build_quantity_class(self, force_ndarray)
        self.Measurement = build_measurement_class(self, force_ndarray)

        #: Action to take in case a unit is redefined. 'warn', 'raise', 'ignore'
        self._on_redefinition = on_redefinition

        #: Map between name (string) and value (string) of defaults stored in the definitions file.
        self._defaults = {}

        #: Map dimension name (string) to its definition (DimensionDefinition).
        self._dimensions = {}

        #: Map system name to system.
        #: :type: dict[ str | System]
        self._systems = {}

        #: Map group name to group.
        #: :type: dict[ str | Group]
        self._groups = {}
        self.Group = systems.build_group_class(self)
        self._groups['root'] = self.Group('root')
        self.System = systems.build_system_class(self)

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

        #: Maps dimensionality (UnitsContainer) to Units (str)
        self._dimensional_equivalents = dict()

        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self._root_units_cache = dict()

        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self._base_units_cache = dict()

        #: Maps dimensionality (UnitsContainer) to Units (UnitsContainer)
        self._dimensionality_cache = dict()

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

        #: Copy units in root group to the default group
        if 'group' in self._defaults:
            grp = self.get_group(self._defaults['group'], True)
            grp.add_units(*self.get_group('root', False).non_inherited_unit_names)

        #: System name to be used by default.
        self._default_system = system or self._defaults.get('system', None)

        self._build_cache()

    def __name__(self):
        return 'UnitRegistry'

    def __getattr__(self, item):
        return self.Unit(item)

    def __getitem__(self, item):
        logger.warning('Calling the getitem method from a UnitRegistry is deprecated. '
                       'use `parse_expression` method or use the registry as a callable.')
        return self.parse_expression(item)

    def __dir__(self):
        return list(self._units.keys()) + \
            ['define', 'load_definitions', 'get_name', 'get_symbol',
             'get_dimensionality', 'Quantity', 'wraps',
             'parse_units', 'parse_expression', 'pi_theorem',
             'convert', 'get_base_units']

    @property
    def default_format(self):
        """Default formatting string for quantities.
        """
        return self.Quantity.default_format

    @default_format.setter
    def default_format(self, value):
        self.Unit.default_format = value
        self.Quantity.default_format = value

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

    def define(self, definition, add_to_root_group=False):
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

            # We add all units to the root group
            if add_to_root_group:
                self.get_group('root').add_units(definition.name)

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
                                       d_reference, definition.is_base),
                        add_to_root_group=True)

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

        ifile = SourceIterator(file)
        for no, line in ifile:
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

            elif line.startswith('@defaults'):
                next(ifile)
                for lineno, part in ifile.block_iter():
                    k, v = part.split('=')
                    self._defaults[k.strip()] = v.strip()

            elif line.startswith('@context'):
                try:
                    self.add_context(Context.from_lines(ifile.block_iter(),
                                                        self.get_dimensionality))
                except KeyError as e:
                    raise DefinitionSyntaxError('unknown dimension {0} in context'.format(str(e)), lineno=no)

            elif line.startswith('@group'):
                self.Group.from_lines(ifile.block_iter(), self.define)

            elif line.startswith('@system'):
                self.System.from_lines(ifile.block_iter(), self.get_root_units)

            else:
                try:
                    self.define(Definition.from_string(line),
                                add_to_root_group=True)
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

                    bu = self._get_root_units(uc)
                    di = self._get_dimensionality(uc)

                    self._root_units_cache[uc] = bu
                    self._dimensionality_cache[uc] = di

                    if not prefixed:
                        if di not in self._dimensional_equivalents:
                            self._dimensional_equivalents[di] = set()

                        self._dimensional_equivalents[di].add(self._units[unit_name]._name)

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
        input_units = to_units_container(input_units)

        f, units = self._get_base_units(input_units, check_nonmult, system)

        return f, self.Unit(units)

    def _get_base_units(self, input_units, check_nonmult=True, system=None):
        """
        :param registry:
        :param input_units:
        :param check_nonmult:
        :param system: System
        :return:
        """

        if system is None:
            system = self._default_system

        # The cache is only done for check_nonmult=True and the current system.
        if check_nonmult and system == self._default_system and input_units in self._base_units_cache:
            return self._base_units_cache[input_units]

        factor, units = self.get_root_units(input_units, check_nonmult)

        if not system:
            return factor, units

        # This will not be necessary after integration with the registry as it has a UnitsContainer intermediate
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

        if group_or_system is None:
            group_or_system = self._default_system

        equiv = self._get_compatible_units(input_units, group_or_system)

        return frozenset(self.Unit(eq) for eq in equiv)

    def _get_compatible_units(self, input_units, group_or_system):
        """
        """
        if not input_units:
            return frozenset()

        src_dim = self._get_dimensionality(input_units)

        ret = self._dimensional_equivalents[src_dim]

        if self._active_ctx:
            nodes = find_connected_nodes(self._active_ctx.graph, src_dim)
            ret = set()
            if nodes:
                for node in nodes:
                    ret |= self._dimensional_equivalents[node]

        if group_or_system:
            if group_or_system in self._systems:
                members = self._systems[group_or_system].members
            elif group_or_system in self._groups:
                members = self._groups[group_or_system].members
            else:
                raise ValueError("Unknown Group o System with name '%s'" % group_or_system)
            return frozenset(ret.intersection(members))

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

        return self._convert(value, src, dst, inplace)

    def _convert(self, value, src, dst, inplace=False):
        """Convert value from some source to destination units.

        :param value: value
        :param src: source units.
        :type src: UnitsContainer
        :param dst: destination units.
        :type dst: UnitsContainer

        :return: converted value
        """
        if src == dst:
            return value

        src_dim = self._get_dimensionality(src)
        dst_dim = self._get_dimensionality(dst)

        # If there is an active context, we look for a path connecting source and
        # destination dimensionality. If it exists, we transform the source value
        # by applying sequentially each transformation of the path.
        if self._active_ctx:
            path = find_shortest_path(self._active_ctx.graph, src_dim, dst_dim)
            if path:
                src = self.Quantity(value, src)
                for a, b in zip(path[:-1], path[1:]):
                    src = self._active_ctx.transform(a, b, self, src)

                value, src = src._magnitude, src._units

                src_dim = self._get_dimensionality(src)

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

        # TODO: Shouldn't this (until factor, units) be inside the If above?

        # clean src from offset units by converting to reference
        for u, e in src_offset_units:
            value = self._units[u].converter.to_reference(value, inplace)
        src = src.remove([u for u, e in src_offset_units])

        # clean dst units from offset units
        dst = dst.remove([u for u, e in dst_offset_units])

        # Here src and dst have only multiplicative units left. Thus we can
        # convert with a factor.
        factor, units = self._get_root_units(src / dst)

        # factor is type float and if our magnitude is type Decimal then
        # must first convert to Decimal before we can '*' the values
        if isinstance(value, Decimal):
            factor = Decimal(str(factor))

        if isinstance(value, Fraction):
            factor = Fraction(str(factor))

        if inplace:
            value *= factor
        else:
            value = value * factor

        # Finally convert to offset units specified in destination
        for u, e in dst_offset_units:
            value = self._units[u].converter.from_reference(value, inplace)

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
        units = self._parse_units(input_string, as_delta)
        return self.Unit(units)

    def _parse_units(self, input_string, as_delta=None):
        """
        """
        if as_delta is None:
            as_delta = self.default_as_delta

        if as_delta and input_string in self._parse_unit_cache:
            return self._parse_unit_cache[input_string]

        if not input_string:
            return UnitsContainer()

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
        
        return build_eval_tree(gen).evaluate(lambda x : self._eval_token(x, case_sensitive=case_sensitive,
                                                                          **values))

    __call__ = parse_expression

    wraps = registry_helpers.wraps

    check = registry_helpers.check


def build_unit_class(registry):

    class Unit(_Unit):
        pass

    Unit._REGISTRY = registry
    return Unit


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
