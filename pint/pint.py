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

import os
import sys
import copy
import math
import logging
import operator
import functools
import itertools


from collections import Iterable

from io import BytesIO
from numbers import Number
import tokenize
from tokenize import untokenize, NUMBER, STRING, NAME, OP

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if sys.version < '3':
    from io import open
    from StringIO import StringIO
    string_types = basestring
    _tokenize = lambda input: tokenize.generate_tokens(StringIO(input).readline)
else:
    string_types = str
    _tokenize = lambda input: tokenize.tokenize(BytesIO(input.encode('utf-8')).readline)

PRETTY = '⁰¹²³⁴⁵⁶⁷⁸⁹·⁻'

try:
    import numpy as np
    from numpy import ndarray
    from numpy.linalg import svd

    HAS_NUMPY = True
    NUMERIC_TYPES = (Number, ndarray)

    def _to_magnitude(value, force_ndarray=False):
        if value is None:
            return value
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
        if force_ndarray:
            return np.asarray(value)
        return value

    def nullspace(A, atol=1e-7, rtol=0):
        """Compute an approximate basis for the nullspace of A.

        The algorithm used by this function is based on the singular value
        decomposition of `A`.
        """

        A = np.atleast_2d(A)
        u, s, vh = svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

except ImportError:

    class ndarray(object):
        pass

    HAS_NUMPY = False
    NUMERIC_TYPES = (Number, )

    def _to_magnitude(value, force_ndarray=False):
        return value

    def nullspace(A, atol=1e-13, rtol=0):
        raise Exception('NumPy is required for this operation.')


def _eq(first, second):
    """Comparison of scalars and arrays
    """
    out = first == second
    if isinstance(out, Iterable):
        if isinstance(out, ndarray):
            return np.all(out)
        else:
            return all(out)
    return out


def _definitions_from_file(filename):
    """Load definition from file.
    """
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


class _Exception(Exception):

    def __init__(self, internal):
        self.internal = internal


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


def pi_theorem(quantities, registry=None):
    """Builds dimensionless quantities using the Buckingham π theorem

    :param quantities: mapping between variable name and units
    :type quantities: dict
    :return: a list of dimensionless quantities expressed as dicts
    """
    if registry is None:
        registry = _DEFAULT_REGISTRY

    # Preprocess input
    quant = []
    dimensions = set()
    for name, value in quantities.items():
        if isinstance(value, UnitsContainer):
            if any((not unit.startswith('[') for unit in value)):
                dims = registry.Quantity(1, value).dimensionality
            else:
                dims = value
        elif not hasattr(value, 'dimensionality'):
            dims = registry[value].dimensionality
        else:
            dims = value.dimensionality

        quant.append((name, dims))
        dimensions = dimensions.union(dims.keys())

    dimensions = list(dimensions)

    # Calculate dimensionless  quantities
    M = np.zeros((len(dimensions), len(quant)))

    for row, dimension in enumerate(dimensions):
        for col, (name, dimensionality) in enumerate(quant):
            M[row, col] = dimensionality[dimension]

    kernel = np.atleast_2d(nullspace(M))

    # Sanitizing output.
    _decimals = 7
    kernel[abs(kernel) < 10.**-_decimals] = 0
    for col in range(kernel.shape[1]):
        vector = kernel[:, col]
        if sum(vector < 0) > sum(vector > 0):
            vector = -vector
        vector /= min(abs(vector[vector > 0]))
        kernel[:, col] = vector

    kernel = np.round(kernel, _decimals)

    result = []
    for col in range(kernel.shape[1]):
        r = {}
        for row in range(kernel.shape[0]):
            if kernel[row, col] != 0:
                r[quant[row][0]] = kernel[row, col]
        result.append(r)
    return result


def formatter(items, product_symbol=' * ', power_format=' ** {:n}',
              as_ratio=True, single_denominator=False):
    """Format a list of (name, exponent) pairs.

    :param items: a list of (name, exponent) pairs.
    :param product_symbol: the symbol used for multiplication.
    :param power_format: the symbol used for exponentiation including a,
                         formatting place holder for the power.
    :param as_ratio: True to display as ratio, False as negative powers.
    :param single_denominator: put

    :return: the formulas as a string.
    """
    if as_ratio:
        fun = abs
    else:
        fun = lambda x: x

    tmp_plus = []
    tmp_minus = []
    for key, value in sorted(items):
        if value == 1:
            tmp_plus.append(key)
        elif value > 1:
            tmp_plus.append(key + power_format.format(value))
        elif value == -1:
            tmp_minus.append(key)
        else:
            tmp_minus.append(key + power_format.format(fun(value)))

    if tmp_plus:
        ret = product_symbol.join(tmp_plus)
    elif as_ratio and tmp_minus:
        ret = '1'
    else:
        ret = ''

    if tmp_minus:
        if as_ratio:
            ret += ' / '
            if single_denominator:
                ret += ' / '.join(tmp_minus)
            else:
                ret += product_symbol.join(tmp_minus)
        else:
            ret += product_symbol.join(tmp_minus)

    return ret


class AliasDict(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.preferred_alias = {}

    def add_alias(self, key, value, preferred=False):
        if value not in self:
            raise IndexError("The aliased value '{}' is not present in the dictionary".format(value))
        self[key] = value = self.get_aliased(value)
        if preferred:
            self.preferred_alias[value] = key

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
            if not isinstance(value, NUMERIC_TYPES):
                raise TypeError('value must be a NUMERIC_TYPES, not {}'.format(type(value)))
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
            tmp = formatter(self.items(), r' \cdot ', '^[{:n}]', True, True).replace('[', '{').replace(']', '}')
            if '/' in tmp:
                return r'\frac{%s}' % tmp.replace(' / ', '}{')
            return tmp
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
    :param force_ndarray: convert any input, scalar or not to a numpy.ndarray.
    """

    #: Map unit name (string) to unit value (Quantity), and unit alias to canonical unit name
    _UNITS = AliasDict()

    #: Map prefix name (string) to prefix value (float), and unit alias to canonical prefix name
    _PREFIXES = AliasDict({'': 1})

    #: Map suffix name (string) to canonical , and unit alias to canonical unit name
    _SUFFIXES = AliasDict({'': None, 's': ''})

    def __init__(self, filename='', force_ndarray=False):
        self.Quantity = _build_quantity_class(self, force_ndarray)
        self._definition_files = []
        if filename == '':
            self.add_from_file(os.path.join(os.path.dirname(__file__), 'default_en.txt'))
        elif filename is not None:
            self.add_from_file(filename)

    def __getattr__(self, item):
        return self.Quantity(1, item)

    def __getitem__(self, item):
        return self._parse_expression(item)

    def add_unit(self, name, value, aliases=tuple(), **modifiers):
        """Add unit to the registry.
        """
        if not isinstance(value, self.Quantity):
            value = self.Quantity(value, **modifiers)

        self._UNITS[name] = value

        for ndx, alias in enumerate(aliases):
            if ' ' in alias:
                logger.warn('Alias cannot contain a space ' + alias)
            self._UNITS.add_alias(alias.strip(), name, not ndx)

    def add_prefix(self, name, value, aliases=tuple()):
        """Add prefix to the registry.
        """

        if not isinstance(value, NUMERIC_TYPES):
            value = eval(value, {'__builtins__': None}, {})
        self._PREFIXES[name] = float(value)

        for ndx, alias in enumerate(aliases):
            self._PREFIXES.add_alias(alias.strip(), name, not ndx)

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
                    #value = value.strip('[]')
                    if value == '[]':
                        value = ''
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
                logger.error("Exception: Cannot add '{}' {}".format(name, ex))

        dep2 = {}
        for unit_name, deps in dependencies.items():
            dep2[unit_name] = set(conv[dep_name] for dep_name in deps)

        for unit_names in _solve_dependencies(dep2):
            for unit_name in unit_names:
                if not unit_name in self._UNITS:
                    self.add_unit(unit_name, *pending[unit_name])

    def get_alias(self, name):
        """Return the preferred alias for a unit
        """
        candidates = self._dedup_candidates(self._parse_candidate(name))
        if not candidates:
            raise UndefinedUnitError(name)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logging.warning('Parsing {} yield multiple results. '
                            'Options are: {!r}'.format(name, candidates))
            prefix, unit_name, _ = candidates[0]

        return self._PREFIXES.preferred_alias.get(prefix, prefix) + \
               self._UNITS.preferred_alias.get(unit_name, unit_name)

    def _to_canonical(self, candidate):
        """Return the canonical name of a unit.
        """

        if candidate == 'dimensionless':
            return ''

        try:
            return self._UNITS.get_aliased(candidate)
        except KeyError:
            pass

        candidates = self._dedup_candidates(self._parse_candidate(candidate))
        if not candidates:
            raise UndefinedUnitError(candidate)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning('Parsing {} yield multiple results. '
                           'Options are: {}'.format(candidate, candidates))
            prefix, unit_name, _ = candidates[0]

        if prefix:
            alias = self.get_alias(prefix + unit_name)
            if prefix + unit_name == 'kilogram':
                pass
            self.add_unit(prefix + unit_name, self.Quantity(self._PREFIXES[prefix], unit_name), (alias, ))
            return prefix + unit_name

        return unit_name

    def _dedup_candidates(self, candidates):
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

    def _parse_candidate(self, candidate):
        """Parse a unit to identify prefix, suffix and unit name
        by walking the list of prefix and suffix.
        """
        for suffix, prefix in itertools.product(self._SUFFIXES.keys(), self._PREFIXES.keys()):
            if candidate.startswith(prefix) and candidate.endswith(suffix):
                unit_name = candidate[len(prefix):]
                if suffix:
                    unit_name = unit_name[:-len(suffix)]
                    if len(unit_name) == 1:
                        continue
                if unit_name in self._UNITS:
                    yield (self._PREFIXES.get_aliased(prefix),
                           self._UNITS.get_aliased(unit_name),
                           self._SUFFIXES.get_aliased(suffix))

    def _parse_expression(self, input):
        """Parse expression mathematical units and return a quantity object.
        """

        if not input:
            return self.Quantity(1)

        gen = _tokenize(input)
        result = []
        unknown = set()
        for toknum, tokval, _, _, _  in gen:
            if toknum in (STRING, NAME):  # replace NUMBER tokens
                # TODO: Integrate math better, Replace eval
                if tokval == 'pi':
                    result.append((toknum, str(math.pi)))
                    continue
                try:
                    tokval = self._to_canonical(tokval)
                except UndefinedUnitError as ex:
                    unknown.add(ex.unit_names)
                if tokval:
                    result.extend([
                        (NAME, 'Q_'),
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
                    result.extend([
                        (NAME, 'Q_'),
                        (OP, '('),
                        (NUMBER, '1'),
                        (OP, ','),
                        (NAME, 'U_'),
                        (OP, '('),
                        (OP, ')'),
                        (OP, ')')
                    ])
            else:
                result.append((toknum, tokval))

        if unknown:
            raise UndefinedUnitError(unknown)

        return eval(untokenize(result), {'__builtins__': None},
                                        {'REGISTRY': self._UNITS,
                                         'Q_': self.Quantity,
                                         'U_': UnitsContainer})


def _build_quantity_class(registry, force_ndarray):
    """Create a Quantity Class.
    """

    @functools.total_ordering
    class _Quantity(object):
        """Quantity object constituted by magnitude and units.

        :param value: value of the physical quantity to be created.
        :type value: str, Quantity or any numeric type.
        :param units: units of the physical quantity to be created.
        :type units: UnitsContainer, str or Quantity.
        """

        _REGISTRY = registry

        def __reduce__(self):
            return _build_quantity, (self.magnitude, self.units)

        def __new__(cls, value, units=None, offset=0, log_base=0):
            if units is None:
                if isinstance(value, string_types):
                    inst = cls._REGISTRY._parse_expression(value)
                elif isinstance(value, cls):
                    inst = copy.copy(value)
                else:
                    inst = object.__new__(cls)
                    inst._magnitude = _to_magnitude(value, force_ndarray)
                    inst._units = UnitsContainer()
            elif isinstance(units, UnitsContainer):
                inst = object.__new__(cls)
                inst._magnitude = _to_magnitude(value, force_ndarray)
                inst._units = units
            elif isinstance(units, string_types):
                inst = cls._REGISTRY._parse_expression(units)
                inst._magnitude = _to_magnitude(value, force_ndarray)
            elif isinstance(units, cls):
                inst = copy.copy(units)
                inst._magnitude = _to_magnitude(value, force_ndarray)
            else:
                raise TypeError('units must be of type str, Quantity or UnitsContainer; not {}.'.format(type(units)))

            return inst

        def __copy__(self):
            return self.__class__(copy.copy(self._magnitude), copy.copy(self._units))

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

            if conv.endswith('~'):
                units = UnitsContainer({self._REGISTRY.get_alias(key): value
                                       for key, value in self.units.items()})
                conv = conv[:-1]
            else:
                units = self.units

            return format(self.magnitude, fmt) + ' ' + format(units, conv)

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
            tmp = copy.copy(self).convert_to_reference()

            return not bool(tmp.dimensionality)

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
                return self

            factor = self.__class__(1, self.units / other.units)
            factor = factor.convert_to_reference()

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

        def _convert_to_reference(self, input_units):

            factor = 1
            units = UnitsContainer()
            for key, value in input_units.items():
                reg = self._REGISTRY._UNITS[key]
                if reg._magnitude is None:
                    units.add(key, value)
                else:
                    fac, uni = self._convert_to_reference(reg.units)
                    factor *= (reg._magnitude * fac) ** value
                    units *= uni ** value

            return factor, units

        def convert_to_reference(self):
            """Return Quantity rescaled to reference units.
            """

            factor, units = self._convert_to_reference(self.units)

            return self.__class__(self.magnitude * factor, units)

        def convert_to_reference2(self):
            """Return Quantity rescaled to reference units.
            """

            tmp = self.__class__(self.magnitude)

            for key, value in self.units.items():
                reg = self._REGISTRY._UNITS[key]
                if reg._magnitude is None:
                    factor = self.__class__(1, key) ** value
                else:
                    factor = reg.convert_to_reference() ** value

                tmp = tmp * factor

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
                if self.dimensionless:
                    self._magnitude = fun(self._magnitude, _to_magnitude(other, force_ndarray))
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
                self._magnitude *= _to_magnitude(other, force_ndarray)

            return self

        def __mul__(self, other):
            if isinstance(other, self.__class__):
                return self.__class__(self._magnitude * other._magnitude, self._units * other._units)
            else:
                return self.__class__(self._magnitude * other, self._units)

        __rmul__ = __mul__

        def __itruediv__(self, other):
            if isinstance(other, self.__class__):
                self._magnitude /= other._magnitude
                self._units /= other._units
            else:
                self._magnitude /= _to_magnitude(other, force_ndarray)

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
            if isinstance(other, self.__class__):
                self._magnitude //= other._magnitude
                self._units /= other._units
            else:
                self._magnitude //= _to_magnitude(other, force_ndarray)

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
            self._magnitude **= _to_magnitude(other, force_ndarray)
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
                return self.dimensionless and _eq(self.magnitude, other)

            if _eq(self._magnitude, 0) and _eq(other._magnitude, 0):
                return self.dimensionality == other.dimensionality

            if self._units == other._units:
                return _eq(self._magnitude, other._magnitude)

            try:
                return _eq(self.to(other).magnitude, other._magnitude)
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


        # Experimental NumPy Support

        #: Dictionary mapping ufunc/attributes names to the units that they
        #: require (conversion will be tried).
        __require_units = {'cumprod': '',
                           'arccos': '', 'arcsin': '', 'arctan': '',
                           'arccosh': '', 'arcsinh': '', 'arctanh': '',
                           'exp': '', 'expm1': '',
                           'log': '', 'log10': '', 'log1p': '', 'log2': '',
                           'sin': 'radian', 'cos': 'radian', 'tan': 'radian',
                           'sinh': 'radian', 'cosh': 'radian', 'tanh': 'radian',
                           'radians': 'degree', 'degrees': 'radian',
                           'add': '', 'subtract': ''}

        #: Dictionary mapping ufunc/attributes names to the units that they
        #: will set on output.
        __set_units = {'cos': '', 'sin': '', 'tan': '',
                       'cosh': '', 'sinh': '', 'tanh': '',
                       'arccos': 'radian', 'arcsin': 'radian',
                       'arctan': 'radian', 'arctan2': 'radian',
                       'arccosh': 'radian', 'arcsinh': 'radian',
                       'arctanh': 'radian',
                       'degrees': 'degree', 'radians': 'radian',
                       'expm1': '', 'cumprod': ''}

        #: List of ufunc/attributes names in which units are copied from the
        #: original.
        __copy_units = 'compress conj conjugate copy cumsum diagonal flatten ' \
                       'max mean min ptp ravel repeat reshape round ' \
                       'squeeze std sum take trace transpose ' \
                       'ceil floor hypot rint' \
                       'add subtract multiply'.split()

        #: Dictionary mapping ufunc/attributes names to the units that they will
        #: set on output. The value is interpreded as the power to which the unit
        #: will be raised.
        __prod_units = {'var': 2, 'prod': 'size', 'true_divide': -1, 'divide': -1}


        __skip_other_args = 'left_shift right_shift'.split()

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

        def __getattr__(self, item):
            if item.startswith('__array_'):
                if isinstance(self._magnitude, ndarray):
                    return getattr(self._magnitude, item)
                else:
                    raise AttributeError('__array_* attributes are only taken from ndarray objects.')
            try:
                attr = getattr(self._magnitude, item)
                if callable(attr):
                    return functools.partial(self.__numpy_method_wrap, attr)
                return attr
            except AttributeError:
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
                    factor = self.__class__(value.magnitude, value.units / self.units).convert_to_reference()
                else:
                    factor = self.__class__(value, self._units ** (-1)).convert_to_reference()

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

        __array_priority__ = 21

        def __array_prepare__(self, obj, context=None):
            uf, objs, huh = context
            if uf.__name__ in self.__require_units:
                self.__ito_if_needed(self.__require_units[uf.__name__])
            elif len(objs) > 1 and uf.__name__ not in self.__skip_other_args:
                to_units = objs[0]
                objs = (to_units, ) + \
                       tuple((other if self.units == other.units else other.to(self)
                              for other in objs[1:]))
            return self.magnitude.__array_prepare__(obj, (uf, objs, huh))

        def __array_wrap__(self, obj, context=None):
            try:
                uf, objs, huh = context
                out = self.magnitude.__array_wrap__(obj, context)
                if uf.__name__ in self.__set_units:
                    try:
                        out = self.__class__(out, self.__set_units[uf.__name__])
                    except:
                        raise _Exception(ValueError)
                elif uf.__name__ in self.__copy_units:
                    try:
                        out = self.__class__(out, self.units)
                    except:
                        raise _Exception(ValueError)
                elif uf.__name__ in self.__prod_units:
                    tmp = self.__prod_units[uf.__name__]
                    if tmp == 'size':
                        out = self.__class__(out, self.units ** self._magnitude.size)
                    else:
                        out = self.__class__(out, self.units ** tmp)
                return out
            except _Exception as ex:
                raise ex.internal
            except Exception as ex:
                print(ex)
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

            return Measurement(self, error)

    return _Quantity

class Measurement(object):

    def __init__(self, value, error):
        """
        :param value: The most likely value of the measurement.
        :type value: Quantity or Number
        :param error: The error or uncertainty of the measurement.
        :type value: Quantity or Number
        """
        if not (value/error).unitless:
            raise ValueError('{} and {} have incompatible units'.format(value, error))
        try:
            emag = error.magnitude
        except AttributeError:
            emag = error

        if emag < 0:
            raise ValueError('The magnitude of the error cannot be negative'.format(value, error))

        self._value = value
        self._error = error

    @property
    def value(self):
        return self._value

    @property
    def error(self):
        return self._error

    @property
    def rel(self):
        return float(abs(self._error / self._value))

    def _add_sub(self, other, operator):
        result = self.value + other.value
        if isinstance(other, self.__class__):
            error = (self.error ** 2.0 + other.error ** 2.0) ** (1/2)
        else:
            error = self.error
        return result.plus_minus(error)

    def __add__(self, other):
        return self._add_sub(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._add_sub(other, operator.sub)

    __rsub__ = __sub__

    def _mul_div(self, other, operator):
        if isinstance(other, self.__class__):
            result = operator(self.value, other.value)
            return result.plus_minus((self.rel ** 2.0 + other.rel ** 2.0) ** (1/2), relative=True)
        else:
            result = operator(self.value, other)
            return result.plus_minus(abs(operator(self.error, other)))

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._mul_div(other, operator.truediv)

    def __floordiv__(self, other):
        return self._mul_div(other, operator.floordiv)

    __div__ = __floordiv__

    def __str__(self):
        return '{}'.format(self)

    def __repr__(self):
        return "<Measurement({:!r}, {:!r})>".format(self._value, self._error)

    def __format__(self, spec):
        if '!' in spec:
            fmt, conv = spec.split('!')
            conv = '!' + conv
        else:
            fmt, conv = spec, ''

        left, right = '(', ')'
        if '!l' == conv:
            pm = r'\pm'
            left = r'\left' + left
            right = r'\right' + right
        elif '!p' == conv:
            pm = '±'
        else:
            pm = '+/-'

        if hasattr(self.value, 'units'):
            vmag = format(self.value.magnitude, fmt)
            if self.value.units != self.error.units:
                emag = self.error.to(self.value.units).magnitude
            else:
                emag = self.error.magnitude
            emag = format(emag, fmt)
            units = ' ' + format(self.value.units, conv)
        else:
            vmag, emag, units = self.value, self.error, ''

        return left + vmag + ' ' + pm + ' ' +  emag + right + units if units else ''


_DEFAULT_REGISTRY = UnitRegistry()
def _build_quantity(value, units):
    return _DEFAULT_REGISTRY.Quantity(value, units)
