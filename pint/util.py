# -*- coding: utf-8 -*-
"""
    pint.util
    ~~~~~~~~~

    Miscellaneous functions for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

from decimal import Decimal
import locale
import sys
import re
import operator
from numbers import Number
from fractions import Fraction

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

from logging import NullHandler

import logging
from token import STRING, NAME, OP, NUMBER
from tokenize import untokenize

from .compat import string_types, tokenizer, lru_cache, maketrans, NUMERIC_TYPES
from .formatting import format_unit,siunitx_format_unit
from .pint_eval import build_eval_tree
from .errors import DefinitionSyntaxError

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def matrix_to_string(matrix, row_headers=None, col_headers=None, fmtfun=lambda x: str(int(x))):
    """Takes a 2D matrix (as nested list) and returns a string.
    """
    ret = []
    if col_headers:
        ret.append(('\t' if row_headers else '') + '\t'.join(col_headers))
    if row_headers:
        ret += [rh + '\t' + '\t'.join(fmtfun(f) for f in row)
                for rh, row in zip(row_headers, matrix)]
    else:
        ret += ['\t'.join(fmtfun(f) for f in row)
                for row in matrix]

    return '\n'.join(ret)


def transpose(matrix):
    """Takes a 2D matrix (as nested list) and returns the transposed version.
    """
    return [list(val) for val in zip(*matrix)]


def column_echelon_form(matrix, ntype=Fraction, transpose_result=False):
    """Calculates the column echelon form using Gaussian elimination.

    :param matrix: a 2D matrix as nested list.
    :param ntype: the numerical type to use in the calculation.
    :param transpose_result: indicates if the returned matrix should be transposed.
    :return: column echelon form, transformed identity matrix, swapped rows
    """
    lead = 0

    M = transpose(matrix)

    _transpose = transpose if transpose_result else lambda x: x

    rows, cols = len(M), len(M[0])

    new_M = []
    for row in M:
        r = []
        for x in row:
            if isinstance(x, float):
                x = ntype.from_float(x)
            else:
                x = ntype(x)
            r.append(x)
        new_M.append(r)
    M = new_M

#    M = [[ntype(x) for x in row] for row in M]
    I = [[ntype(1) if n == nc else ntype(0) for nc in range(rows)] for n in range(rows)]
    swapped = []

    for r in range(rows):
        if lead >= cols:
            return _transpose(M), _transpose(I), swapped
        i = r
        while M[i][lead] == 0:
            i += 1
            if i != rows:
                continue
            i = r
            lead += 1
            if cols == lead:
                return _transpose(M), _transpose(I), swapped

        M[i], M[r] = M[r], M[i]
        I[i], I[r] = I[r], I[i]

        swapped.append(i)
        lv = M[r][lead]
        M[r] = [mrx / lv for mrx in M[r]]
        I[r] = [mrx / lv for mrx in I[r]]

        for i in range(rows):
            if i == r:
                continue
            lv = M[i][lead]
            M[i] = [iv - lv*rv for rv, iv in zip(M[r], M[i])]
            I[i] = [iv - lv*rv for rv, iv in zip(I[r], I[i])]

        lead += 1

    return _transpose(M), _transpose(I), swapped


def pi_theorem(quantities, registry=None):
    """Builds dimensionless quantities using the Buckingham π theorem

    :param quantities: mapping between variable name and units
    :type quantities: dict
    :return: a list of dimensionless quantities expressed as dicts
    """

    # Preprocess input and build the dimensionality Matrix
    quant = []
    dimensions = set()

    if registry is None:
        getdim = lambda x: x
    else:
        getdim = registry.get_dimensionality

    for name, value in quantities.items():
        if isinstance(value, string_types):
            value = ParserHelper.from_string(value)
        if isinstance(value, dict):
            dims = getdim(UnitsContainer(value))
        elif not hasattr(value, 'dimensionality'):
            dims = getdim(value)
        else:
            dims = value.dimensionality

        if not registry and any(not key.startswith('[') for key in dims):
            logger.warning('A non dimension was found and a registry was not provided. '
                           'Assuming that it is a dimension name: {}.'.format(dims))

        quant.append((name, dims))
        dimensions = dimensions.union(dims.keys())

    dimensions = list(dimensions)

    # Calculate dimensionless  quantities
    M = [[dimensionality[dimension] for name, dimensionality in quant]
         for dimension in dimensions]

    M, identity, pivot = column_echelon_form(M, transpose_result=False)

    # Collect results
    # Make all numbers integers and minimize the number of negative exponents.
    # Remove zeros
    results = []
    for rowm, rowi in zip(M, identity):
        if any(el != 0 for el in rowm):
            continue
        max_den = max(f.denominator for f in rowi)
        neg = -1 if sum(f < 0 for f in rowi) > sum(f > 0 for f in rowi) else 1
        results.append(dict((q[0], neg * f.numerator * max_den / f.denominator)
                            for q, f in zip(quant, rowi) if f.numerator != 0))
    return results


def solve_dependencies(dependencies):
    """Solve a dependency graph.

    :param dependencies: dependency dictionary. For each key, the value is
                         an iterable indicating its dependencies.
    :return: list of sets, each containing keys of independents tasks dependent
                           only of the previous tasks in the list.
    """
    d = dict((key, set(dependencies[key])) for key in dependencies)
    r = []
    while d:
        # values not in keys (items without dep)
        t = set(i for v in d.values() for i in v) - set(d.keys())
        # and keys without value (items without dep)
        t.update(k for k, v in d.items() if not v)
        # can be done right away
        if not t:
            raise ValueError('Cyclic dependencies exist among these items: {}'.format(', '.join(repr(x) for x in d.items())))
        r.append(t)
        # and cleaned up
        d = dict(((k, v - t) for k, v in d.items() if v))
    return r


def find_shortest_path(graph, start, end, path=None):
    path = (path or []) + [start]
    if start == end:
        return path
    if not start in graph:
        return None
    shortest = None
    for node in graph[start]:
        if node not in path:
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                if not shortest or len(newpath) < len(shortest):
                    shortest = newpath
    return shortest


def find_connected_nodes(graph, start, visited=None):
    if not start in graph:
        return None

    visited = (visited or set())
    visited.add(start)

    for node in graph[start]:
        if node not in visited:
            find_connected_nodes(graph, node, visited)

    return visited


class udict(dict):
    """ Custom dict implementing __missing__.

    """
    def __missing__(self, key):
        return 0.


class UnitsContainer(Mapping):
    """The UnitsContainer stores the product of units and their respective
    exponent and implements the corresponding operations.

    UnitsContainer is a read-only mapping. All operations (even in place ones)
    return new instances.

    """
    __slots__ = ('_d', '_hash')

    def __init__(self, *args, **kwargs):
        d = udict(*args, **kwargs)
        self._d = d
        for key, value in d.items():
            if not isinstance(key, string_types):
                raise TypeError('key must be a str, not {}'.format(type(key)))
            if not isinstance(value, Number):
                raise TypeError('value must be a number, not {}'.format(type(value)))
            if not isinstance(value, float):
                d[key] = float(value)
        self._hash = hash(frozenset(self._d.items()))

    def copy(self):
        return self.__copy__()

    def add(self, key, value):
        newval = self._d[key] + value
        new = self.copy()
        if newval:
            new._d[key] = newval
        else:
            del new._d[key]

        return new

    def remove(self, keys):
        """ Create a new UnitsContainer purged from given keys.

        """
        d = udict(self._d)
        return UnitsContainer(((key, d[key]) for key in d if key not in keys))

    def rename(self, oldkey, newkey):
        """ Create a new UnitsContainer in which an entry has been renamed.

        """
        d = udict(self._d)
        d[newkey] = d.pop(oldkey)
        return UnitsContainer(d)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __hash__(self):
        return self._hash

    def __getstate__(self):
        return {'_d': self._d, '_hash': self._hash}

    def __setstate__(self, state):
        self._d = state['_d']
        self._hash = state['_hash']

    def __eq__(self, other):
        if isinstance(other, UnitsContainer):
            other = other._d
        elif isinstance(other, string_types):
            other = ParserHelper.from_string(other)
            other = other._d

        return dict.__eq__(self._d, other)

    def __str__(self):
        return self.__format__('')

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value)
                                  for key, value in sorted(self._d.items())])
        return '<UnitsContainer({})>'.format(tmp)

    def __format__(self, spec):
        return format_unit(self, spec)

    def format_babel(self, spec, **kwspec):
        return format_unit(self, spec, **kwspec)

    def __copy__(self):
        return UnitsContainer(self._d)

    def __mul__(self, other):
        d = udict(self._d)
        if not isinstance(other, self.__class__):
            err = 'Cannot multiply UnitsContainer by {}'
            raise TypeError(err.format(type(other)))
        for key, value in other.items():
            d[key] += value
        keys = [key for key, value in d.items() if value == 0]
        for key in keys:
            del d[key]

        return UnitsContainer(d)

    __rmul__ = __mul__

    def __pow__(self, other):
        if not isinstance(other, NUMERIC_TYPES):
            err = 'Cannot power UnitsContainer by {}'
            raise TypeError(err.format(type(other)))
        d = udict(self._d)
        for key, value in d.items():
            d[key] *= other
        return UnitsContainer(d)

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            err = 'Cannot divide UnitsContainer by {}'
            raise TypeError(err.format(type(other)))

        d = udict(self._d)

        for key, value in other.items():
            d[key] -= value

        keys = [key for key, value in d.items() if value == 0]
        for key in keys:
            del d[key]

        return UnitsContainer(d)

    def __rtruediv__(self, other):
        if not isinstance(other, self.__class__) and other != 1:
            err = 'Cannot divide {} by UnitsContainer'
            raise TypeError(err.format(type(other)))

        return self**-1


class ParserHelper(UnitsContainer):
    """ The ParserHelper stores in place the product of variables and
    their respective exponent and implements the corresponding operations.

    ParserHelper is a read-only mapping. All operations (even in place ones)
    return new instances.

    WARNING : The hash value used does not take into account the scale
    attribute so be careful if you use it as a dict key and then two unequal
    object can have the same hash.

    """

    __slots__ = ('scale', )

    def __init__(self, scale=1, *args, **kwargs):
        super(ParserHelper, self).__init__(*args, **kwargs)
        self.scale = scale

    @classmethod
    def from_word(cls, input_word):
        """Creates a ParserHelper object with a single variable with exponent one.

        Equivalent to: ParserHelper({'word': 1})

        """
        return cls(1, [(input_word, 1)])

    @classmethod
    def from_string(cls, input_string):
        return cls._from_string(input_string)

    @classmethod
    def eval_token(cls, token, use_decimal=False):
        token_type = token.type
        token_text = token.string
        if token_type == NUMBER:
            try:
                return int(token_text)
            except ValueError:
                if use_decimal:
                    return Decimal(token_text)
                return float(token_text)
        elif token_type == NAME:
            return ParserHelper.from_word(token_text)
        else:
            raise Exception('unknown token type')

    @classmethod
    @lru_cache()
    def _from_string(cls, input_string):
        """Parse linear expression mathematical units and return a quantity object.

        """
        if not input_string:
            return cls()

        input_string = string_preprocessor(input_string)
        if '[' in input_string:
            input_string = input_string.replace('[', '__obra__').replace(']', '__cbra__')
            reps = True
        else:
            reps = False

        gen = tokenizer(input_string)
        ret = build_eval_tree(gen).evaluate(cls.eval_token)

        if isinstance(ret, Number):
            return ParserHelper(ret)

        if not reps:
            return ret

        return ParserHelper(ret.scale,
                            dict((key.replace('__obra__', '[').replace('__cbra__', ']'), value)
                                 for key, value in ret.items()))

    def __copy__(self):
        return ParserHelper(scale=self.scale, **self)

    def copy(self):
        return self.__copy__()

    def __hash__(self):
        if self.scale != 1.0:
            mess = 'Only scale 1.0 ParserHelper instance should be considered hashable'
            raise ValueError(mess)
        return self._hash

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.scale == other.scale and\
                super(ParserHelper, self).__eq__(other)
        elif isinstance(other, string_types):
            return self == ParserHelper.from_string(other)
        elif isinstance(other, Number):
            return self.scale == other and not len(self._d)
        else:
            return self.scale == 1. and super(ParserHelper, self).__eq__(other)

    def operate(self, items, op=operator.iadd, cleanup=True):
        d = udict(self._d)
        for key, value in items:
            d[key] = op(d[key], value)

        if cleanup:
            keys = [key for key, value in d.items() if value == 0]
            for key in keys:
                del d[key]

        return self.__class__(self.scale, d)

    def __str__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value)
                                  for key, value in sorted(self._d.items())])
        return '{} {}'.format(self.scale, tmp)

    def __repr__(self):
        tmp = '{%s}' % ', '.join(["'{}': {}".format(key, value)
                                  for key, value in sorted(self._d.items())])
        return '<ParserHelper({}, {})>'.format(self.scale, tmp)

    def __mul__(self, other):
        if isinstance(other, string_types):
            new = self.add(other, 1)
        elif isinstance(other, Number):
            new = self.copy()
            new.scale *= other
        elif isinstance(other, self.__class__):
            new = self.operate(other.items())
            new.scale *= other.scale
        else:
            new = self.operate(other.items())
        return new

    __rmul__ = __mul__

    def __pow__(self, other):
        d = self._d.copy()
        for key in self._d:
            d[key] *= other
        return self.__class__(self.scale**other, d)

    def __truediv__(self, other):
        if isinstance(other, string_types):
            new = self.add(other, -1)
        elif isinstance(other, Number):
            new = self.copy()
            new.scale /= other
        elif isinstance(other, self.__class__):
            new = self.operate(other.items(), operator.sub)
            new.scale /= other.scale
        else:
            new = self.operate(other.items(), operator.sub)
        return new

    __floordiv__ = __truediv__

    def __rtruediv__(self, other):
        new = self.__pow__(-1)
        if isinstance(other, string_types):
            new = new.add(other, 1)
        elif isinstance(other, Number):
            new.scale *= other
        elif isinstance(other, self.__class__):
            new = self.operate(other.items(), operator.add)
            new.scale *= other.scale
        else:
            new = new.operate(other.items(), operator.add)
        return new


#: List of regex substitution pairs.
_subs_re = [('\N{DEGREE SIGN}', " degree"),
            (r"([\w\.\-\+\*\\\^])\s+", r"\1 "), # merge multiple spaces
            (r"({}) squared", r"\1**2"),  # Handle square and cube
            (r"({}) cubed", r"\1**3"),
            (r"cubic ({})", r"\1**3"),
            (r"square ({})", r"\1**2"),
            (r"sq ({})", r"\1**2"),
            (r"\b([0-9]+\.?[0-9]*)(?=[e|E][a-zA-Z]|[a-df-zA-DF-Z])", r"\1*"),  # Handle numberLetter for multiplication
            (r"([\w\.\-])\s+(?=\w)", r"\1*"),  # Handle space for multiplication
            ]

#: Compiles the regex and replace {} by a regex that matches an identifier.
_subs_re = [(re.compile(a.format(r"[_a-zA-Z][_a-zA-Z0-9]*")), b) for a, b in _subs_re]
_pretty_table = maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹·⁻', '0123456789*-')
_pretty_exp_re = re.compile(r"⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?:\.[⁰¹²³⁴⁵⁶⁷⁸⁹]*)?")


def string_preprocessor(input_string):

    input_string = input_string.replace(",", "")
    input_string = input_string.replace(" per ", "/")

    for a, b in _subs_re:
        input_string = a.sub(b, input_string)

    # Replace pretty format characters
    for pretty_exp in _pretty_exp_re.findall(input_string):
        exp = '**' + pretty_exp.translate(_pretty_table)
        input_string = input_string.replace(pretty_exp, exp)
    input_string = input_string.translate(_pretty_table)

    # Handle caret exponentiation
    input_string = input_string.replace("^", "**")
    return input_string


def _is_dim(name):
    return name[0] == '[' and name[-1] == ']'


class SharedRegistryObject(object):
    """Base class for object keeping a refrence to the registree.

    Such object are for now _Quantity and _Unit, in a number of places it is
    that an object from this class has a '_units' attribute.

    """

    def _check(self, other):
        """Check if the other object use a registry and if so that it is the
        same registry.

        Return True is both use a registry and they use the same, False is
        other don't use a registry and raise ValueError if other don't use the
        same unit registry.

        """
        if self._REGISTRY is getattr(other, '_REGISTRY', None):
            return True

        elif isinstance(other, SharedRegistryObject):
            mess = 'Cannot operate with {} and {} of different registries.'
            raise ValueError(mess.format(self.__class__.__name__,
                                         other.__class__.__name__))
        else:
            return False


class PrettyIPython(object):
    """Mixin to add pretty-printers for IPython"""

    def _repr_html_(self):
        if "~" in self.default_format:
            return "{:~H}".format(self)
        else:
            return "{:H}".format(self)

    def _repr_latex_(self):
        if "~" in self.default_format:
            return "${:~L}$".format(self)
        else:
            return "${:L}$".format(self)

    def _repr_pretty_(self, p, cycle):
        if "~" in self.default_format:
            p.text("{:~P}".format(self))
        else:
            p.text("{:P}".format(self))


def to_units_container(unit_like, registry=None):
    """ Convert a unit compatible type to a UnitsContainer.

    """
    mro = type(unit_like).mro()
    if UnitsContainer in mro:
        return unit_like
    elif SharedRegistryObject in mro:
        return unit_like._units
    elif string_types in mro:
        if registry:
            return registry._parse_units(unit_like)
        else:
            return ParserHelper.from_string(unit_like)
    elif dict in mro:
        return UnitsContainer(unit_like)


def infer_base_unit(q):
    """Return UnitsContainer of q with all prefixes stripped."""
    d = udict()
    parse = q._REGISTRY.parse_unit_name
    for unit_name, power in q._units.items():
        completely_parsed_unit = list(parse(unit_name))[-1]

        _, base_unit, __ = completely_parsed_unit
        d[base_unit] += power
    return UnitsContainer(dict((k, v) for k, v in d.items() if v != 0))  # remove values that resulted in a power of 0


def fix_str_conversions(cls):
    """Enable python2/3 compatible behaviour for __str__."""
    def __bytes__(self):
        return self.__unicode__().encode(locale.getpreferredencoding())
    cls.__unicode__ = __unicode__ = cls.__str__
    cls.__bytes__ = __bytes__
    if sys.version_info[0] == 2:
        cls.__str__ = __bytes__
    else:
        cls.__str__ = __unicode__
    return cls


class SourceIterator(object):
    """Iterator to facilitate reading the definition files.

    Accepts any sequence (like a list of lines, a file or another SourceIterator)

    The iterator yields the line number and line (skipping comments and empty lines)
    and stripping white spaces.

    for lineno, line in SourceIterator(sequence):
        # do something here

    """

    def __new__(cls, sequence):
        if isinstance(sequence, SourceIterator):
            return sequence

        obj = object.__new__(cls)

        if sequence is not None:
            obj.internal = enumerate(sequence, 1)
            obj.last = (None, None)

        return obj

    def __iter__(self):
        return self

    def __next__(self):
        line = ''
        while not line or line.startswith('#'):
            lineno, line = next(self.internal)
            line = line.split('#', 1)[0].strip()

        self.last = lineno, line
        return lineno, line

    next = __next__

    def block_iter(self):
        """Iterate block including header.
        """
        return BlockIterator(self)


class BlockIterator(SourceIterator):
    """Like SourceIterator but stops when it finds '@end'
    It also raises an error if another '@' directive is found inside.
    """

    def __new__(cls, line_iterator):
        obj = SourceIterator.__new__(cls, None)
        obj.internal = line_iterator.internal
        obj.last = line_iterator.last
        obj.done_last = False
        return obj

    def __next__(self):
        if not self.done_last:
            self.done_last = True
            return self.last

        lineno, line = SourceIterator.__next__(self)
        if line.startswith('@end'):
            raise StopIteration
        elif line.startswith('@'):
            raise DefinitionSyntaxError('cannot nest @ directives', lineno=lineno)

        return lineno, line

    next = __next__
