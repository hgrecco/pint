# -*- coding: utf-8 -*-
"""
    pint.util
    ~~~~~~~~~

    Miscellaneous functions for pint.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re
import sys
import tokenize
import operator
from numbers import Number
from fractions import Fraction

import logging
from token import STRING, NAME, OP
from tokenize import untokenize

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if sys.version < '3':
    from StringIO import StringIO
    string_types = basestring
    ptok = lambda input_string: tokenize.generate_tokens(StringIO(input_string).readline)
else:
    from io import BytesIO
    string_types = str
    ptok = lambda input_string: tokenize.tokenize(BytesIO(input_string.encode('utf-8')).readline)


def matrix_to_string(matrix, row_headers=None, col_headers=None, fmtfun=lambda x: str(int(x))):
    """Takes a 2D matrix (as nested list) and returns a string.
    """
    ret = []
    if col_headers:
        ret.append('\t' if row_headers else '' + '\t'.join(col_headers))
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

    M = [[ntype(x) for x in row] for row in M]
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

try:
    import numpy as np
    from numpy import ndarray

    HAS_NUMPY = True
    NUMERIC_TYPES = (Number, ndarray)

except ImportError:

    class ndarray(object):
        pass

    HAS_NUMPY = False
    NUMERIC_TYPES = (Number, )


def _join(fmt, iterable):
    if not iter:
        return ''
    if not '{}' in fmt:
        return fmt.join(iterable)
    miter = iter(iterable)
    first = next(miter)
    for val in miter:
        ret = fmt.format(first, val)
        first = ret
    return first


def formatter(items, as_ratio=True, single_denominator=False,
              product_fmt=' * ', division_fmt=' / ', power_fmt='{} ** {}',
              parentheses_fmt='({})', exp_call=lambda x: '{:n}'.format(x)):
    """Format a list of (name, exponent) pairs.

    :param items: a list of (name, exponent) pairs.
    :param as_ratio: True to display as ratio, False as negative powers.
    :param single_denominator: all with terms with negative exponents are
                               collected together.
    :param product_fmt: the format used for multiplication.
    :param division_fmt: the format used for division.
    :param power_fmt: the format used for exponentiation.
    :param parentheses_fmt: the format used for parenthesis.

    :return: the formula as a string.
    """
    if as_ratio:
        fun = lambda x: exp_call(abs(x))
    else:
        fun = exp_call

    pos_terms, neg_terms = [], []

    for key, value in sorted(items):
        if value == 1:
            pos_terms.append(key)
        elif value > 0:
            pos_terms.append(power_fmt.format(key, fun(value)))
        elif value == -1:
            neg_terms.append(key)
        else:
            neg_terms.append(power_fmt.format(key, fun(value)))

    if pos_terms:
        pos_ret = _join(product_fmt, pos_terms)
    elif as_ratio and neg_terms:
        pos_ret = '1'
    else:
        pos_ret = ''

    if not neg_terms:
        return pos_ret

    if as_ratio:
        if single_denominator:
            neg_ret = _join(product_fmt, neg_terms)
            if len(neg_terms) > 1:
                neg_ret = parentheses_fmt.format(neg_ret)
        else:
            neg_ret = _join(division_fmt, neg_terms)
    else:
        neg_ret = product_fmt.join(neg_terms)

    return _join(division_fmt, [pos_ret, neg_ret])


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
            dims = getdim(value)
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
        results.append({q[0]: neg * f.numerator * max_den / f.denominator
                        for q, f in zip(quant, rowi) if f.numerator != 0})

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
        r.append(t)
        # and cleaned up
        d = dict(((k, v - t) for k, v in d.items() if v))
    return r


def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
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


class ParserHelper(dict):
    """The ParserHelper stores in place the product of variables and
    their respective exponent and implements the corresponding operations.
    """

    __slots__ = ('scale', )

    def __init__(self, scale=1, *args, **kwargs):
        self.scale = scale
        dict.__init__(self, *args, **kwargs)

    @classmethod
    def from_word(cls, input_word):
        """Creates a ParserHelper object with a single variable with exponent one.

        Equivalent to: ParserHelper({'word': 1})

        """
        ret = cls()
        ret.add(input_word, 1)
        return ret

    @classmethod
    def from_string(cls, input_string):
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

        gen = ptok(input_string)
        result = []
        for toknum, tokval, _, _, _ in gen:
            if toknum == NAME:
                if not tokval:
                    continue
                result.extend([
                    (NAME, 'L_'),
                    (OP, '('),
                    (STRING, '"' + tokval + '"'),
                    (OP, ')')
                ])
            else:
                result.append((toknum, tokval))

        ret = eval(untokenize(result),
                   {'__builtins__': None},
                   {'L_': cls.from_word})
        if isinstance(ret, Number):
            return ParserHelper(ret)

        if not reps:
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


#: List of regex substitution pairs.
_subs_re = [(r"({0}) squared", r"\1**2"),  # Handle square and cube
            (r"({0}) cubed", r"\1**3"),
            (r"cubic ({0})", r"\1**3"),
            (r"square ({0})", r"\1**2"),
            (r"sq ({0})", r"\1**2"),
            (r"(\w)\s+(?=\w)", r"\1*"),  # Handle space for multiplication
            (r"([0-9])(?={0})(?!(?:[e|E][-+]?[0-9]+))", r"\1*")
            ]

#: Compiles the regex and replace {0} by a regex that matches an identifier.
_subs_re = [(re.compile(a.format(r"[_a-zA-Z][_a-zA-Z0-9]*")), b) for a, b in _subs_re]


def string_preprocessor(input_string):

    input_string = input_string.replace(",", "")
    input_string = input_string.replace(" per ", "/")

    for a, b in _subs_re:
        input_string = a.sub(b, input_string)

    # Handle caret exponentiation
    input_string = input_string.replace("^", "**")
    return input_string
