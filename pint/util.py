# -*- coding: utf-8 -*-
"""
    pint.util
    ~~~~~~~~~

    Miscellaneous functions for pint.

    :copyright: 2012 by Hernan E. Grecco.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re
import sys
import tokenize
import operator
from numbers import Number
from collections import Iterable

import logging
from token import STRING, NAME, OP, NUMBER
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

    def nullspace(A, atol=1e-13, rtol=0):
        raise Exception('NumPy is required for this operation.')

    class ndarray(object):
        pass

    HAS_NUMPY = False
    NUMERIC_TYPES = (Number, )

    def _to_magnitude(value, force_ndarray=False):
        return value


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


def formatter(items, product_symbol=' * ', power_format=' ** {:n}',
              as_ratio=True, single_denominator=False, parentheses=('(', ')')):
    """Format a list of (name, exponent) pairs.

    :param items: a list of (name, exponent) pairs.
    :param product_symbol: the symbol used for multiplication.
    :param power_format: the symbol used for exponentiation including a,
                         formatting place holder for the power.
    :param as_ratio: True to display as ratio, False as negative powers.
    :param single_denominator: all with terms with negative exponents are
                               collected together.
    :param parentheses: tuple with the symbols to open and close parentheses.

    :return: the formula as a string.
    """
    if as_ratio:
        fun = abs
    else:
        fun = lambda x: x

    pos_terms, neg_terms = [], []

    for key, value in sorted(items):
        if value == 1:
            pos_terms.append(key)
        elif value > 1:
            pos_terms.append(key + power_format.format(value))
        elif value == -1:
            neg_terms.append(key)
        else:
            neg_terms.append(key + power_format.format(fun(value)))

    if pos_terms:
        ret = product_symbol.join(pos_terms)
    elif as_ratio and neg_terms:
        ret = '1'
    else:
        ret = ''

    if neg_terms:
        if as_ratio:
            ret += ' / '
            if single_denominator:
                if len(neg_terms) > 1:
                    ret += parentheses[0]
                ret += product_symbol.join(neg_terms)
                if len(neg_terms) > 1:
                    ret += parentheses[1]
            else:
                ret += ' / '.join(neg_terms)
        else:
            ret += product_symbol.join(neg_terms)

    return ret


def pi_theorem(quantities, registry=None):
    """Builds dimensionless quantities using the Buckingham π theorem

    :param quantities: mapping between variable name and units
    :type quantities: dict
    :return: a list of dimensionless quantities expressed as dicts
    """

    # Preprocess input
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

        gen = ptok(input_string)
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


def string_preprocessor(input_string):

    input_string = input_string.replace(",", "")
    input_string = input_string.replace(" per ", "/")

    # Handle square and cube
    input_string = re.sub(r"([a-zA-Z]+) squared", r"\1**2", input_string)
    input_string = re.sub(r"([a-zA-Z]+) cubed", r"\1**3", input_string)
    input_string = re.sub(r"cubic ([a-zA-Z]+)", r"\1**3", input_string)
    input_string = re.sub(r"square ([a-zA-Z]+)", r"\1**2", input_string)
    input_string = re.sub(r"sq ([a-zA-Z]+)", r"\1**2", input_string)

    # Handle space for multiplication
    input_string = re.sub(r"([a-zA-Z0-9])\s+(?=[a-zA-Z0-9])", r"\1*", input_string)

    # Handle caret exponentiation
    input_string = input_string.replace("^", "**")
    return input_string
