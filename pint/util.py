# -*- coding: utf-8 -*-
"""
    pint.util
    ~~~~~~~~~

    Miscellaneous functions for pint.

    :copyright: 2012 by Hernan E. Grecco.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import sys
from numbers import Number
from collections import Iterable

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if sys.version < '3':
    string_types = basestring
else:
    string_types = str

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
    if registry is None:
        from . import _DEFAULT_REGISTRY
        registry = _DEFAULT_REGISTRY

    # Preprocess input
    quant = []
    dimensions = set()
    for name, value in quantities.items():
        if isinstance(value, dict):
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
