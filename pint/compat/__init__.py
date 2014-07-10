# -*- coding: utf-8 -*-
"""
    pint.compat
    ~~~~~~~~~~~

    Compatibility layer.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import sys
import tokenize

from numbers import Number
from decimal import Decimal


PYTHON3 = sys.version >= '3'

if PYTHON3:
    from io import BytesIO
    string_types = str
    tokenizer = lambda input_string: tokenize.tokenize(BytesIO(input_string.encode('utf-8')).readline)

    def u(x):
        return x

    maketrans = str.maketrans

    long_type = int
else:
    from StringIO import StringIO
    string_types = basestring
    tokenizer = lambda input_string: tokenize.generate_tokens(StringIO(input_string).readline)

    import codecs

    def u(x):
        return codecs.unicode_escape_decode(x)[0]

    maketrans = lambda f, t: dict((ord(a), b) for a, b in zip(f, t))

    long_type = long

if sys.version_info < (2, 7):
    try:
        import unittest2 as unittest
    except ImportError:
        raise Exception("Testing Pint in Python 2.6 requires package 'unittest2'")
else:
    import unittest


try:
    from collections import Chainmap
except ImportError:
    from .chainmap import ChainMap

try:
    from collections import TransformDict
except ImportError:
    from .transformdict import TransformDict

try:
    from functools import lru_cache
except ImportError:
    from .lrucache import lru_cache

try:
    from logging import NullHandler
except ImportError:
    from .nullhandler import NullHandler

try:
    import numpy as np
    from numpy import ndarray

    HAS_NUMPY = True
    NUMPY_VER = np.__version__
    NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise TypeError('Invalid magnitude for Quantity: {0!r}'.format(value))
        elif isinstance(value, string_types) and value == '':
            raise ValueError('Quantity magnitude cannot be an empty string.')
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
        if force_ndarray:
            return np.asarray(value)
        return value

except ImportError:

    np = None

    class ndarray(object):
        pass

    HAS_NUMPY = False
    NUMPY_VER = '0'
    NUMERIC_TYPES = (Number, Decimal)

    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise TypeError('Invalid magnitude for Quantity: {0!r}'.format(value))
        elif isinstance(value, string_types) and value == '':
            raise ValueError('Quantity magnitude cannot be an empty string.')
        elif isinstance(value, (list, tuple)):
            raise TypeError('lists and tuples are valid magnitudes for '
                             'Quantity only when NumPy is present.')
        return value

try:
    from uncertainties import ufloat
    HAS_UNCERTAINTIES = True
except ImportError:
    ufloat = None
    HAS_UNCERTAINTIES = False

