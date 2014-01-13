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
else:
    from StringIO import StringIO
    string_types = basestring
    tokenizer = lambda input_string: tokenize.generate_tokens(StringIO(input_string).readline)

    import codecs
    string_types = basestring

    def u(x):
        return codecs.unicode_escape_decode(x)[0]

try:
    from collections import Chainmap
except:
    from .chainmap import ChainMap

try:
    from functools import lru_cache
except:
    from .lrucache import lru_cache

try:
    import numpy as np
    from numpy import ndarray

    HAS_NUMPY = True
    NUMPY_VER = np.__version__
    NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, NUMERIC_TYPES):
            if force_ndarray:
                return np.asarray(value)
            else:
                return value
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
        else:
            raise TypeError('Invalid type of magnitude for Quantity: {}'.format(type(value)))

except ImportError:

    np = None

    class ndarray(object):
        pass

    HAS_NUMPY = False
    NUMPY_VER = 0
    NUMERIC_TYPES = (Number, Decimal)

    def _to_magnitude(value, force_ndarray=False):
        if isinstance(value, NUMERIC_TYPES):
            return value
        elif isinstance(value, (list, tuple)):
            raise TypeError('lists and tuples are valid magnitudes for '
                            'Quantity only when NumPy is present.')
        else:
            raise TypeError('Invalid type of magnitude for Quantity: {}'.format(type(value)))
