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

from io import BytesIO
from numbers import Number
from decimal import Decimal

from . import tokenize

ENCODING_TOKEN = tokenize.ENCODING

PYTHON3 = sys.version >= '3'

def tokenizer(input_string):
    for tokinfo in tokenize.tokenize(BytesIO(input_string.encode('utf-8')).readline):
        if tokinfo.type == ENCODING_TOKEN:
            continue
        yield tokinfo


if PYTHON3:
    string_types = str

    def u(x):
        return x

    maketrans = str.maketrans

    long_type = int
else:
    string_types = basestring

    import codecs

    def u(x):
        return codecs.unicode_escape_decode(x)[0]

    maketrans = lambda f, t: dict((ord(a), b) for a, b in zip(f, t))

    long_type = long

try:
    from collections import Chainmap
except ImportError:
    from .chainmap import ChainMap

try:
    from functools import lru_cache
except ImportError:
    from .lrucache import lru_cache

try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest



def _raise_invalid_magnitude(value):
    if isinstance(value, (dict, bool)) or value is None:
        raise TypeError('Invalid magnitude for Quantity: {0!r}'.format(value))
    elif isinstance(value, string_types) and value == '':
        raise ValueError('Quantity magnitude cannot be an empty string.')

try:
    import numpy as np
    from numpy import ndarray

    HAS_NUMPY = True
    NUMPY_VER = np.__version__
    NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    def _distill_sequence(value, units=None):
        """
        Distills sequences of quantities to magnitude arrays with a single unit.
        """
        
        _raise_invalid_magnitude(value)

        try:
            # might trigger IndexError if sequence empty
            new_units = value[0].units if units is None else units

            # might trigger TypeError if not a sequence (no __len__)
            a = np.empty(len(value))

            for i, seq_i in enumerate(value):
                # might raise AttributeError if not a quantity
                a[i] = seq_i.m_as(new_units)
                # raises DimensionalityError if incompatible units are used in the sequence

            return a, new_units

        except (TypeError, IndexError, KeyError, AttributeError):
            # if any element is not a quantity we pass value on
            return value, units


    def _to_magnitude(value, force_ndarray=False):
        _raise_invalid_magnitude(value)

        if isinstance(value, (list, tuple)):
            try:
                a = np.asarray(value)
            except ValueError:
                raise ValueError('Given magnitude value cannot be used as an numpy array! Use e.g. scalars, Quantities, lists of scalars or lists of Quantities!') from None
            return a
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

    def _distill_sequence(value, units=None, ignore_mix=True):
        _raise_invalid_magnitude(value)
        return value, units

    def _to_magnitude(value, force_ndarray=False):
        _raise_invalid_magnitude(value)

        if isinstance(value, (list, tuple)):
            raise TypeError('lists and tuples are valid magnitudes for '
                             'Quantity only when NumPy is present.')
        return value

try:
    from uncertainties import ufloat
    HAS_UNCERTAINTIES = True
except ImportError:
    ufloat = None
    HAS_UNCERTAINTIES = False

try:
    from babel import Locale as Loc
    from babel import units as babel_units
    HAS_BABEL = True
    HAS_PROPER_BABEL = hasattr(babel_units, 'format_unit')
except ImportError:
    HAS_PROPER_BABEL = HAS_BABEL = False

if not HAS_PROPER_BABEL:
    Loc = babel_units = None

try:
    import pandas as pd
    HAS_PANDAS = True
    # pin Pandas version for now
    HAS_PROPER_PANDAS = pd.__version__.startswith("0.24.0.dev0+625.gbdb7a16")
except ImportError:
    HAS_PROPER_PANDAS = HAS_PANDAS = False

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
