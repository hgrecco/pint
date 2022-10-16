"""
    pint.compat
    ~~~~~~~~~~~

    Compatibility layer.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
import token as tokenlib
import tokenize
from decimal import Decimal
from io import BytesIO
from numbers import Number

try:
    from uncertainties import UFloat, ufloat
    from uncertainties import unumpy as unp

    HAS_UNCERTAINTIES = True
except ImportError:
    UFloat = ufloat = unp = None
    HAS_UNCERTAINTIES = False


def missing_dependency(package, display_name=None):
    display_name = display_name or package

    def _inner(*args, **kwargs):
        raise Exception(
            "This feature requires %s. Please install it by running:\n"
            "pip install %s" % (display_name, package)
        )

    return _inner


# https://stackoverflow.com/a/1517965/1291237
class tokens_with_lookahead:
    def __init__(self, iter):
        self.iter = iter
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop(0)
        else:
            return self.iter.__next__()

    def lookahead(self, n):
        """Return an item n entries ahead in the iteration."""
        while n >= len(self.buffer):
            try:
                self.buffer.append(self.iter.__next__())
            except StopIteration:
                return None
        return self.buffer[n]


def tokenizer(input_string):
    def _number_or_nan(token):
        if token.type == tokenlib.NUMBER or (
            token.type == tokenlib.NAME and token.string == "nan"
        ):
            return True
        return False

    gen = tokenize.tokenize(BytesIO(input_string.encode("utf-8")).readline)
    toklist = tokens_with_lookahead(gen)
    for tokinfo in toklist:
        if tokinfo.type != tokenize.ENCODING:
            if (
                tokinfo.string == "+"
                and toklist.lookahead(0).string == "/"
                and toklist.lookahead(1).string == "-"
            ):
                line = tokinfo.line
                start = tokinfo.start
                for i in range(-1, 1):
                    next(toklist)
                end = tokinfo.end
                tokinfo = tokenize.TokenInfo(
                    type=tokenlib.OP,
                    string="+/-",
                    start=start,
                    end=end,
                    line=line,
                )
                yield tokinfo
            elif (
                tokinfo.string == "("
                and _number_or_nan(toklist.lookahead(0))
                and toklist.lookahead(1).string == "+"
                and toklist.lookahead(2).string == "/"
                and toklist.lookahead(3).string == "-"
                and _number_or_nan(toklist.lookahead(4))
                and toklist.lookahead(5).string == ")"
            ):
                # ( NUM_OR_NAN +/- NUM_OR_NAN )
                start = tokinfo.start
                end = toklist.lookahead(5).end
                line = tokinfo.line[start[1] : end[1]]
                nominal_value = toklist.lookahead(0)
                std_dev = toklist.lookahead(4)
                plus_minus_op = tokenize.TokenInfo(
                    type=tokenlib.OP,
                    string="+/-",
                    start=toklist.lookahead(1).start,
                    end=toklist.lookahead(3).end,
                    line=line,
                )
                # Strip parentheses and let tight binding of +/- do its work
                for i in range(-1, 5):
                    next(toklist)
                yield nominal_value
                yield plus_minus_op
                yield std_dev
            elif (
                tokinfo.type == tokenlib.NUMBER
                and toklist.lookahead(0).string == "("
                and toklist.lookahead(1).type == tokenlib.NUMBER
                and toklist.lookahead(2).string == ")"
            ):
                line = tokinfo.line
                start = tokinfo.start
                nominal_value = tokinfo
                std_dev = toklist.lookahead(1)
                plus_minus_op = tokenize.TokenInfo(
                    type=tokenlib.OP,
                    string="+/-",
                    start=toklist.lookahead(0).start,
                    end=toklist.lookahead(2).end,
                    line=line,
                )
                for i in range(-1, 2):
                    next(toklist)
                yield nominal_value
                yield plus_minus_op
                if "." not in std_dev.string:
                    std_dev = tokenize.TokenInfo(
                        type=std_dev.type,
                        string="0." + std_dev.string,
                        start=std_dev.start,
                        end=std_dev.end,
                        line=line,
                    )
                yield std_dev
            else:
                yield tokinfo


# TODO: remove this warning after v0.10
class BehaviorChangeWarning(UserWarning):
    pass


try:
    import numpy as np
    from numpy import datetime64 as np_datetime64
    from numpy import ndarray

    HAS_NUMPY = True
    NUMPY_VER = np.__version__
    if HAS_UNCERTAINTIES:
        NUMERIC_TYPES = (Number, Decimal, ndarray, np.number, UFloat)
    else:
        NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    def _to_magnitude(value, force_ndarray=False, force_ndarray_like=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise TypeError("Invalid magnitude for Quantity: {0!r}".format(value))
        elif isinstance(value, str) and value == "":
            raise ValueError("Quantity magnitude cannot be an empty string.")
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
        elif HAS_UNCERTAINTIES:
            from pint.facets.measurement.objects import Measurement

            if isinstance(value, Measurement):
                return ufloat(value.value, value.error)
        if force_ndarray or (
            force_ndarray_like and not is_duck_array_type(type(value))
        ):
            return np.asarray(value)
        return value

    def _test_array_function_protocol():
        # Test if the __array_function__ protocol is enabled
        try:

            class FakeArray:
                def __array_function__(self, *args, **kwargs):
                    return

            np.concatenate([FakeArray()])
            return True
        except ValueError:
            return False

    HAS_NUMPY_ARRAY_FUNCTION = _test_array_function_protocol()

    NP_NO_VALUE = np._NoValue

except ImportError:

    np = None

    class ndarray:
        pass

    class np_datetime64:
        pass

    HAS_NUMPY = False
    NUMPY_VER = "0"
    NUMERIC_TYPES = (Number, Decimal)
    HAS_NUMPY_ARRAY_FUNCTION = False
    NP_NO_VALUE = None

    def _to_magnitude(value, force_ndarray=False, force_ndarray_like=False):
        if force_ndarray or force_ndarray_like:
            raise ValueError(
                "Cannot force to ndarray or ndarray-like when NumPy is not present."
            )
        elif isinstance(value, (dict, bool)) or value is None:
            raise TypeError("Invalid magnitude for Quantity: {0!r}".format(value))
        elif isinstance(value, str) and value == "":
            raise ValueError("Quantity magnitude cannot be an empty string.")
        elif isinstance(value, (list, tuple)):
            raise TypeError(
                "lists and tuples are valid magnitudes for "
                "Quantity only when NumPy is present."
            )
        elif HAS_UNCERTAINTIES:
            from pint.facets.measurement.objects import Measurement

            if isinstance(value, Measurement):
                return ufloat(value.value, value.error)
        return value


try:
    from babel import Locale as Loc
    from babel import units as babel_units

    babel_parse = Loc.parse

    HAS_BABEL = hasattr(babel_units, "format_unit")
except ImportError:
    HAS_BABEL = False

# Defines Logarithm and Exponential for Logarithmic Converter
if HAS_NUMPY:
    from numpy import exp  # noqa: F401
    from numpy import log  # noqa: F401
else:
    from math import exp  # noqa: F401
    from math import log  # noqa: F401

if not HAS_BABEL:
    babel_parse = babel_units = missing_dependency("Babel")  # noqa: F811

# Define location of pint.Quantity in NEP-13 type cast hierarchy by defining upcast
# types using guarded imports
upcast_types = []

# pint-pandas (PintArray)
try:
    from pint_pandas import PintArray

    upcast_types.append(PintArray)
except ImportError:
    pass

# Pandas (Series)
try:
    from pandas import Series

    upcast_types.append(Series)
except ImportError:
    pass

# xarray (DataArray, Dataset, Variable)
try:
    from xarray import DataArray, Dataset, Variable

    upcast_types += [DataArray, Dataset, Variable]
except ImportError:
    pass

try:
    from dask import array as dask_array
    from dask.base import compute, persist, visualize

except ImportError:
    compute, persist, visualize = None, None, None
    dask_array = None


def is_upcast_type(other) -> bool:
    """Check if the type object is a upcast type using preset list.

    Parameters
    ----------
    other : object

    Returns
    -------
    bool
    """
    return other in upcast_types


def is_duck_array_type(cls) -> bool:
    """Check if the type object represents a (non-Quantity) duck array type.

    Parameters
    ----------
    cls : class

    Returns
    -------
    bool
    """
    # TODO (NEP 30): replace duck array check with hasattr(other, "__duckarray__")
    return issubclass(cls, ndarray) or (
        not hasattr(cls, "_magnitude")
        and not hasattr(cls, "_units")
        and HAS_NUMPY_ARRAY_FUNCTION
        and hasattr(cls, "__array_function__")
        and hasattr(cls, "ndim")
        and hasattr(cls, "dtype")
    )


def is_duck_array(obj):
    return is_duck_array_type(type(obj))


def eq(lhs, rhs, check_all: bool):
    """Comparison of scalars and arrays.

    Parameters
    ----------
    lhs : object
        left-hand side
    rhs : object
        right-hand side
    check_all : bool
        if True, reduce sequence to single bool;
        return True if all the elements are equal.

    Returns
    -------
    bool or array_like of bool
    """
    out = lhs == rhs
    if check_all and is_duck_array_type(type(out)):
        return out.all()
    return out


def isnan(obj, check_all: bool):
    """Test for NaN or NaT

    Parameters
    ----------
    obj : object
        scalar or vector
    check_all : bool
        if True, reduce sequence to single bool;
        return True if any of the elements are NaN.

    Returns
    -------
    bool or array_like of bool.
    Always return False for non-numeric types.
    """
    if is_duck_array_type(type(obj)):
        if obj.dtype.kind in "if":
            out = np.isnan(obj)
        elif obj.dtype.kind in "Mm":
            out = np.isnat(obj)
        else:
            # Not a numeric or datetime type
            out = np.full(obj.shape, False)
        return out.any() if check_all else out
    if isinstance(obj, np_datetime64):
        return np.isnat(obj)
    try:
        return math.isnan(obj)
    except TypeError:
        if HAS_UNCERTAINTIES:
            return unp.isnan(obj)
        return False


def zero_or_nan(obj, check_all: bool):
    """Test if obj is zero, NaN, or NaT

    Parameters
    ----------
    obj : object
        scalar or vector
    check_all : bool
        if True, reduce sequence to single bool;
        return True if all the elements are zero, NaN, or NaT.

    Returns
    -------
    bool or array_like of bool.
    Always return False for non-numeric types.
    """
    out = eq(obj, 0, False) + isnan(obj, False)
    if check_all and is_duck_array_type(type(out)):
        return out.all()
    return out
