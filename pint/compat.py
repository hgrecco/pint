"""
    pint.compat
    ~~~~~~~~~~~

    Compatibility layer.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import os
import tokenize
from decimal import Decimal
from io import BytesIO
from numbers import Number


def missing_dependency(package, display_name=None):
    display_name = display_name or package

    def _inner(*args, **kwargs):
        raise Exception(
            "This feature requires %s. Please install it by running:\n"
            "pip install %s" % (display_name, package)
        )

    return _inner


def tokenizer(input_string):
    for tokinfo in tokenize.tokenize(BytesIO(input_string.encode("utf-8")).readline):
        if tokinfo.type != tokenize.ENCODING:
            yield tokinfo


# TODO: remove this warning after v0.10
class BehaviorChangeWarning(UserWarning):
    pass


array_function_change_msg = """The way Pint handles NumPy operations has changed with the
implementation of NEP 18. Unimplemented NumPy operations will now fail instead of making
assumptions about units. Some functions, eg concat, will now return Quanties with units, where
they returned ndarrays previously. See https://github.com/hgrecco/pint/pull/905.

To hide this warning, wrap your first creation of an array Quantity with
warnings.catch_warnings(), like the following:

import numpy as np
import warnings
from pint import Quantity

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])

To disable the new behavior, see
https://www.numpy.org/neps/nep-0018-array-function-protocol.html#implementation
"""


try:
    import numpy as np
    from numpy import ndarray

    HAS_NUMPY = True
    NUMPY_VER = np.__version__
    NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    def _to_magnitude(value, force_ndarray=False, force_ndarray_like=False):
        if isinstance(value, (dict, bool)) or value is None:
            raise TypeError("Invalid magnitude for Quantity: {0!r}".format(value))
        elif isinstance(value, str) and value == "":
            raise ValueError("Quantity magnitude cannot be an empty string.")
        elif isinstance(value, (list, tuple)):
            return np.asarray(value)
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
    SKIP_ARRAY_FUNCTION_CHANGE_WARNING = not HAS_NUMPY_ARRAY_FUNCTION

    NP_NO_VALUE = np._NoValue

    ARRAY_FALLBACK = bool(int(os.environ.get("PINT_ARRAY_PROTOCOL_FALLBACK", 1)))

except ImportError:

    np = None

    class ndarray:
        pass

    HAS_NUMPY = False
    NUMPY_VER = "0"
    NUMERIC_TYPES = (Number, Decimal)
    HAS_NUMPY_ARRAY_FUNCTION = False
    SKIP_ARRAY_FUNCTION_CHANGE_WARNING = True
    NP_NO_VALUE = None
    ARRAY_FALLBACK = False

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

    babel_parse = Loc.parse

    HAS_BABEL = hasattr(babel_units, "format_unit")
except ImportError:
    HAS_BABEL = False

if not HAS_BABEL:
    babel_parse = babel_units = missing_dependency("Babel")  # noqa: F811

# Define location of pint.Quantity in NEP-13 type cast hierarchy by defining upcast
# types using guarded imports
upcast_types = []

# pint-pandas (PintArray)
try:
    from pintpandas import PintArray

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


def is_upcast_type(other):
    """Check if the type object is a upcast type using preset list.

    Parameters
    ----------
    other : object

    Returns
    -------
    bool
    """
    return other in upcast_types


def is_duck_array_type(other):
    """Check if the type object represents a (non-Quantity) duck array type.

    Parameters
    ----------
    other : object

    Returns
    -------
    bool
    """
    # TODO (NEP 30): replace duck array check with hasattr(other, "__duckarray__")
    return other is ndarray or (
        not hasattr(other, "_magnitude")
        and not hasattr(other, "_units")
        and HAS_NUMPY_ARRAY_FUNCTION
        and hasattr(other, "__array_function__")
        and hasattr(other, "ndim")
        and hasattr(other, "dtype")
    )


def eq(lhs, rhs, check_all):
    """Comparison of scalars and arrays.

    Parameters
    ----------
    lhs : object
        left-hand side
    rhs : object
        right-hand side
    check_all : bool
        if True, reduce sequence to single bool.

    Returns
    -------
    bool or array_like of bool
    """
    out = lhs == rhs
    if check_all and isinstance(out, ndarray):
        return np.all(out)
    return out
