"""
    pint.compat
    ~~~~~~~~~~~

    Compatibility layer.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
import tokenize
from decimal import Decimal
from importlib import import_module
from io import BytesIO
from numbers import Number
from typing import Mapping, Optional


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


try:
    import numpy as np
    from numpy import datetime64 as np_datetime64
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

try:
    import mip

    mip_model = mip.model
    mip_Model = mip.Model
    mip_INF = mip.INF
    mip_INTEGER = mip.INTEGER
    mip_xsum = mip.xsum
    mip_OptimizationStatus = mip.OptimizationStatus

    HAS_MIP = True
except ImportError:
    HAS_MIP = False

# Defines Logarithm and Exponential for Logarithmic Converter
if HAS_NUMPY:
    from numpy import exp  # noqa: F401
    from numpy import log  # noqa: F401
else:
    from math import exp  # noqa: F401
    from math import log  # noqa: F401

if not HAS_BABEL:
    babel_parse = babel_units = missing_dependency("Babel")  # noqa: F811

if not HAS_MIP:
    mip_missing = missing_dependency("mip")
    mip_model = mip_missing
    mip_Model = mip_missing
    mip_INF = mip_missing
    mip_INTEGER = mip_missing
    mip_xsum = mip_missing
    mip_OptimizationStatus = mip_missing

# Define location of pint.Quantity in NEP-13 type cast hierarchy by defining upcast
# types using guarded imports

try:
    from dask import array as dask_array
    from dask.base import compute, persist, visualize
except ImportError:
    compute, persist, visualize = None, None, None
    dask_array = None


upcast_type_names = (
    "pint_pandas.PintArray",
    "pandas.Series",
    "xarray.core.dataarray.DataArray",
    "xarray.core.dataset.Dataset",
    "xarray.core.variable.Variable",
    "pandas.core.series.Series",
    "xarray.core.dataarray.DataArray",
)

upcast_type_map: Mapping[str : Optional[type]] = {k: None for k in upcast_type_names}


def fully_qualified_name(t: type) -> str:
    module = t.__module__
    name = t.__qualname__

    if module is None or module == "builtins":
        return name

    return f"{module}.{name}"


def check_upcast_type(obj: type) -> bool:
    fqn = fully_qualified_name(obj)
    if fqn not in upcast_type_map:
        return False
    else:
        module_name, class_name = fqn.rsplit(".", 1)
        cls = getattr(import_module(module_name), class_name)

    upcast_type_map[fqn] = cls
    # This is to check we are importing the same thing.
    # and avoid weird problems. Maybe instead of return
    # we should raise an error if false.
    return obj in upcast_type_map.values()


def is_upcast_type(other: type) -> bool:
    if other in upcast_type_map.values():
        return True
    return check_upcast_type(other)


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
