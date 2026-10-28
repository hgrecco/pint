"""
pint.compat
~~~~~~~~~~~

Compatibility layer.

Array detection/support functions try the APIs in the following order:
- Array API standard (experimental)
- Duck arrays (e.g. dask/pandas; partial support)
- NumPy ndarray (full support)

:copyright: 2013 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from decimal import Decimal
from fractions import Fraction
from importlib import import_module
from importlib.util import find_spec
from numbers import Number
from typing import (
    Any,
    Literal,
    # Remove once all dependent packages change their imports.
    NoReturn,
    TypeAlias,  # noqa
    Unpack,  # noqa
    overload,
)

from typing_extensions import TypeIs, deprecated  # noqa

from ._array_typing import ArrayAPIModule
from ._typing import Array


def coerce_scalar(value, scalar):
    """Coerce a scalar (a conversion scale or offset) to be arithmetically
    compatible with value (magnitude).

    Decimal refuses arithmetic with float (both directions raise TypeError);
    Fraction loses its type when combined with float. Promote the scalar to
    match a Decimal/Fraction magnitude, and demote a Decimal scalar to float
    when value is a plain float (the only case where float OP Decimal fails).
    int OP Decimal works fine and is left unchanged.
    """
    if isinstance(value, Decimal):
        if not isinstance(scalar, Decimal):
            return Decimal(str(scalar))
    elif isinstance(value, Fraction):
        if not isinstance(scalar, Fraction):
            return Fraction(str(scalar))
    elif isinstance(value, float) and isinstance(scalar, Decimal):
        return float(scalar)
    return scalar


def missing_dependency(
    package: str, display_name: str | None = None
) -> Callable[..., NoReturn]:
    """Return a helper function that raises an exception when used.

    It provides a way delay a missing dependency exception until it is used.
    """
    display_name = display_name or package

    def _inner(*args: Any, **kwargs: Any) -> NoReturn:
        raise Exception(
            "This feature requires %s. Please install it by running:\n"
            "pip install %s" % (display_name, package)
        )

    return _inner


def fully_qualified_name(t: type) -> str:
    """Return the fully qualified name of a type."""
    module = t.__module__
    name = t.__qualname__

    if module is None or module == "builtins":
        return name

    return f"{module}.{name}"


def is_upcast_type(cls: type) -> bool:
    """Check if the type object is an upcast type."""
    # easy case: the class has already been cached
    if cls in upcast_type_map.values():
        return True

    # easy case: the class is not intended to be an upcast type
    fqn = fully_qualified_name(cls)
    if fqn not in upcast_type_map:
        return False

    # before caching this class, we want to make sure it's actually
    #   the exact type it claims to be
    module_name, class_name = fqn.rsplit(".", 1)
    try:
        real_cls = getattr(import_module(module_name), class_name)
    except ModuleNotFoundError:
        # this might happen if the user does not have the module installed,
        #   in which case the class has to be different from what its
        #   __qualname__ says
        return False
    else:
        # cache the actual class anyway
        upcast_type_map[fqn] = real_cls
        return cls is real_cls or cls == real_cls


#######
# Array support
#######


def is_duck_array(obj: object) -> TypeIs[Array]:
    """Check if an object represents a (non-Quantity) duck array."""
    # TODO (NEP 30): replace duck array check with hasattr(other, "__duckarray__")
    return isinstance(obj, ndarray) or (
        not hasattr(obj, "_magnitude")
        and not hasattr(obj, "_units")
        and hasattr(obj, "__array_function__")
        and hasattr(obj, "ndim")
        and hasattr(obj, "dtype")
    )


def is_array_or_duck(obj: object) -> TypeIs[Array]:
    # TODO (NEP 30): replace duck array check with hasattr(other, "__duckarray__")
    return hasattr(obj, "__array_namespace__") or (
        not hasattr(obj, "_magnitude")
        and not hasattr(obj, "_units")
        and hasattr(obj, "__array_function__")
        and hasattr(obj, "ndim")
        and hasattr(obj, "dtype")
    )


def array_namespace(array: Array, /) -> ArrayAPIModule:
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()  # pyright: ignore[reportReturnType]
    # backwards-compatible duck array support
    elif HAS_NUMPY:
        return numpy
    else:
        raise TypeError("Pint only fully supports duck arrays if NumPy is available.")


def array_size(array: Array, /) -> int:
    if hasattr(array, "size"):
        return array.size
    return array_namespace(array).size(array)  # pyright: ignore[reportAttributeAccessIssue]


@overload
def eq(lhs: Any, rhs: Any, check_all: Literal[True]) -> "bool | np.bool": ...
@overload
def eq(lhs: Any, rhs: Any, check_all: bool) -> "bool | np.bool | Iterable[bool]": ...
def eq(lhs: Any, rhs: Any, check_all: bool) -> "bool | np.bool | Iterable[bool]":
    """Comparison of scalars and arrays.

    Parameters
    ----------
    lhs
        left-hand side
    rhs
        right-hand side
    check_all
        if True, reduce sequence to single bool;
        return True if all the elements are equal.

    Returns
    -------
    bool or array_like of bool
    """
    out = lhs == rhs
    if check_all and is_array_or_duck(out):
        return array_namespace(out).all(out).item()
    return out


@overload
def isnan(obj: Any, /, check_all: Literal[True]) -> "bool | np.bool": ...
@overload
def isnan(obj: Any, /, check_all: bool) -> "bool | np.bool | Iterable[bool]": ...
def isnan(obj: Any, /, check_all: bool) -> "bool | np.bool | Iterable[bool]":
    """Test for NaN or NaT.

    Parameters
    ----------
    obj
        scalar or vector
    check_all
        if True, reduce sequence to single bool;
        return True if any of the elements are NaN.

    Returns
    -------
    bool or array_like of bool.
        Always return False for non-numeric types.
    """
    # arrays
    if is_duck_array(obj):
        xp = array_namespace(obj)
        if xp.isdtype(obj.dtype, "numeric"):
            out = xp.isnan(obj)
        elif getattr(obj.dtype, "kind", "?") in "Mm":  # numpy datetime or timedelta
            out = xp.isnat(obj)
        else:
            if HAS_UNCERTAINTIES:
                try:
                    out = unp.isnan(obj)
                except TypeError:
                    # Not a numeric or UFloat type
                    out = np.full(obj.shape, False)
            else:
                # Not a numeric or datetime type
                out = np.full(obj.shape, False)
        return xp.any(out).item() if check_all else out
    # scalars
    if isinstance(obj, np_datetime64):
        return np.isnat(obj)
    elif HAS_UNCERTAINTIES and isinstance(obj, UFloat):
        return unp.isnan(obj)
    try:
        return math.isnan(obj)
    except TypeError:
        return False


def zero_or_nan(obj: Any, /, check_all: bool) -> bool | Iterable[bool]:
    """Test if obj is zero, NaN, or NaT.

    Parameters
    ----------
    obj
        scalar or vector
    check_all
        if True, reduce sequence to single bool;
        return True if all the elements are zero, NaN, or NaT.

    Returns
    -------
    bool or array_like of bool.
        Always return False for non-numeric types.
    """
    out = (obj == 0) | isnan(obj, check_all=False)
    if check_all and is_array_or_duck(out):
        return array_namespace(out).all(out).item()
    return out


# TODO: remove this warning after v0.10
class BehaviorChangeWarning(UserWarning):
    pass


##############
# try imports
##############

try:
    import babel  # noqa: F401
    from babel import units as babel_units

    HAS_BABEL = hasattr(babel_units, "format_unit")
except ImportError:
    HAS_BABEL = False

try:
    import uncertainties  # noqa: F401

    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

HAS_DASK = find_spec("dask") is not None


##############################
# Imports are handled here
# in order to be able to have
# them as constants
# in mypy configuration.
##############################

if HAS_BABEL:
    from babel import Locale
    from babel import units as babel_units

    babel_parse = Locale.parse
else:
    babel_parse = missing_dependency("Babel")  # noqa: F811 # type:ignore
    babel_units = babel_parse
    Locale = missing_dependency

if HAS_UNCERTAINTIES:
    from uncertainties import UFloat, ufloat

    unp = None
else:
    UFloat = ufloat = unp = None


if HAS_NUMPY:
    import numpy as np
    from numpy import datetime64 as np_datetime64
    from numpy import (
        exp,  # noqa: F401
        log,  # noqa: F401
        ndarray,
    )
    from numpy import timedelta64 as np_timedelta64

    NUMPY_VER = np.__version__
    if HAS_UNCERTAINTIES:
        from uncertainties import unumpy as unp

        NUMERIC_TYPES = (Number, Decimal, ndarray, np.number, UFloat)
    else:
        NUMERIC_TYPES = (Number, Decimal, ndarray, np.number)

    NP_NO_VALUE = np._NoValue

else:
    np = None

    class ndarray:
        pass

    class np_datetime64:
        pass

    class np_timedelta64:
        pass

    from math import (
        exp,  # noqa: F401
        log,  # noqa: F401
    )

    NUMPY_VER = "0"
    NUMERIC_TYPES = (Number, Decimal)
    NP_NO_VALUE = None


if HAS_SCIPY:
    import scipy
else:
    scipy = missing_dependency("scipy")


# Define location of pint.Quantity in NEP-13 type cast hierarchy by defining upcast
# types using guarded imports

if HAS_DASK:
    from dask.base import compute, persist, visualize

    class _LazyDaskArray:
        def __getattr__(self, attr):
            return getattr(import_module("dask.array"), attr)

    dask_array = _LazyDaskArray()
else:
    compute, persist, visualize = None, None, None
    dask_array = None


#: Map type name to the actual type (for upcast types)
#
# NOTE: this dictionary is meant to be public API and it can
#   be extended by users to include other classes.
upcast_type_map: dict[str, type | None] = {
    k: None
    for k in (
        "pint_pandas.pint_array.PintArray",
        "xarray.core.dataarray.DataArray",
        "xarray.core.dataset.Dataset",
        "xarray.core.variable.Variable",
        "pandas.core.series.Series",
        "pandas.core.frame.DataFrame",
        "pandas.Series",
        "pandas.DataFrame",
        "xarray.core.dataarray.DataArray",
    )
}
