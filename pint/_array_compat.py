"""
pint._array_compat
~~~~~~~~~~~~~~~~~~

Compatibility layer for supporting arrays as pint magnitudes.

APIs are tried in the following order:
- Array API standard (experimental)
- Duck arrays (e.g. dask/pandas; partial support)
- NumPy ndarray (full support)

:copyright: 2013 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from typing import TYPE_CHECKING, Any, Protocol

from typing_extensions import TypeIs

from ._typing import Array
from .compat import HAS_NUMPY

if TYPE_CHECKING:
    import numpy as np


class ArrayAPIModule(Protocol):
    def broadcast_arrays(self, /, *arrays: Array) -> tuple[Array, ...]: ...
    def full_like(
        self,
        x: Array,
        /,
        fill_value: bool | int | float | complex,
        *,
        dtype: type | None = None,
        device: Any | None = None,
    ) -> Array: ...
    def logical_not(self, array: Array, /) -> Array: ...

    bool: type["np.bool"]


def namespace(array: Array, /) -> ArrayAPIModule:
    if hasattr(array, "__array_namespace__"):
        return array.__array_namespace__()  # pyright: ignore[reportReturnType]
    # backwards-compatible duck array support
    elif HAS_NUMPY:
        import numpy

        return numpy
    else:
        raise TypeError("Pint only fully supports duck arrays if NumPy is available.")


def is_array_or_duck(obj: object) -> TypeIs[Array]:
    return hasattr(obj, "__array_namespace__") or (
        not hasattr(obj, "_magnitude")
        and not hasattr(obj, "_units")
        and hasattr(obj, "__array_function__")
        and hasattr(obj, "ndim")
        and hasattr(obj, "dtype")
    )


def size(array: Array, /) -> int:
    if hasattr(array, "size"):
        return array.size
    return namespace(array).size(array)  # pyright: ignore[reportAttributeAccessIssue]
