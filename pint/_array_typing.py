from typing import TYPE_CHECKING, Any, Literal, Protocol

from ._typing import Array

if TYPE_CHECKING:
    import numpy as np


type _DTypeKind = Literal[
    "bool",
    "signed integer",
    "unsigned integer",
    "integral",
    "real floating",
    "complex floating",
    "numeric",
]


class ArrayAPIModule(Protocol):
    # Constructors
    def full_like(
        self,
        x: Array,
        /,
        fill_value: bool | int | float | complex,
        *,
        dtype: type | None = None,
        device: Any | None = None,
    ) -> Array: ...

    # Shape operations
    def broadcast_arrays(self, /, *arrays: Array) -> tuple[Array, ...]: ...

    # Logical operations
    def logical_not(self, array: Array, /) -> Array: ...
    def all(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array: ...
    def any(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array: ...

    # Introspection
    def isnan(self, x: Array, /) -> Array: ...
    def isdtype(
        self,
        /,
        dtype: np.dtype,
        kind: np.dtype | _DTypeKind | tuple[np.dtype | _DTypeKind, ...],
    ) -> bool: ...

    # dtypes (defined last to avoid conflicting with builtins)
    bool: type["np.bool"]
