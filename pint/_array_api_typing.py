# Temporary replacement for [array-api-typing](https://github.com/data-apis/array-api-typing) until it's in a usable state.

from typing import TYPE_CHECKING, Literal, Never, Protocol

type Shape = tuple[int, ...]
type _DTypeKind = Literal[
    "bool",
    "signed integer",
    "unsigned integer",
    "integral",
    "real floating",
    "complex floating",
    "numeric",
]
try:
    import numpy as np

    if TYPE_CHECKING:
        import optype.numpy as npt

    type Array = np.ndarray
    type Device = npt.Device
except ModuleNotFoundError:
    assert not TYPE_CHECKING
    type Array = Never
    type Device = Never


class HasShape[S: Shape](Protocol):
    @property
    def shape(self, /) -> S: ...


class ArrayAPIModule(Protocol):
    # Constructors
    def full(
        self,
        /,
        shape: int | Shape,
        fill_value: bool | int | float | complex,
        *,
        dtype: type | None = None,
        device: Device | None = None,
    ) -> Array: ...
    def full_like(
        self,
        x: Array,
        /,
        fill_value: bool | int | float | complex,
        *,
        dtype: type | None = None,
        device: Device | None = None,
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

    # Rounding
    def ceil(self, x: Array, /) -> Array: ...
    def floor(self, x: Array, /) -> Array: ...

    # Other mathematical functions
    def abs(self, x: Array, /) -> Array: ...
    def log10(self, x: Array, /) -> Array: ...

    # Index operations
    def argmax(
        self, x: Array, /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array: ...
    def argmin(
        self, x: Array, /, *, axis: int | None = None, keepdims: bool = False
    ) -> Array: ...

    # Introspection
    def isfinite(self, x: Array, /) -> Array: ...
    def isinf(self, x: Array, /) -> Array: ...
    def isnan(self, x: Array, /) -> Array: ...
    def isdtype(
        self,
        /,
        dtype: "np.dtype",
        kind: "np.dtype | _DTypeKind | tuple[np.dtype | _DTypeKind, ...]",
    ) -> bool: ...

    # DType conversion
    def astype(
        self,
        x: Array,
        dtype: type,
        /,
        *,
        copy: bool = True,
        device: Device | None = None,
    ) -> Array: ...

    # DTypes (defined last to avoid conflicting with builtins)
    bool: type["np.bool"]
    float64: type["np.float64"]
