from typing import TYPE_CHECKING, Any, Protocol

from ._typing import Array

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

    # Logical operations
    def all(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array: ...
    def logical_not(self, array: Array, /) -> Array: ...

    bool: type["np.bool"]
