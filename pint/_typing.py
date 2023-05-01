from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, Protocol

# TODO: Remove when 3.11 becomes minimal version.
Self = TypeVar("Self")

if TYPE_CHECKING:
    from .facets.plain import PlainQuantity as Quantity
    from .facets.plain import PlainUnit as Unit
    from .util import UnitsContainer


class PintScalar(Protocol):
    def __add__(self, other: Any) -> Any:
        ...

    def __sub__(self, other: Any) -> Any:
        ...

    def __mul__(self, other: Any) -> Any:
        ...

    def __truediv__(self, other: Any) -> Any:
        ...

    def __floordiv__(self, other: Any) -> Any:
        ...

    def __mod__(self, other: Any) -> Any:
        ...

    def __divmod__(self, other: Any) -> Any:
        ...

    def __pow__(self, other: Any, modulo: Any) -> Any:
        ...


class PintArray(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...


# TODO: Change when Python 3.10 becomes minimal version.
# Magnitude = PintScalar | PintArray
Magnitude = Union[PintScalar, PintArray]

UnitLike = Union[str, "UnitsContainer", "Unit"]

QuantityOrUnitLike = Union["Quantity", UnitLike]

Shape = tuple[int, ...]

_MagnitudeType = TypeVar("_MagnitudeType")
S = TypeVar("S")

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
