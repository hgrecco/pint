from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, Protocol
from decimal import Decimal
from fractions import Fraction


if TYPE_CHECKING:
    from .facets.plain import PlainQuantity as Quantity
    from .facets.plain import PlainUnit as Unit
    from .util import UnitsContainer


class ScalarProtocol(Protocol):
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

    def __gt__(self, other: Any) -> bool:
        ...

    def __lt__(self, other: Any) -> bool:
        ...

    def __ge__(self, other: Any) -> bool:
        ...

    def __le__(self, other: Any) -> bool:
        ...


class ArrayProtocol(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, key: Any) -> Any:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...


HAS_NUMPY = False
if TYPE_CHECKING:
    from .compat import HAS_NUMPY

if HAS_NUMPY:
    from .compat import np

    Scalar = Union[ScalarProtocol, float, int, Decimal, Fraction, np.number[Any]]
    Array = Union[np.ndarray[Any, Any]]
else:
    Scalar = Union[ScalarProtocol, float, int, Decimal, Fraction]
    Array = ArrayProtocol


# TODO: Change when Python 3.10 becomes minimal version.
Magnitude = Union[ScalarProtocol, ArrayProtocol]

UnitLike = Union[str, dict[str, Scalar], "UnitsContainer", "Unit"]

QuantityOrUnitLike = Union["Quantity", UnitLike]

Shape = tuple[int]

S = TypeVar("S")

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


# TODO: Improve or delete types
QuantityArgument = Any

T = TypeVar("T")


class Handler(Protocol):
    def __getitem__(self, item: type[T]) -> Callable[[T], None]:
        ...
