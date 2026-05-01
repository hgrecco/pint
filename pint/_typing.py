from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Never, Protocol

if TYPE_CHECKING:
    from .facets.plain import PlainQuantity as Quantity
    from .facets.plain import PlainUnit as Unit
    from .util import UnitsContainer


HAS_NUMPY = False
if TYPE_CHECKING:
    from .compat import HAS_NUMPY

type _BuiltinScalar = complex | float | int | Decimal | Fraction
if HAS_NUMPY:
    from .compat import np

    type Scalar = _BuiltinScalar | np.number[Any]
    type Array = np.ndarray[Any, Any]
else:
    type Scalar = _BuiltinScalar
    type Array = Never

# TODO: Change when Python 3.10 becomes minimal version.
Magnitude = Scalar | Array

type UnitLike = str | dict[str, Scalar] | UnitsContainer | Unit

type QuantityOrUnitLike = Quantity | UnitLike

type Shape = tuple[int, ...]

type FuncType = Callable[..., Any]


# TODO: Improve or delete types
QuantityArgument = Any


class Handler(Protocol):
    def __getitem__[T](self, item: type[T]) -> Callable[[T], None]: ...
