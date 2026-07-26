from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Never, Protocol

if TYPE_CHECKING:
    from .facets.plain import PlainQuantity as Quantity
    from .facets.plain import PlainUnit as Unit
    from .util import UnitsContainer


# NOTE: There is no supported pattern for conditionally defining types
#   based on the availability of external dependencies.
#
# The pattern used here allows type checkers to correctly infer types when numpy
#   is available, but falls back to Any/Unknown (and not Never) in case of error
#   (tested: pyright 1.1.411)
#
# See https://discuss.python.org/t/conditional-imports-in-stub-files/50326 for context
type _BuiltinScalar = complex | float | Decimal | Fraction
try:
    import numpy as np

    type Scalar = _BuiltinScalar | np.number[Any]
    type Array = np.ndarray[Any, Any]
except ModuleNotFoundError:
    # NOTE: redefining type aliases is not supported and may lead to type checker misbehavior
    assert not TYPE_CHECKING
    type Scalar = _BuiltinScalar
    type Array = Never

type Magnitude = Scalar | Array

type UnitLike = str | dict[str, Scalar] | UnitsContainer | Unit

type QuantityOrUnitLike = Quantity[Any] | UnitLike

type Shape = tuple[int, ...]

type FuncType = Callable[..., Any]


# TODO: Improve or delete types
QuantityArgument = Any


class Handler(Protocol):
    def __getitem__[T](self, item: type[T]) -> Callable[[T], None]: ...
