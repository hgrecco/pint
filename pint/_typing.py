from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from importlib.util import find_spec
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
#
# The runtime branch below keeps numpy off the import path: the value of a
# `type` alias (PEP 695) is only computed when it is first accessed, so numpy is
# imported then -- if ever -- rather than when this module is imported. Whether
# numpy is installed is checked without importing it.
type _BuiltinScalar = complex | float | Decimal | Fraction
if TYPE_CHECKING:
    import numpy as np

    type Scalar = _BuiltinScalar | np.number[Any]
    type Array = np.ndarray[Any, Any]
else:
    # NOTE: redefining type aliases is not supported and may lead to type checker misbehavior
    _HAS_NUMPY = find_spec("numpy") is not None

    def _np_number():
        import numpy as np

        return np.number[Any]

    def _np_ndarray():
        import numpy as np

        return np.ndarray[Any, Any]

    type Scalar = (_BuiltinScalar | _np_number()) if _HAS_NUMPY else _BuiltinScalar
    type Array = _np_ndarray() if _HAS_NUMPY else Never

type Magnitude = Scalar | Array

type UnitLike = str | dict[str, Scalar] | UnitsContainer | Unit

type QuantityOrUnitLike = Quantity[Any] | UnitLike

type Shape = tuple[int, ...]

type FuncType = Callable[..., Any]


# TODO: Improve or delete types
QuantityArgument = Any


class Handler(Protocol):
    def __getitem__[T](self, item: type[T]) -> Callable[[T], None]: ...
