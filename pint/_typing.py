from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Tuple, TypeVar, Union

if TYPE_CHECKING:
    from .facets.plain import Quantity, Unit, UnitsContainer

UnitLike = Union[str, "UnitsContainer", "Unit"]

QuantityOrUnitLike = Union["Quantity", UnitLike]

Shape = Tuple[int, ...]

_MagnitudeType = TypeVar("_MagnitudeType")
S = TypeVar("S")

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
