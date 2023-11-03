"""
    pint.protocols
    ~~~~~~~~~~~~~~

    Protocols for Registry, Quantity and Unit.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details..
    :license: BSD, see LICENSE for more details.
"""
from typing_extensions import Self
from typing import Protocol, TypeVar, Optional, Sequence, overload

from ._typing import Magnitude, Scalar, UnitLike


MagnitudeT = TypeVar("MagnitudeT", bound=Magnitude)
ScalarT = TypeVar("ScalarT", bound=Scalar)


class Unit(Protocol):
    """Basic Unit protocol."""

    @property
    def dimensionless(self) -> bool:
        """Return True if the PlainUnit is dimensionless; False otherwise."""
        ...

    @property
    def dimensionality(self) -> dict[str, Scalar]:
        """Returns the dimenstionality of the object"""
        ...


class Quantity(Protocol):
    """Basic Quantity protocol."""

    @overload
    def __new__(cls, value: Scalar, units: Optional[UnitLike] = None) -> Self:
        ...

    @overload
    def __new__(cls, value: str, units: Optional[UnitLike] = None) -> Self:
        ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, value: Sequence[Scalar], units: Optional[UnitLike] = None
    ) -> Self:
        ...

    @overload
    def __new__(cls, value: Self, units: Optional[UnitLike] = None) -> Self:
        ...

    @property
    def magnitude(self) -> Scalar:
        ...

    @property
    def m(self) -> Scalar:
        ...

    def m_as(self, units: UnitLike) -> Scalar:
        ...

    @property
    def units(self) -> Unit:
        ...

    @property
    def u(self) -> Unit:
        ...

    @property
    def unitless(self) -> bool:
        ...

    @property
    def dimensionless(self) -> bool:
        ...

    @property
    def dimensionality(self) -> dict[str, Scalar]:
        ...


class Formatter(Protocol):
    def format_quantity(self, quantity: Quantity, spec: str = "") -> str:
        ...

    def format_unit(self, unit: Unit, spec: str = "") -> str:
        ...
