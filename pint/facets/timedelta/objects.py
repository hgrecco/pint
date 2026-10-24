"""
pint.facets.timedelta.objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast, overload

from ..._typing import Magnitude, Scalar, UnitLike
from ...util import SharedRegistryObject
from ..plain import PlainQuantity, PlainUnit

if TYPE_CHECKING:
    import numpy as np
    import optype.numpy as npt


class TimedeltaQuantity[MagnitudeT: Magnitude](PlainQuantity[MagnitudeT]):
    """Quantity that can be built from and combined with timedeltas and datetimes."""

    @overload
    def __new__(
        cls, value: datetime.timedelta | np.timedelta64, units: UnitLike | None = None
    ) -> TimedeltaQuantity[float]: ...
    @overload
    def __new__(
        cls, value: str, units: UnitLike | None = None
    ) -> TimedeltaQuantity[Any]: ...
    @overload  # FIXME: This could be more precise
    def __new__[ScalarT: Scalar](  # type: ignore[misc]
        cls, value: Sequence[ScalarT], units: UnitLike | None = None
    ) -> TimedeltaQuantity[npt.Array1D]: ...
    @overload
    def __new__[T: Magnitude](
        cls, value: T, units: UnitLike | None = None
    ) -> TimedeltaQuantity[T]: ...
    @overload
    def __new__(cls, value: Self, units: UnitLike | None = None) -> Self: ...

    def __new__(cls, value: Any, units: UnitLike | None = None) -> Any:
        ureg = SharedRegistryObject.__new__(cls)._REGISTRY
        if ureg._is_timedelta(value):
            m, u = ureg._convert_timedelta(value)
            inst = super().__new__(cls, m, u)
            if units:
                inst.ito(units)
            return inst
        return super().__new__(cls, value, units)

    def _from_timedelta(self, other: Any) -> Any:
        if self._REGISTRY._is_timedelta(other):
            return self.__class__(other)
        return other

    def _add_sub(self, other, op):
        return super()._add_sub(self._from_timedelta(other), op)

    def _iadd_sub(self, other, op):
        return super()._iadd_sub(self._from_timedelta(other), op)

    def _mul_div(
        self, other, magnitude_op, units_op=None, parse_string_as_magnitude=False
    ):
        return super()._mul_div(
            self._from_timedelta(other),
            magnitude_op,
            units_op,
            parse_string_as_magnitude,
        )

    def _imul_div(self, other, magnitude_op, units_op=None):
        return super()._imul_div(self._from_timedelta(other), magnitude_op, units_op)

    def __iadd__(self, other):
        if isinstance(other, datetime.datetime):
            return cast("TimedeltaQuantity[int | float]", self).to_timedelta() + other
        return super().__iadd__(other)  # type: ignore[misc]

    def __add__(self, other):
        if isinstance(other, datetime.datetime):
            return cast("TimedeltaQuantity[int | float]", self).to_timedelta() + other
        return super().__add__(other)

    def __rsub__(self, other):
        if isinstance(other, datetime.datetime):
            return other - cast("TimedeltaQuantity[int | float]", self).to_timedelta()
        return super().__rsub__(other)

    def __rtruediv__(self, other):
        if self._REGISTRY._is_timedelta(other):
            return self.__class__(other) / self
        return super().__rtruediv__(other)

    __rdiv__ = __rtruediv__

    def to_timedelta(self: TimedeltaQuantity[int | float]) -> datetime.timedelta:
        return datetime.timedelta(microseconds=self.to("microseconds").magnitude)


class TimedeltaUnit(PlainUnit):
    pass
