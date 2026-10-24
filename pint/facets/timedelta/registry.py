"""
pint.facets.timedelta.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

from ...compat import TypeAlias, is_duck_array, np_timedelta64
from ..plain import GenericPlainRegistry, PlainQuantity
from .objects import TimedeltaQuantity, TimedeltaUnit

if TYPE_CHECKING:
    from ..._typing import Quantity, Unit

_DTYPE_TO_UNIT = {
    "timedelta64[Y]": "year",
    "timedelta64[M]": "month",
    "timedelta64[W]": "week",
    "timedelta64[D]": "day",
    "timedelta64[h]": "hour",
    "timedelta64[m]": "minute",
    "timedelta64[s]": "s",
    "timedelta64[ms]": "ms",
    "timedelta64[us]": "us",
    "timedelta64[ns]": "ns",
    "timedelta64[ps]": "ps",
    "timedelta64[fs]": "fs",
    "timedelta64[as]": "as",
}


class GenericTimedeltaRegistry[QuantityT: Quantity, UnitT: Unit](
    GenericPlainRegistry[QuantityT, UnitT]
):
    """Handle `datetime.timedelta` and `numpy.timedelta64` values."""

    def _is_timedelta(self, value: Any) -> bool:
        return isinstance(value, (datetime.timedelta, np_timedelta64)) or (
            not isinstance(value, PlainQuantity)
            and is_duck_array(value)
            and value.dtype.type == np_timedelta64
        )

    def _convert_timedelta(self, value: Any) -> tuple[Any, str]:
        """Convert a timedelta object to magnitude and unit string."""
        if isinstance(value, datetime.timedelta):
            return value.total_seconds(), "seconds"
        if isinstance(value, np_timedelta64) or value.dtype.type == np_timedelta64:
            return value.astype(float), _DTYPE_TO_UNIT[str(value.dtype)]
        raise TypeError(f"Cannot convert {value!r} to seconds.")


class TimedeltaRegistry(GenericTimedeltaRegistry[TimedeltaQuantity, TimedeltaUnit]):
    Quantity: TypeAlias = TimedeltaQuantity
    Unit: TypeAlias = TimedeltaUnit
