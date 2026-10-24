"""
pint.facets.numpy.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...compat import TypeAlias, is_duck_array, np_timedelta64
from ..plain import GenericPlainRegistry, PlainQuantity
from .quantity import NumpyQuantity
from .unit import NumpyUnit

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


class GenericNumpyRegistry[QuantityT: Quantity, UnitT: Unit](
    GenericPlainRegistry[QuantityT, UnitT]
):
    def _unit_carrying_parts(self, value: Any) -> tuple[Any, str] | None:
        if isinstance(value, np_timedelta64) or (
            not isinstance(value, PlainQuantity)
            and is_duck_array(value)
            and value.dtype.type == np_timedelta64
        ):
            return value.astype(float), _DTYPE_TO_UNIT[str(value.dtype)]
        return super()._unit_carrying_parts(value)


class NumpyRegistry(GenericNumpyRegistry[NumpyQuantity, NumpyUnit]):
    Quantity: TypeAlias = NumpyQuantity
    Unit: TypeAlias = NumpyUnit
