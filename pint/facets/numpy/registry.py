"""
pint.facets.numpy.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload, override

from ...compat import HAS_NUMPY, HAS_UNCERTAINTIES, TypeAlias, is_duck_array, np_timedelta64, ufloat
from ..plain import GenericPlainRegistry, PlainQuantity
from .quantity import NumpyQuantity
from .unit import NumpyUnit

if HAS_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from ..._typing import Magnitude, Quantity, Unit, UnitsContainer

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
    @overload
    def _into_magnitude(
        self,
        value: object,
        /,
        units: UnitsContainer | None = None,
        *,
        parse_strings: bool = False,
        # numpy-specific kwargs
        force_ndarray: Literal[True],
        force_ndarray_like: bool | None = None,
    ) -> np.ndarray: ...
    @overload
    def _into_magnitude(
        self,
        value: object,
        /,
        units: UnitsContainer | None = None,
        *,
        parse_strings: bool = False,
        # numpy-specific kwargs
        force_ndarray: bool | None = None,
        force_ndarray_like: bool | None = None,
    ) -> Magnitude: ...
    @override
    def _into_magnitude(
        self,
        value: object,
        /,
        units: UnitsContainer | None = None,
        *,
        parse_strings: bool = False,
        # numpy-specific kwargs
        force_ndarray: bool | None = None,
        force_ndarray_like: bool | None = None,
    ) -> Magnitude:
        if force_ndarray is None:
            force_ndarray = self.force_ndarray
        if force_ndarray_like is None:
            force_ndarray_like = self.force_ndarray_like
        # process `force_ndarray` and `force_ndarray_like`
        if (force_ndarray or force_ndarray_like) and not HAS_NUMPY:
            raise ValueError(
                "Cannot force to ndarray or ndarray-like when NumPy is not present."
            )
        if force_ndarray or (force_ndarray_like and not is_duck_array(value)):
            # NOTE: per the check above, we know numpy is installed
            return np.asarray(value)
        # handle arrays & scalars
        if HAS_NUMPY and isinstance(value, np.ndarray | np.number):
            return value
        # handle array-likes
        elif isinstance(value, (list, tuple)):
            if HAS_NUMPY:
                return np.asarray(value)
            else:
                raise TypeError(
                    "lists and tuples are valid magnitudes for "
                    "Quantity only when NumPy is present."
                )
        # handle uncertainties (TODO: move this part to the measurement facet itself)
        elif HAS_UNCERTAINTIES:
            from uncertainties import UFloat

            from pint.facets.measurement.objects import Measurement

            if isinstance(value, UFloat):
                return value
            if isinstance(value, Measurement):
                return ufloat(value.value, value.error)
        # let `PlainQuantity` handle all other objects
        return super()._into_magnitude(value, units=units, parse_strings=parse_strings)
 
    def _as_magnitude_and_units(self, value: Any) -> tuple[Any, str] | None:
        if isinstance(value, np_timedelta64) or (
            not isinstance(value, PlainQuantity)
            and is_duck_array(value)
            and value.dtype.type == np_timedelta64
        ):
            return value.astype(float), _DTYPE_TO_UNIT[str(value.dtype)]
        return super()._as_magnitude_and_units(value)


class NumpyRegistry(GenericNumpyRegistry[NumpyQuantity, NumpyUnit]):
    Quantity: TypeAlias = NumpyQuantity
    Unit: TypeAlias = NumpyUnit
