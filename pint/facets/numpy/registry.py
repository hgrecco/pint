"""
pint.facets.numpy.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload, override

from ...compat import HAS_NUMPY, HAS_UNCERTAINTIES, TypeAlias, is_duck_array, ufloat
from ..plain import GenericPlainRegistry
from .quantity import NumpyQuantity
from .unit import NumpyUnit

if HAS_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from ..._typing import Magnitude, Quantity, Unit, UnitsContainer


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
        if (force_ndarray or force_ndarray_like) and not HAS_NUMPY:
            raise ValueError(
                "Cannot force to ndarray or ndarray-like when NumPy is not present."
            )
        # handle array-likes
        if isinstance(value, (list, tuple)):
            if HAS_NUMPY:
                return np.asarray(value)
            else:
                raise TypeError(
                    "lists and tuples are valid magnitudes for "
                    "Quantity only when NumPy is present."
                )
        elif HAS_UNCERTAINTIES:
            # TODO: Move this check to the measurement facet directly
            from pint.facets.measurement.objects import Measurement

            if isinstance(value, Measurement):
                return ufloat(value.value, value.error)
        if force_ndarray or (force_ndarray_like and not is_duck_array(value)):
            # NOTE: per the check above, we know numpy is installed
            return np.asarray(value)
        # let `PlainQuantity` handle all other objects
        return super()._into_magnitude(value, units=units, parse_strings=parse_strings)


class NumpyRegistry(GenericPlainRegistry[NumpyQuantity, NumpyUnit]):
    Quantity: TypeAlias = NumpyQuantity
    Unit: TypeAlias = NumpyUnit
