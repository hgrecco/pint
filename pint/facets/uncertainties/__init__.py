"""
    pint.facets.Uncertainty
    ~~~~~~~~~~~~~~~~

    Adds pint the capability to interoperate with Uncertainty

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from typing import Generic, Any

from ...compat import Uncertainty, TypeAlias
from ..plain import (
    GenericPlainRegistry,
    PlainQuantity,
    QuantityT,
    UnitT,
    PlainUnit,
    MagnitudeT,
)


class UncertaintyQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    @property
    def value(self):
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.value * self.units
        else:
            return self._magnitude * self.units

    @property
    def error(self):
        if isinstance(self._magnitude, Uncertainty):
            return self._magnitude.error * self.units
        else:
            return (0 * self._magnitude) * self.units


class UncertaintyUnit(PlainUnit):
    pass


class GenericUncertaintyRegistry(
    Generic[QuantityT, UnitT], GenericPlainRegistry[QuantityT, UnitT]
):
    pass


class UncertaintyRegistry(
    GenericUncertaintyRegistry[UncertaintyQuantity[Any], UncertaintyUnit]
):
    Quantity: TypeAlias = UncertaintyQuantity[Any]
    Unit: TypeAlias = UncertaintyUnit
