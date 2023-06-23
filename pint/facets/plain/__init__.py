"""
    pint.facets.plain
    ~~~~~~~~~~~~~~~~~

    Base implementation for registry, units and quantities.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from .definitions import (
    AliasDefinition,
    DefaultsDefinition,
    DimensionDefinition,
    PrefixDefinition,
    ScaleConverter,
    UnitDefinition,
)
from .objects import PlainQuantity, PlainUnit
from .registry import PlainRegistry, GenericPlainRegistry, QuantityT, UnitT
from .quantity import MagnitudeT

__all__ = [
    "GenericPlainRegistry",
    "PlainUnit",
    "PlainQuantity",
    "PlainRegistry",
    "AliasDefinition",
    "DefaultsDefinition",
    "DimensionDefinition",
    "PrefixDefinition",
    "ScaleConverter",
    "UnitDefinition",
    "QuantityT",
    "UnitT",
    "MagnitudeT",
]
