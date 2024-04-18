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
from .objects import PlainKind, PlainQuantity, PlainUnit
from .quantity import MagnitudeT
from .registry import GenericPlainRegistry, KindT, PlainRegistry, QuantityT, UnitT

__all__ = [
    "GenericPlainRegistry",
    "PlainUnit",
    "PlainQuantity",
    "PlainKind",
    "PlainRegistry",
    "AliasDefinition",
    "DefaultsDefinition",
    "DimensionDefinition",
    "PrefixDefinition",
    "ScaleConverter",
    "UnitDefinition",
    "QuantityT",
    "UnitT",
    "KindT",
    "MagnitudeT",
]
