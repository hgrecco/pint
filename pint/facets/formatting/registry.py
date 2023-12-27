"""
    pint.facets.formatting.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import Generic, Any

from ...compat import TypeAlias
from ..plain import GenericPlainRegistry, QuantityT, UnitT
from . import objects


class GenericFormattingRegistry(
    Generic[QuantityT, UnitT], GenericPlainRegistry[QuantityT, UnitT]
):
    pass


class FormattingRegistry(
    GenericFormattingRegistry[objects.FormattingQuantity[Any], objects.FormattingUnit]
):
    Quantity: TypeAlias = objects.FormattingQuantity[Any]
    Unit: TypeAlias = objects.FormattingUnit
