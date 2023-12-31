"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT


class BaseFormatter:
    def format_quantity(
        self, quantity: PlainQuantity[MagnitudeT], spec: str = ""
    ) -> str:
        # TODO Fill the proper functions
        return str(quantity.magnitude) + " " + self.format_unit(quantity.units, spec)

    def format_unit(self, unit: PlainUnit, spec: str = "") -> str:
        # TODO Fill the proper functions and discuss
        # how to make it that _units is not accessible directly
        return " ".join(k if v == 1 else f"{k} ** {v}" for k, v in unit._units.items())
