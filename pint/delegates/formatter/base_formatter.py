"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Common class and function for all formatters.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from ... import protocols


class BaseFormatter:
    def format_quantity(self, quantity: protocols.Quantity, spec: str = "") -> str:
        # TODO Fill the proper functions
        return str(quantity.magnitude) + " " + self.format_unit(quantity.units, spec)

    def format_unit(self, unit: protocols.Unit, spec: str = "") -> str:
        # TODO Fill the proper functions and discuss
        # how to make it that _units is not accessible directly
        return " ".join(k if v == 1 else f"{k} ** {v}" for k, v in unit._units.items())
