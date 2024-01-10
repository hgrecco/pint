"""
    pint.delegates.formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Formats quantities and units.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from .base_formatter import BaseFormatter


class Formatter(BaseFormatter):
    # TODO: this should derive from all relevant formaters to
    # reproduce the current behavior of Pint.
    pass


__all__ = [
    "Formatter",
]
