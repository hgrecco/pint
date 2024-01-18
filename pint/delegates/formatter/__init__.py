"""
    pint.delegates.formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Formats quantities and units.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from .full import MultipleFormatter


class Formatter(MultipleFormatter):
    # TODO: this should derive from all relevant formaters to
    # reproduce the current behavior of Pint.
    pass


__all__ = [
    "Formatter",
]
