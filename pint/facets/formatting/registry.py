"""
    pint.facets.formatting.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from ..plain import PlainRegistry
from .objects import FormattingQuantity, FormattingUnit


class FormattingRegistry(PlainRegistry):

    _quantity_class = FormattingQuantity
    _unit_class = FormattingUnit
