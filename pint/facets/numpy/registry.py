"""
    pint.facets.numpy.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from ..plain import PlainRegistry
from .quantity import NumpyQuantity
from .unit import NumpyUnit


class NumpyRegistry(PlainRegistry):
    Quantity = NumpyQuantity
    Unit = NumpyUnit
