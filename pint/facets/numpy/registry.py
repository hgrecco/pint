"""
pint.facets.numpy.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...compat import TypeAlias
from ..plain import GenericPlainRegistry
from .quantity import NumpyQuantity
from .unit import NumpyUnit

if TYPE_CHECKING:
    from ..._typing import Quantity, Unit


class GenericNumpyRegistry[QuantityT: Quantity, UnitT: Unit](
    GenericPlainRegistry[QuantityT, UnitT]
):
    pass


class NumpyRegistry(GenericPlainRegistry[NumpyQuantity, NumpyUnit]):
    Quantity: TypeAlias = NumpyQuantity
    Unit: TypeAlias = NumpyUnit
