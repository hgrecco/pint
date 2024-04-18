"""
    pint.facets.numpy.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from typing import Any, Generic

from ...compat import TypeAlias
from ..plain import GenericPlainRegistry, KindT, QuantityT, UnitT
from .quantity import NumpyQuantity
from .unit import NumpyKind, NumpyUnit


class GenericNumpyRegistry(
    Generic[QuantityT, UnitT, KindT], GenericPlainRegistry[QuantityT, UnitT, KindT]
):
    pass


class NumpyRegistry(GenericPlainRegistry[NumpyQuantity[Any], NumpyUnit, NumpyKind]):
    Quantity: TypeAlias = NumpyQuantity[Any]
    Unit: TypeAlias = NumpyUnit
    Kind: TypeAlias = NumpyKind
