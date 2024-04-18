"""
    pint.facets.kind.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2024 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from typing import Any, Generic

from ...compat import TypeAlias, ufloat
from ...util import create_class_with_registry, ParserHelper
from ..plain import GenericPlainRegistry, QuantityT, UnitT
from . import objects


class GenericKindRegistry(
    Generic[QuantityT, UnitT], GenericPlainRegistry[QuantityT, UnitT]
):
    Kind = objects.KindKind

    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""
        super()._init_dynamic_classes()

        self.Kind = create_class_with_registry(self, self.Kind)
    
    def get_compatible_kinds(self, dimensionality: UnitsContainer) -> frozenset[str]:
        return self._cache.kind_dimensional_equivalents.setdefault(
            dimensionality, frozenset()
        )

    def _dimensions_to_base_units(self, dimensions: UnitsContainer) -> UnitsContainer:
        """Get the base units for a dimension."""
        base_units = {self.Unit(unit).dimensonality: unit for unit in self._base_units}
        return UnitsContainer({base_units[dim]: exp for dim, exp in dimensions.items()})

    def _parse_kinds_as_container(self, input_string: str) -> UnitsContainer:
        """Parse a units expression and returns a UnitContainer with
        the canonical names.
        """

        if not input_string:
            return self.UnitsContainer()

        # Sanitize input_string with whitespaces.
        input_string = input_string.strip()

        kinds = ParserHelper.from_string(input_string, self.non_int_type)
        if kinds.scale != 1:
            raise ValueError("Unit expression cannot have a scaling factor.")

        return self.UnitsContainer(kinds)



class KindRegistry(
    GenericKindRegistry[
        objects.KindQuantity[Any],
        objects.KindUnit,
    ]
):
    Quantity: TypeAlias = objects.KindQuantity[Any]
    Unit: TypeAlias = objects.KindUnit
