"""
pint.metrology
~~~~~~~~~~~~~~

Support for the metrology API standard
(https://github.com/quantity-dev/metrology-apis).

:copyright: 2026 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .facets.plain import GenericPlainRegistry, PlainQuantity, PlainUnit
    from .registry import ApplicationRegistry

    type AnyRegistry = GenericPlainRegistry[Any, Any] | ApplicationRegistry
    from .util import UnitsContainer

#: Version of the metrology API standard implemented by pint.
METROLOGY_API_VERSION = "0.0.1.dev0"


class MetrologyNamespace:
    """The metrology API namespace of a registry.

    Returned by ``__metrology_namespace__()`` on registries, quantities,
    units and dimensions.
    """

    def __init__(self, registry: AnyRegistry) -> None:
        self._registry = registry

    def asdimension(self, obj: str | UnitsContainer, /) -> UnitsContainer:
        """Convert an object, e.g. ``"[length] / [time]"``, to a dimension."""
        return self._registry.get_dimensionality(obj)

    def asunit(self, obj: str | PlainUnit, /) -> PlainUnit:
        """Convert an object to a unit."""
        return self._registry.Unit(obj)

    def asquantity(self, obj: Any, /, *, unit: str | PlainUnit) -> PlainQuantity:
        """Convert an object to a quantity, converting quantities to ``unit``."""
        return self._registry.Quantity(obj, unit)


def get_namespace(registry: AnyRegistry, api_version: str | None) -> MetrologyNamespace:
    if api_version is not None and api_version != METROLOGY_API_VERSION:
        raise ValueError(
            f"Unsupported metrology API version {api_version!r}; "
            f"pint implements {METROLOGY_API_VERSION!r}"
        )
    return MetrologyNamespace(registry)
