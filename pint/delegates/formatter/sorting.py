"""
pint.delegates.formatter.sorting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sort functions used to order the units within a compound unit's string
representation, and the signature they must implement to be assigned to
``UnitRegistry.formatter.default_sort_func``.

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from ...compat import TypeAlias

if TYPE_CHECKING:
    from ...compat import Number
    from ...registry import UnitRegistry


#: Signature required by ``UnitRegistry.formatter.default_sort_func``.
#:
#: A sort function takes:
#:
#: - ``items``: an iterable of ``(display_name, exponent, unit_name)`` triplets,
#:   one per unit in the compound unit.
#: - ``registry``: the :class:`~pint.UnitRegistry` in use, or ``None``.
#:
#: and returns an iterable of the same triplets, in the desired display order.
SortFunc: TypeAlias = Callable[
    [Iterable[tuple[str, Any, str]], Any], Iterable[tuple[str, Any, str]]
]


def sort_by_unit_name(
    items: Iterable[tuple[str, Number, str]], _registry: UnitRegistry | None
) -> Iterable[tuple[str, Number, str]]:
    """Sort a list of units alphabetically by (canonical) unit name.

    This is the default ``default_sort_func``.
    """
    return sorted(items, key=lambda el: el[2])


def sort_by_display_name(
    items: Iterable[tuple[str, Number, str]], _registry: UnitRegistry | None
) -> Iterable[tuple[str, Number, str]]:
    """Sort a list of units alphabetically by display name."""
    return sorted(items)


def sort_by_dimensionality(
    items: Iterable[tuple[str, Number, str]], registry: UnitRegistry | None
) -> Iterable[tuple[str, Number, str]]:
    """Sort a list of units by dimensional order (from `registry.formatter.dim_order`).

    Parameters
    ----------
    items : tuple
        a list of tuples containing (unit names, exponent values).
    registry : UnitRegistry | None
        the registry to use for looking up the dimensions of each unit.

    Returns
    -------
    list
        the list of units sorted by most significant dimension first.

    Raises
    ------
    KeyError
        If unit cannot be found in the registry.
    """

    if registry is None:
        return items

    dim_order = registry.formatter.dim_order

    def sort_key(item: tuple[str, Number, str]):
        _display_name, _unit_exponent, unit_name = item
        cname = registry.get_name(unit_name)
        cname_dims = registry.get_dimensionality(cname) or {"[]": None}
        for cname_dim in cname_dims:
            if cname_dim in dim_order:
                return dim_order.index(cname_dim), cname

        raise KeyError(f"Unit {unit_name} (aka {cname}) has no recognized dimensions")

    return sorted(items, key=sort_key)
