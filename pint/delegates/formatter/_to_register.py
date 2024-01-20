"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from ...compat import ndarray, np, Unpack
from ._helpers import (
    split_format,
    join_mu,
)

from ..._typing import Magnitude

from ._unit_handlers import format_compound_unit, BabelKwds, override_locale

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...registry import UnitRegistry


def register_unit_format(name: str):
    """register a function as a new format for units

    The registered function must have a signature of:

    .. code:: python

        def new_format(unit, registry, **options):
            pass

    Parameters
    ----------
    name : str
        The name of the new format (to be used in the format mini-language). A error is
        raised if the new format would overwrite a existing format.

    Examples
    --------
    .. code:: python

        @pint.register_unit_format("custom")
        def format_custom(unit, registry, **options):
            result = "<formatted unit>"  # do the formatting
            return result


        ureg = pint.UnitRegistry()
        u = ureg.m / ureg.s ** 2
        f"{u:custom}"
    """

    from ...formatting import _ORPHAN_FORMATTER

    # TODO: kwargs missing in typing
    def wrapper(func: Callable[[PlainUnit, UnitRegistry], str]):
        if name in _ORPHAN_FORMATTER._formatters:
            raise ValueError(f"format {name!r} already exists")  # or warn instead

        class NewFormatter:
            def format_magnitude(
                self,
                magnitude: Magnitude,
                mspec: str = "",
                **babel_kwds: Unpack[BabelKwds],
            ) -> str:
                with override_locale(
                    mspec, babel_kwds.get("locale", None)
                ) as format_number:
                    if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                        # Use custom ndarray text formatting--need to handle scalars differently
                        # since they don't respond to printoptions
                        with np.printoptions(formatter={"float_kind": format_number}):
                            mstr = format(magnitude).replace("\n", "")
                    else:
                        mstr = format_number(magnitude)

                return mstr

            def format_unit(
                self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
            ) -> str:
                units = unit._REGISTRY.UnitsContainer(
                    format_compound_unit(unit, uspec, **babel_kwds)
                )

                return func(units, registry=unit._REGISTRY, **babel_kwds)

            def format_quantity(
                self,
                quantity: PlainQuantity[MagnitudeT],
                qspec: str = "",
                **babel_kwds: Unpack[BabelKwds],
            ) -> str:
                registry = quantity._REGISTRY

                mspec, uspec = split_format(
                    qspec,
                    registry.formatter.default_format,
                    registry.separate_format_defaults,
                )

                joint_fstring = "{} {}"
                return join_mu(
                    joint_fstring,
                    self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
                    self.format_unit(quantity.units, uspec, **babel_kwds),
                )

        _ORPHAN_FORMATTER._formatters[name] = NewFormatter()

    return wrapper
