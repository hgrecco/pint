"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations
from functools import partial

from typing import TYPE_CHECKING, Any
import re
from ...compat import ndarray, np, Unpack
from ._helpers import (
    _pretty_fmt_exponent,
    split_format,
    formatter,
    join_mu,
)

from ..._typing import Magnitude

from ._unit_handlers import format_compound_unit, BabelKwds, override_locale

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT


def format_number(value: Any, spec: str = "") -> str:
    if isinstance(value, float):
        return format(value, spec or ".16n")

    elif isinstance(value, int):
        return format(value, spec or "n")

    elif isinstance(value, ndarray) and value.ndim == 0:
        if issubclass(value.dtype.type, np.integer):
            return format(value, spec or "n")
        else:
            return format(value, spec or ".16n")
    else:
        return str(value)


_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class RawFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        return str(magnitude)

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return " * ".join(k if v == 1 else f"{k} ** {v}" for k, v in units)

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"
        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )


class DefaultFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(babel_kwds.get("locale", None)):
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(
                    formatter={"float_kind": partial(format_number, spec=mspec)}
                ):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude, mspec)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt=" * ",
            division_fmt=" / ",
            power_fmt="{} ** {}",
            parentheses_fmt=r"({})",
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"
        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )


class CompactFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(babel_kwds.get("locale", None)):
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(
                    formatter={"float_kind": partial(format_number, spec=mspec)}
                ):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude, mspec)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt="*",  # TODO: Should this just be ''?
            division_fmt="/",
            power_fmt="{}**{}",
            parentheses_fmt=r"({})",
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"

        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )


class PrettyFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(babel_kwds.get("locale", None)):
            if isinstance(magnitude, ndarray) and magnitude.ndim > 0:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                with np.printoptions(
                    formatter={"float_kind": partial(format_number, spec=mspec)}
                ):
                    mstr = format(magnitude).replace("\n", "")
            else:
                mstr = format_number(magnitude, mspec)

            m = _EXP_PATTERN.match(mstr)

            if m:
                exp = int(m.group(2) + m.group(3))
                mstr = _EXP_PATTERN.sub(r"\1×10" + _pretty_fmt_exponent(exp), mstr)

            return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=False,
            product_fmt="·",
            division_fmt="/",
            power_fmt="{}{}",
            parentheses_fmt="({})",
            exp_call=_pretty_fmt_exponent,
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.default_format, registry.separate_format_defaults
        )

        joint_fstring = "{} {}"

        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )
