"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import re
from ...util import iterable
from ...compat import ndarray, np, Unpack
from ._helpers import (
    split_format,
    formatter,
)

from ..._typing import Magnitude
from ._unit_handlers import BabelKwds, format_compound_unit

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT


_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class HTMLFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        if hasattr(magnitude, "_repr_html_"):
            # If magnitude has an HTML repr, nest it within Pint's
            mstr = magnitude._repr_html_()  # type: ignore
            assert isinstance(mstr, str)
        else:
            if isinstance(magnitude, ndarray):
                # Use custom ndarray text formatting with monospace font
                formatter = f"{{:{mspec or 'n'}}}"
                # Need to override for scalars, which are detected as iterable,
                # and don't respond to printoptions.
                if magnitude.ndim == 0:
                    mstr = formatter.format(magnitude)
                else:
                    with np.printoptions(formatter={"float_kind": formatter.format}):
                        mstr = (
                            "<pre>" + format(magnitude).replace("\n", "<br>") + "</pre>"
                        )
            elif not iterable(magnitude):
                # Use plain text for scalars
                mstr = format(magnitude, mspec or "n")
            else:
                # Use monospace font for other array-likes
                mstr = (
                    "<pre>"
                    + format(magnitude, mspec or "n").replace("\n", "<br>")
                    + "</pre>"
                )

        m = _EXP_PATTERN.match(mstr)
        _exp_formatter = lambda s: f"<sup>{s}</sup>"

        if m:
            exp = int(m.group(2) + m.group(3))
            mstr = _EXP_PATTERN.sub(r"\1×10" + _exp_formatter(exp), mstr)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        return formatter(
            units,
            as_ratio=True,
            single_denominator=True,
            product_fmt=r" ",
            division_fmt=r"{}/{}",
            power_fmt=r"{}<sup>{}</sup>",
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

        if iterable(quantity.magnitude):
            # Use HTML table instead of plain text template for array-likes
            joint_fstring = (
                "<table><tbody>"
                "<tr><th>Magnitude</th>"
                "<td style='text-align:left;'>{}</td></tr>"
                "<tr><th>Units</th><td style='text-align:left;'>{}</td></tr>"
                "</tbody></table>"
            )
        else:
            joint_fstring = "{} {}"

        return joint_fstring.format(
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )
