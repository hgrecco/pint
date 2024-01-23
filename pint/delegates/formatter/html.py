"""
    pint.delegates.formatter.html
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Implements:
    - HTML: suitable for web/jupyter notebook outputs.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import re
from ...util import iterable
from ...compat import ndarray, np, Unpack
from ._spec_helpers import (
    split_format,
    join_mu,
    join_unc,
    remove_custom_flags,
)

from ..._typing import Magnitude
from ._format_helpers import BabelKwds, format_compound_unit, formatter, override_locale

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...facets.measurement import Measurement

_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class HTMLFormatter:
    """HTML localizable text formatter."""

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(mspec, babel_kwds.get("locale", None)) as format_number:
            if hasattr(magnitude, "_repr_html_"):
                # If magnitude has an HTML repr, nest it within Pint's
                mstr = magnitude._repr_html_()  # type: ignore
                assert isinstance(mstr, str)
            else:
                if isinstance(magnitude, ndarray):
                    # Need to override for scalars, which are detected as iterable,
                    # and don't respond to printoptions.
                    if magnitude.ndim == 0:
                        mstr = format_number(magnitude)
                    else:
                        with np.printoptions(formatter={"float_kind": format_number}):
                            mstr = (
                                "<pre>" + format(magnitude).replace("\n", "") + "</pre>"
                            )
                elif not iterable(magnitude):
                    # Use plain text for scalars
                    mstr = format_number(magnitude)
                else:
                    # Use monospace font for other array-likes
                    mstr = (
                        "<pre>"
                        + format_number(magnitude).replace("\n", "<br>")
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
            sort_func=lambda x: unit._REGISTRY.formatter.default_sort_func(
                x, unit._REGISTRY
            ),
        )

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        qspec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            qspec, registry.formatter.default_format, registry.separate_format_defaults
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

        return join_mu(
            joint_fstring,
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )

    def format_uncertainty(
        self,
        uncertainty,
        unc_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        unc_str = format(uncertainty, unc_spec).replace("+/-", " &plusmn; ")

        unc_str = re.sub(r"\)e\+0?(\d+)", r")×10<sup>\1</sup>", unc_str)
        unc_str = re.sub(r"\)e-0?(\d+)", r")×10<sup>-\1</sup>", unc_str)
        return unc_str

    def format_measurement(
        self,
        measurement: Measurement,
        meas_spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        registry = measurement._REGISTRY

        mspec, uspec = split_format(
            meas_spec,
            registry.formatter.default_format,
            registry.separate_format_defaults,
        )

        unc_spec = remove_custom_flags(meas_spec)

        joint_fstring = "{} {}"

        return join_unc(
            joint_fstring,
            "(",
            ")",
            self.format_uncertainty(measurement.magnitude, unc_spec, **babel_kwds),
            self.format_unit(measurement.units, uspec, **babel_kwds),
        )
