"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any
import locale
from ...compat import babel_parse
import re
from ...util import UnitsContainer, iterable

from ...compat import ndarray, np
from ...formatting import (
    _pretty_fmt_exponent,
    extract_custom_flags,
    format_unit,
    ndarray_to_latex,
    remove_custom_flags,
    siunitx_format_unit,
    split_format,
)

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...compat import Locale


_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class BaseFormatter:
    def format_quantity(
        self, quantity: PlainQuantity[MagnitudeT], spec: str = ""
    ) -> str:
        # TODO Fill the proper functions
        return str(quantity.magnitude) + " " + self.format_unit(quantity.units, spec)

    def format_unit(self, unit: PlainUnit, spec: str = "") -> str:
        # TODO Fill the proper functions and discuss
        # how to make it that _units is not accessible directly
        return " ".join(k if v == 1 else f"{k} ** {v}" for k, v in unit._units.items())


class BabelFormatter:
    locale: Optional[Locale] = None
    default_format: str = ""

    def set_locale(self, loc: Optional[str]) -> None:
        """Change the locale used by default by `format_babel`.

        Parameters
        ----------
        loc : str or None
            None` (do not translate), 'sys' (detect the system locale) or a locale id string.
        """
        if isinstance(loc, str):
            if loc == "sys":
                loc = locale.getdefaultlocale()[0]

            # We call babel parse to fail here and not in the formatting operation
            babel_parse(loc)

        self.locale = loc

    def format_quantity(
        self, quantity: PlainQuantity[MagnitudeT], spec: str = ""
    ) -> str:
        if self.locale is not None:
            return self.format_quantity_babel(quantity, spec)

        registry = quantity._REGISTRY

        mspec, uspec = split_format(
            spec, self.default_format, registry.separate_format_defaults
        )

        # If Compact is selected, do it at the beginning
        if "#" in spec:
            # TODO: don't replace '#'
            mspec = mspec.replace("#", "")
            uspec = uspec.replace("#", "")
            obj = quantity.to_compact()
        else:
            obj = quantity

        del quantity

        if "L" in uspec:
            allf = plain_allf = r"{}\ {}"
        elif "H" in uspec:
            allf = plain_allf = "{} {}"
            if iterable(obj.magnitude):
                # Use HTML table instead of plain text template for array-likes
                allf = (
                    "<table><tbody>"
                    "<tr><th>Magnitude</th>"
                    "<td style='text-align:left;'>{}</td></tr>"
                    "<tr><th>Units</th><td style='text-align:left;'>{}</td></tr>"
                    "</tbody></table>"
                )
        else:
            allf = plain_allf = "{} {}"

        if "Lx" in uspec:
            # the LaTeX siunitx code
            # TODO: add support for extracting options
            opts = ""
            ustr = siunitx_format_unit(obj.units._units, registry)
            allf = r"\SI[%s]{{{}}}{{{}}}" % opts
        else:
            # Hand off to unit formatting
            # TODO: only use `uspec` after completing the deprecation cycle
            ustr = self.format_unit(obj.units, mspec + uspec)

        # mspec = remove_custom_flags(spec)
        if "H" in uspec:
            # HTML formatting
            if hasattr(obj.magnitude, "_repr_html_"):
                # If magnitude has an HTML repr, nest it within Pint's
                mstr = obj.magnitude._repr_html_()
            else:
                if isinstance(obj.magnitude, ndarray):
                    # Use custom ndarray text formatting with monospace font
                    formatter = f"{{:{mspec}}}"
                    # Need to override for scalars, which are detected as iterable,
                    # and don't respond to printoptions.
                    if obj.magnitude.ndim == 0:
                        allf = plain_allf = "{} {}"
                        mstr = formatter.format(obj.magnitude)
                    else:
                        with np.printoptions(
                            formatter={"float_kind": formatter.format}
                        ):
                            mstr = (
                                "<pre>"
                                + format(obj.magnitude).replace("\n", "<br>")
                                + "</pre>"
                            )
                elif not iterable(obj.magnitude):
                    # Use plain text for scalars
                    mstr = format(obj.magnitude, mspec)
                else:
                    # Use monospace font for other array-likes
                    mstr = (
                        "<pre>"
                        + format(obj.magnitude, mspec).replace("\n", "<br>")
                        + "</pre>"
                    )
        elif isinstance(obj.magnitude, ndarray):
            if "L" in uspec:
                # Use ndarray LaTeX special formatting
                mstr = ndarray_to_latex(obj.magnitude, mspec)
            else:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                formatter = f"{{:{mspec}}}"
                if obj.magnitude.ndim == 0:
                    mstr = formatter.format(obj.magnitude)
                else:
                    with np.printoptions(formatter={"float_kind": formatter.format}):
                        mstr = format(obj.magnitude).replace("\n", "")
        else:
            mstr = format(obj.magnitude, mspec).replace("\n", "")

        if "L" in uspec and "Lx" not in uspec:
            mstr = _EXP_PATTERN.sub(r"\1\\times 10^{\2\3}", mstr)
        elif "H" in uspec or "P" in uspec:
            m = _EXP_PATTERN.match(mstr)
            _exp_formatter = (
                _pretty_fmt_exponent if "P" in uspec else lambda s: f"<sup>{s}</sup>"
            )
            if m:
                exp = int(m.group(2) + m.group(3))
                mstr = _EXP_PATTERN.sub(r"\1×10" + _exp_formatter(exp), mstr)

        if allf == plain_allf and ustr.startswith("1 /"):
            # Write e.g. "3 / s" instead of "3 1 / s"
            ustr = ustr[2:]
        return allf.format(mstr, ustr).strip()

    def format_quantity_babel(
        self, quantity: PlainQuantity[MagnitudeT], spec: str = "", **kwspec: Any
    ) -> str:
        spec = spec or self.default_format

        # standard cases
        if "#" in spec:
            spec = spec.replace("#", "")
            obj = quantity.to_compact()
        else:
            obj = quantity

        del quantity

        kwspec = kwspec.copy()
        if "length" in kwspec:
            kwspec["babel_length"] = kwspec.pop("length")

        loc = kwspec.get("locale", self.locale)
        if loc is None:
            raise ValueError("Provide a `locale` value to localize translation.")

        kwspec["locale"] = babel_parse(loc)
        kwspec["babel_plural_form"] = kwspec["locale"].plural_form(obj.magnitude)
        return "{} {}".format(
            format(obj.magnitude, remove_custom_flags(spec)),
            self.format_unit_babel(obj.units, spec, **kwspec),
        ).replace("\n", "")

    def format_unit(self, unit: PlainUnit, spec: str = "") -> str:
        registry = unit._REGISTRY

        _, uspec = split_format(
            spec, self.default_format, registry.separate_format_defaults
        )
        if "~" in uspec:
            if not unit._units:
                return ""
            units = UnitsContainer(
                {registry._get_symbol(key): value for key, value in unit._units.items()}
            )
            uspec = uspec.replace("~", "")
        else:
            units = unit._units

        return format_unit(units, uspec, registry=registry)

    def format_unit_babel(
        self,
        unit: PlainUnit,
        spec: str = "",
        locale: Optional[Locale] = None,
        **kwspec: Any,
    ) -> str:
        spec = spec or extract_custom_flags(self.default_format)

        if "~" in spec:
            if unit.dimensionless:
                return ""
            units = UnitsContainer(
                {
                    unit._REGISTRY._get_symbol(key): value
                    for key, value in unit._units.items()
                }
            )
            spec = spec.replace("~", "")
        else:
            units = unit._units

        locale = self.locale if locale is None else locale

        if locale is None:
            raise ValueError("Provide a `locale` value to localize translation.")
        else:
            kwspec["locale"] = babel_parse(locale)

        if "registry" not in kwspec:
            kwspec["registry"] = unit._REGISTRY

        return format_unit(units, spec, **kwspec)
