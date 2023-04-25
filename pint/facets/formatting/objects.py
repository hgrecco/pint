"""
    pint.facets.formatting.objects
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import re
from typing import Any

from ...compat import babel_parse, ndarray, np
from ...formatting import (
    _pretty_fmt_exponent,
    extract_custom_flags,
    format_unit,
    ndarray_to_latex,
    remove_custom_flags,
    siunitx_format_unit,
    split_format,
)
from ...util import UnitsContainer, iterable


class FormattingQuantity:
    _exp_pattern = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")

    def __format__(self, spec: str) -> str:
        if self._REGISTRY.fmt_locale is not None:
            return self.format_babel(spec)

        mspec, uspec = split_format(
            spec, self.default_format, self._REGISTRY.separate_format_defaults
        )

        # If Compact is selected, do it at the beginning
        if "#" in spec:
            # TODO: don't replace '#'
            mspec = mspec.replace("#", "")
            uspec = uspec.replace("#", "")
            obj = self.to_compact()
        else:
            obj = self

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
            ustr = siunitx_format_unit(obj.units._units, obj._REGISTRY)
            allf = r"\SI[%s]{{{}}}{{{}}}" % opts
        else:
            # Hand off to unit formatting
            # TODO: only use `uspec` after completing the deprecation cycle
            ustr = format(obj.units, mspec + uspec)

        # mspec = remove_custom_flags(spec)
        if "H" in uspec:
            # HTML formatting
            if hasattr(obj.magnitude, "_repr_html_"):
                # If magnitude has an HTML repr, nest it within Pint's
                mstr = obj.magnitude._repr_html_()
            else:
                if isinstance(self.magnitude, ndarray):
                    # Use custom ndarray text formatting with monospace font
                    formatter = "{{:{}}}".format(mspec)
                    # Need to override for scalars, which are detected as iterable,
                    # and don't respond to printoptions.
                    if self.magnitude.ndim == 0:
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
        elif isinstance(self.magnitude, ndarray):
            if "L" in uspec:
                # Use ndarray LaTeX special formatting
                mstr = ndarray_to_latex(obj.magnitude, mspec)
            else:
                # Use custom ndarray text formatting--need to handle scalars differently
                # since they don't respond to printoptions
                formatter = "{{:{}}}".format(mspec)
                if obj.magnitude.ndim == 0:
                    mstr = formatter.format(obj.magnitude)
                else:
                    with np.printoptions(formatter={"float_kind": formatter.format}):
                        mstr = format(obj.magnitude).replace("\n", "")
        else:
            mstr = format(obj.magnitude, mspec).replace("\n", "")

        if "L" in uspec and "Lx" not in uspec:
            mstr = self._exp_pattern.sub(r"\1\\times 10^{\2\3}", mstr)
        elif "H" in uspec or "P" in uspec:
            m = self._exp_pattern.match(mstr)
            _exp_formatter = (
                _pretty_fmt_exponent if "P" in uspec else lambda s: f"<sup>{s}</sup>"
            )
            if m:
                exp = int(m.group(2) + m.group(3))
                mstr = self._exp_pattern.sub(r"\1×10" + _exp_formatter(exp), mstr)

        if allf == plain_allf and ustr.startswith("1 /"):
            # Write e.g. "3 / s" instead of "3 1 / s"
            ustr = ustr[2:]
        return allf.format(mstr, ustr).strip()

    def _repr_pretty_(self, p, cycle):
        if cycle:
            super()._repr_pretty_(p, cycle)
        else:
            p.pretty(self.magnitude)
            p.text(" ")
            p.pretty(self.units)

    def format_babel(self, spec: str = "", **kwspec: Any) -> str:
        spec = spec or self.default_format

        # standard cases
        if "#" in spec:
            spec = spec.replace("#", "")
            obj = self.to_compact()
        else:
            obj = self
        kwspec = dict(kwspec)
        if "length" in kwspec:
            kwspec["babel_length"] = kwspec.pop("length")

        loc = kwspec.get("locale", self._REGISTRY.fmt_locale)
        if loc is None:
            raise ValueError("Provide a `locale` value to localize translation.")

        kwspec["locale"] = babel_parse(loc)
        kwspec["babel_plural_form"] = kwspec["locale"].plural_form(obj.magnitude)
        return "{} {}".format(
            format(obj.magnitude, remove_custom_flags(spec)),
            obj.units.format_babel(spec, **kwspec),
        ).replace("\n", "")

    def __str__(self) -> str:
        if self._REGISTRY.fmt_locale is not None:
            return self.format_babel()

        return format(self)


class FormattingUnit:
    def __str__(self):
        return format(self)

    def __format__(self, spec) -> str:
        _, uspec = split_format(
            spec, self.default_format, self._REGISTRY.separate_format_defaults
        )
        if "~" in uspec:
            if not self._units:
                return ""
            units = UnitsContainer(
                dict(
                    (self._REGISTRY._get_symbol(key), value)
                    for key, value in self._units.items()
                )
            )
            uspec = uspec.replace("~", "")
        else:
            units = self._units

        return format_unit(units, uspec, registry=self._REGISTRY)

    def format_babel(self, spec="", locale=None, **kwspec: Any) -> str:
        spec = spec or extract_custom_flags(self.default_format)

        if "~" in spec:
            if self.dimensionless:
                return ""
            units = UnitsContainer(
                dict(
                    (self._REGISTRY._get_symbol(key), value)
                    for key, value in self._units.items()
                )
            )
            spec = spec.replace("~", "")
        else:
            units = self._units

        locale = self._REGISTRY.fmt_locale if locale is None else locale

        if locale is None:
            raise ValueError("Provide a `locale` value to localize translation.")
        else:
            kwspec["locale"] = babel_parse(locale)

        return units.format_babel(spec, registry=self._REGISTRY, **kwspec)
