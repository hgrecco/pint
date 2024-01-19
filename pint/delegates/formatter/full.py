"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Any
import locale
from ...compat import babel_parse, Unpack
from ...util import iterable

from ..._typing import Magnitude
from .html import HTMLFormatter
from .latex import LatexFormatter, SIunitxFormatter
from .plain import RawFormatter, CompactFormatter, PrettyFormatter, DefaultFormatter
from ._unit_handlers import BabelKwds

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...compat import Locale


class MultipleFormatter:
    _formatters: dict[str, Any] = {}

    default_format: str = ""

    locale: Optional[Locale] = None
    babel_length: Literal["short", "long", "narrow"] = "long"

    def set_locale(self, loc: Optional[str]) -> None:
        """Change the locale used by default by `format_babel`.

        Parameters
        ----------
        loc : str or None
            None (do not translate), 'sys' (detect the system locale) or a locale id string.
        """
        if isinstance(loc, str):
            if loc == "sys":
                loc = locale.getdefaultlocale()[0]

            # We call babel parse to fail here and not in the formatting operation
            babel_parse(loc)

        self.locale = loc

    def __init__(self) -> None:
        self._formatters = {}
        self._formatters["raw"] = RawFormatter()
        self._formatters["D"] = DefaultFormatter()
        self._formatters["H"] = HTMLFormatter()
        self._formatters["P"] = PrettyFormatter()
        self._formatters["Lx"] = SIunitxFormatter()
        self._formatters["L"] = LatexFormatter()
        self._formatters["C"] = CompactFormatter()

    def get_formatter(self, spec: str):
        if spec == "":
            return self._formatters["raw"]
        for k, v in self._formatters.items():
            if k in spec:
                return v
        return self._formatters["D"]

    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        return self.get_formatter(mspec).format_magnitude(
            magnitude, mspec, **babel_kwds
        )

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        return self.get_formatter(uspec).format_unit(unit, uspec, **babel_kwds)

    def format_quantity(
        self,
        quantity: PlainQuantity[MagnitudeT],
        spec: str = "",
        **babel_kwds: Unpack[BabelKwds],
    ) -> str:
        spec = spec or self.default_format
        # If Compact is selected, do it at the beginning
        if "#" in spec:
            spec = spec.replace("#", "")
            obj = quantity.to_compact()
        else:
            obj = quantity

        del quantity

        use_plural = obj.magnitude > 1
        if iterable(use_plural):
            use_plural = True

        return self.get_formatter(spec).format_quantity(
            obj,
            spec,
            use_plural=babel_kwds.get("use_plural", use_plural),
            length=babel_kwds.get("length", self.babel_length),
            locale=babel_kwds.get("locale", self.locale),
        )

    #######################################
    # This is for backwards compatibility
    #######################################

    def format_unit_babel(
        self,
        unit: PlainUnit,
        spec: str = "",
        length: Optional[Literal["short", "long", "narrow"]] = "long",
        locale: Optional[Locale] = None,
    ) -> str:
        if self.locale is None and locale is None:
            raise ValueError(
                "format_babel requires a locale argumente if the Formatter locale is not set."
            )

        return self.format_unit(
            unit,
            spec or self.default_format,
            use_plural=False,
            length=length or self.babel_length,
            locale=locale or self.locale,
        )

    def format_quantity_babel(
        self,
        quantity: PlainQuantity[MagnitudeT],
        spec: str = "",
        length: Literal["short", "long", "narrow"] = "long",
        locale: Optional[Locale] = None,
    ) -> str:
        if self.locale is None and locale is None:
            raise ValueError(
                "format_babel requires a locale argumente if the Formatter locale is not set."
            )

        use_plural = quantity.magnitude > 1
        if iterable(use_plural):
            use_plural = True
        return self.format_quantity(
            quantity,
            spec or self.default_format,
            use_plural=use_plural,
            length=length or self.babel_length,
            locale=locale or self.locale,
        )
