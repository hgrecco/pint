"""
    pint.delegates.formatter.base_formatter
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Common class and function for all formatters.
    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations
import functools

from typing import TYPE_CHECKING, Any, Iterable, Union

import re
from ._helpers import split_format, formatter, FORMATTER

from ..._typing import Magnitude
from ...compat import ndarray, Unpack, Number
from ._unit_handlers import BabelKwds, override_locale, format_compound_unit

if TYPE_CHECKING:
    from ...facets.plain import PlainQuantity, PlainUnit, MagnitudeT
    from ...util import ItMatrix
    from ...registry import UnitRegistry


def vector_to_latex(
    vec: Iterable[Any], fmtfun: FORMATTER | str = "{:.2n}".format
) -> str:
    return matrix_to_latex([vec], fmtfun)


def matrix_to_latex(matrix: ItMatrix, fmtfun: FORMATTER | str = "{:.2n}".format) -> str:
    ret: list[str] = []

    for row in matrix:
        ret += [" & ".join(fmtfun(f) for f in row)]

    return r"\begin{pmatrix}%s\end{pmatrix}" % "\\\\ \n".join(ret)


def ndarray_to_latex_parts(
    ndarr, fmtfun: FORMATTER = "{:.2n}".format, dim: tuple[int, ...] = tuple()
):
    if isinstance(fmtfun, str):
        fmtfun = fmtfun.format

    if ndarr.ndim == 0:
        _ndarr = ndarr.reshape(1)
        return [vector_to_latex(_ndarr, fmtfun)]
    if ndarr.ndim == 1:
        return [vector_to_latex(ndarr, fmtfun)]
    if ndarr.ndim == 2:
        return [matrix_to_latex(ndarr, fmtfun)]
    else:
        ret = []
        if ndarr.ndim == 3:
            header = ("arr[%s," % ",".join("%d" % d for d in dim)) + "%d,:,:]"
            for elno, el in enumerate(ndarr):
                ret += [header % elno + " = " + matrix_to_latex(el, fmtfun)]
        else:
            for elno, el in enumerate(ndarr):
                ret += ndarray_to_latex_parts(el, fmtfun, dim + (elno,))

        return ret


def ndarray_to_latex(
    ndarr, fmtfun: FORMATTER | str = "{:.2n}".format, dim: tuple[int, ...] = tuple()
) -> str:
    return "\n".join(ndarray_to_latex_parts(ndarr, fmtfun, dim))


def latex_escape(string: str) -> str:
    """
    Prepend characters that have a special meaning in LaTeX with a backslash.
    """
    return functools.reduce(
        lambda s, m: re.sub(m[0], m[1], s),
        (
            (r"[\\]", r"\\textbackslash "),
            (r"[~]", r"\\textasciitilde "),
            (r"[\^]", r"\\textasciicircum "),
            (r"([&%$#_{}])", r"\\\1"),
        ),
        str(string),
    )


def siunitx_format_unit(
    units: Iterable[tuple[str, Number]], registry: UnitRegistry
) -> str:
    """Returns LaTeX code for the unit that can be put into an siunitx command."""

    def _tothe(power: Union[int, float]) -> str:
        if isinstance(power, int) or (isinstance(power, float) and power.is_integer()):
            if power == 1:
                return ""
            elif power == 2:
                return r"\squared"
            elif power == 3:
                return r"\cubed"
            else:
                return rf"\tothe{{{int(power):d}}}"
        else:
            # limit float powers to 3 decimal places
            return rf"\tothe{{{power:.3f}}}".rstrip("0")

    lpos = []
    lneg = []
    # loop through all units in the container
    for unit, power in sorted(units):
        # remove unit prefix if it exists
        # siunitx supports \prefix commands

        lpick = lpos if power >= 0 else lneg
        prefix = None
        # TODO: fix this to be fore efficient and detect also aliases.
        for p in registry._prefixes.values():
            p = str(p.name)
            if len(p) > 0 and unit.find(p) == 0:
                prefix = p
                unit = unit.replace(prefix, "", 1)

        if power < 0:
            lpick.append(r"\per")
        if prefix is not None:
            lpick.append(rf"\{prefix}")
        lpick.append(rf"\{unit}")
        lpick.append(rf"{_tothe(abs(power))}")

    return "".join(lpos) + "".join(lneg)


_EXP_PATTERN = re.compile(r"([0-9]\.?[0-9]*)e(-?)\+?0*([0-9]+)")


class LatexFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(babel_kwds.get("locale", None)):
            if isinstance(magnitude, ndarray):
                mstr = ndarray_to_latex(magnitude, mspec or "n")
            else:
                mstr = format(magnitude, mspec or "n").replace("\n", "")

            mstr = _EXP_PATTERN.sub(r"\1\\times 10^{\2\3}", mstr)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        units = format_compound_unit(unit, uspec, **babel_kwds)

        preprocessed = {rf"\mathrm{{{latex_escape(u)}}}": p for u, p in units}
        formatted = formatter(
            preprocessed.items(),
            as_ratio=True,
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
        )
        return formatted.replace("[", "{").replace("]", "}")

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

        joint_fstring = r"{}\ {}"

        return joint_fstring.format(
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )


class SIunitxFormatter:
    def format_magnitude(
        self, magnitude: Magnitude, mspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        with override_locale(babel_kwds.get("locale", None)):
            if isinstance(magnitude, ndarray):
                mstr = ndarray_to_latex(magnitude, mspec or "n")
            else:
                mstr = format(magnitude, mspec or "n").replace("\n", "")

            mstr = _EXP_PATTERN.sub(r"\1\\times 10^{\2\3}", mstr)

        return mstr

    def format_unit(
        self, unit: PlainUnit, uspec: str = "", **babel_kwds: Unpack[BabelKwds]
    ) -> str:
        registry = unit._REGISTRY
        if registry is None:
            raise ValueError(
                "Can't format as siunitx without a registry."
                " This is usually triggered when formatting a instance"
                ' of the internal `UnitsContainer` with a spec of `"Lx"`'
                " and might indicate a bug in `pint`."
            )

        # TODO: not sure if I should call format_compound_unit here.
        # siunitx_format_unit requires certain specific names?

        units = format_compound_unit(unit, uspec, **babel_kwds)

        formatted = siunitx_format_unit(units, registry)
        return rf"\si[]{{{formatted}}}"

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

        joint_fstring = r"{}\ {}"

        return joint_fstring.format(
            self.format_magnitude(quantity.magnitude, mspec, **babel_kwds),
            self.format_unit(quantity.units, uspec, **babel_kwds),
        )
