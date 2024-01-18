"""
    pint.formatter
    ~~~~~~~~~~~~~~

    Format units for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING, TypeVar

from .compat import HAS_BABEL


# Backwards compatiblity stuff
from .delegates.formatter.latex import (
    vector_to_latex,  # noqa
    matrix_to_latex,  # noqa
    ndarray_to_latex_parts,  # noqa
    ndarray_to_latex,  # noqa
    latex_escape,  # noqa
    siunitx_format_unit,  # noqa
    _EXP_PATTERN,  # noqa
)  # noqa
from .delegates.formatter._helpers import (
    formatter,
    FORMATTER,  # noqa
    _BASIC_TYPES,  # noqa
    _parse_spec,  # noqa
    __JOIN_REG_EXP,  # noqa,
    _join,  # noqa
    _PRETTY_EXPONENTS,  # noqa
    _pretty_fmt_exponent,  # noqa
    extract_custom_flags,  # noqa
    remove_custom_flags,  # noqa
    split_format,  # noqa
)  # noqa

if TYPE_CHECKING:
    from .registry import UnitRegistry
    from .util import UnitsContainer

    if HAS_BABEL:
        import babel

        Locale = babel.Locale
    else:
        Locale = TypeVar("Locale")


#: _FORMATS maps format specifications to the corresponding argument set to
#: formatter().
_FORMATS: dict[str, dict[str, Any]] = {
    "P": {  # Pretty format.
        "as_ratio": True,
        "single_denominator": False,
        "product_fmt": "·",
        "division_fmt": "/",
        "power_fmt": "{}{}",
        "parentheses_fmt": "({})",
        "exp_call": _pretty_fmt_exponent,
    },
    "L": {  # Latex format.
        "as_ratio": True,
        "single_denominator": True,
        "product_fmt": r" \cdot ",
        "division_fmt": r"\frac[{}][{}]",
        "power_fmt": "{}^[{}]",
        "parentheses_fmt": r"\left({}\right)",
    },
    "Lx": {"siopts": "", "pm_fmt": " +- "},  # Latex format with SIunitx.
    "H": {  # HTML format.
        "as_ratio": True,
        "single_denominator": True,
        "product_fmt": r" ",
        "division_fmt": r"{}/{}",
        "power_fmt": r"{}<sup>{}</sup>",
        "parentheses_fmt": r"({})",
    },
    "": {  # Default format.
        "as_ratio": True,
        "single_denominator": False,
        "product_fmt": " * ",
        "division_fmt": " / ",
        "power_fmt": "{} ** {}",
        "parentheses_fmt": r"({})",
    },
    "C": {  # Compact format.
        "as_ratio": True,
        "single_denominator": False,
        "product_fmt": "*",  # TODO: Should this just be ''?
        "division_fmt": "/",
        "power_fmt": "{}**{}",
        "parentheses_fmt": r"({})",
    },
}

#: _FORMATTERS maps format names to callables doing the formatting
# TODO fix Callable typing
_FORMATTERS: dict[str, Callable] = {}


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

    def wrapper(func):
        if name in _FORMATTERS:
            raise ValueError(f"format {name!r} already exists")  # or warn instead
        _FORMATTERS[name] = func

    return wrapper


@register_unit_format("P")
def format_pretty(unit: UnitsContainer, registry: UnitRegistry, **options) -> str:
    return formatter(
        unit.items(),
        as_ratio=True,
        single_denominator=False,
        product_fmt="·",
        division_fmt="/",
        power_fmt="{}{}",
        parentheses_fmt="({})",
        exp_call=_pretty_fmt_exponent,
        **options,
    )


@register_unit_format("L")
def format_latex(unit: UnitsContainer, registry: UnitRegistry, **options) -> str:
    preprocessed = {rf"\mathrm{{{latex_escape(u)}}}": p for u, p in unit.items()}
    formatted = formatter(
        preprocessed.items(),
        as_ratio=True,
        single_denominator=True,
        product_fmt=r" \cdot ",
        division_fmt=r"\frac[{}][{}]",
        power_fmt="{}^[{}]",
        parentheses_fmt=r"\left({}\right)",
        **options,
    )
    return formatted.replace("[", "{").replace("]", "}")


@register_unit_format("Lx")
def format_latex_siunitx(
    unit: UnitsContainer, registry: UnitRegistry, **options
) -> str:
    if registry is None:
        raise ValueError(
            "Can't format as siunitx without a registry."
            " This is usually triggered when formatting a instance"
            ' of the internal `UnitsContainer` with a spec of `"Lx"`'
            " and might indicate a bug in `pint`."
        )

    formatted = siunitx_format_unit(unit.items(), registry)
    return rf"\si[]{{{formatted}}}"


@register_unit_format("H")
def format_html(unit: UnitsContainer, registry: UnitRegistry, **options) -> str:
    return formatter(
        unit.items(),
        as_ratio=True,
        single_denominator=True,
        product_fmt=r" ",
        division_fmt=r"{}/{}",
        power_fmt=r"{}<sup>{}</sup>",
        parentheses_fmt=r"({})",
        **options,
    )


@register_unit_format("D")
def format_default(unit: UnitsContainer, registry: UnitRegistry, **options) -> str:
    return formatter(
        unit.items(),
        as_ratio=True,
        single_denominator=False,
        product_fmt=" * ",
        division_fmt=" / ",
        power_fmt="{} ** {}",
        parentheses_fmt=r"({})",
        **options,
    )


@register_unit_format("C")
def format_compact(unit: UnitsContainer, registry: UnitRegistry, **options) -> str:
    return formatter(
        unit.items(),
        as_ratio=True,
        single_denominator=False,
        product_fmt="*",  # TODO: Should this just be ''?
        division_fmt="/",
        power_fmt="{}**{}",
        parentheses_fmt=r"({})",
        **options,
    )


def format_unit(unit, spec: str, registry=None, **options):
    # registry may be None to allow formatting `UnitsContainer` objects
    # in that case, the spec may not be "Lx"

    if not unit:
        if spec.endswith("%"):
            return ""
        else:
            return "dimensionless"

    if not spec:
        spec = "D"

    fmt = _FORMATTERS.get(spec)
    if fmt is None:
        raise ValueError(f"Unknown conversion specified: {spec}")

    return fmt(unit, registry=registry, **options)
