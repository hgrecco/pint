"""
    pint.formatter
    ~~~~~~~~~~~~~~

    Format units for pint.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import Callable, Any, TYPE_CHECKING, TypeVar, Optional, Union
from collections.abc import Iterable
from numbers import Number

from .babel_names import _babel_lengths, _babel_units
from .compat import babel_parse, HAS_BABEL

if TYPE_CHECKING:
    from .registry import UnitRegistry
    from .util import ItMatrix, UnitsContainer

    if HAS_BABEL:
        import babel

        Locale = babel.Locale
    else:
        Locale = TypeVar("Locale")


__JOIN_REG_EXP = re.compile(r"{\d*}")

FORMATTER = Callable[
    [
        Any,
    ],
    str,
]


def _join(fmt: str, iterable: Iterable[Any]) -> str:
    """Join an iterable with the format specified in fmt.

    The format can be specified in two ways:
    - PEP3101 format with two replacement fields (eg. '{} * {}')
    - The concatenating string (eg. ' * ')

    Parameters
    ----------
    fmt : str

    iterable :


    Returns
    -------
    str

    """
    if not iterable:
        return ""
    if not __JOIN_REG_EXP.search(fmt):
        return fmt.join(iterable)
    miter = iter(iterable)
    first = next(miter)
    for val in miter:
        ret = fmt.format(first, val)
        first = ret
    return first


_PRETTY_EXPONENTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _pretty_fmt_exponent(num: Number) -> str:
    """Format an number into a pretty printed exponent.

    Parameters
    ----------
    num : int

    Returns
    -------
    str

    """
    # unicode dot operator (U+22C5) looks like a superscript decimal
    ret = f"{num:n}".replace("-", "⁻").replace(".", "\u22C5")
    for n in range(10):
        ret = ret.replace(str(n), _PRETTY_EXPONENTS[n])
    return ret


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

    formatted = siunitx_format_unit(unit, registry)
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


def formatter(
    items: Iterable[tuple[str, Number]],
    as_ratio: bool = True,
    single_denominator: bool = False,
    product_fmt: str = " * ",
    division_fmt: str = " / ",
    power_fmt: str = "{} ** {}",
    parentheses_fmt: str = "({0})",
    exp_call: FORMATTER = "{:n}".format,
    locale: Optional[str] = None,
    babel_length: str = "long",
    babel_plural_form: str = "one",
    sort: bool = True,
) -> str:
    """Format a list of (name, exponent) pairs.

    Parameters
    ----------
    items : list
        a list of (name, exponent) pairs.
    as_ratio : bool, optional
        True to display as ratio, False as negative powers. (Default value = True)
    single_denominator : bool, optional
        all with terms with negative exponents are
        collected together. (Default value = False)
    product_fmt : str
        the format used for multiplication. (Default value = " * ")
    division_fmt : str
        the format used for division. (Default value = " / ")
    power_fmt : str
        the format used for exponentiation. (Default value = "{} ** {}")
    parentheses_fmt : str
        the format used for parenthesis. (Default value = "({0})")
    locale : str
        the locale object as defined in babel. (Default value = None)
    babel_length : str
        the length of the translated unit, as defined in babel cldr. (Default value = "long")
    babel_plural_form : str
        the plural form, calculated as defined in babel. (Default value = "one")
    exp_call : callable
         (Default value = lambda x: f"{x:n}")
    sort : bool, optional
        True to sort the formatted units alphabetically (Default value = True)

    Returns
    -------
    str
        the formula as a string.

    """

    if not items:
        return ""

    if as_ratio:
        fun = lambda x: exp_call(abs(x))
    else:
        fun = exp_call

    pos_terms, neg_terms = [], []

    if sort:
        items = sorted(items)
    for key, value in items:
        if locale and babel_length and babel_plural_form and key in _babel_units:
            _key = _babel_units[key]
            locale = babel_parse(locale)
            unit_patterns = locale._data["unit_patterns"]
            compound_unit_patterns = locale._data["compound_unit_patterns"]
            plural = "one" if abs(value) <= 0 else babel_plural_form
            if babel_length not in _babel_lengths:
                other_lengths = [
                    _babel_length
                    for _babel_length in reversed(_babel_lengths)
                    if babel_length != _babel_length
                ]
            else:
                other_lengths = []
            for _babel_length in [babel_length] + other_lengths:
                pat = unit_patterns.get(_key, {}).get(_babel_length, {}).get(plural)
                if pat is not None:
                    # Don't remove this positional! This is the format used in Babel
                    key = pat.replace("{0}", "").strip()
                    break
            division_fmt = compound_unit_patterns.get("per", {}).get(
                babel_length, division_fmt
            )
            power_fmt = "{}{}"
            exp_call = _pretty_fmt_exponent
        if value == 1:
            pos_terms.append(key)
        elif value > 0:
            pos_terms.append(power_fmt.format(key, fun(value)))
        elif value == -1 and as_ratio:
            neg_terms.append(key)
        else:
            neg_terms.append(power_fmt.format(key, fun(value)))

    if not as_ratio:
        # Show as Product: positive * negative terms ** -1
        return _join(product_fmt, pos_terms + neg_terms)

    # Show as Ratio: positive terms / negative terms
    pos_ret = _join(product_fmt, pos_terms) or "1"

    if not neg_terms:
        return pos_ret

    if single_denominator:
        neg_ret = _join(product_fmt, neg_terms)
        if len(neg_terms) > 1:
            neg_ret = parentheses_fmt.format(neg_ret)
    else:
        neg_ret = _join(division_fmt, neg_terms)

    return _join(division_fmt, [pos_ret, neg_ret])


# Extract just the type from the specification mini-language: see
# http://docs.python.org/2/library/string.html#format-specification-mini-language
# We also add uS for uncertainties.
_BASIC_TYPES = frozenset("bcdeEfFgGnosxX%uS")


def _parse_spec(spec: str) -> str:
    result = ""
    for ch in reversed(spec):
        if ch == "~" or ch in _BASIC_TYPES:
            continue
        elif ch in list(_FORMATTERS.keys()) + ["~"]:
            if result:
                raise ValueError("expected ':' after format specifier")
            else:
                result = ch
        elif ch.isalpha():
            raise ValueError("Unknown conversion specified " + ch)
        else:
            break
    return result


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


def siunitx_format_unit(units: UnitsContainer, registry) -> str:
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
    for unit, power in sorted(units.items()):
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


def extract_custom_flags(spec: str) -> str:
    import re

    if not spec:
        return ""

    # sort by length, with longer items first
    known_flags = sorted(_FORMATTERS.keys(), key=len, reverse=True)

    flag_re = re.compile("(" + "|".join(known_flags + ["~"]) + ")")
    custom_flags = flag_re.findall(spec)

    return "".join(custom_flags)


def remove_custom_flags(spec: str) -> str:
    for flag in sorted(_FORMATTERS.keys(), key=len, reverse=True) + ["~"]:
        if flag:
            spec = spec.replace(flag, "")
    return spec


def split_format(
    spec: str, default: str, separate_format_defaults: bool = True
) -> tuple[str, str]:
    mspec = remove_custom_flags(spec)
    uspec = extract_custom_flags(spec)

    default_mspec = remove_custom_flags(default)
    default_uspec = extract_custom_flags(default)

    if separate_format_defaults in (False, None):
        # should we warn always or only if there was no explicit choice?
        # Given that we want to eventually remove the flag again, I'd say yes?
        if spec and separate_format_defaults is None:
            if not uspec and default_uspec:
                warnings.warn(
                    (
                        "The given format spec does not contain a unit formatter."
                        " Falling back to the builtin defaults, but in the future"
                        " the unit formatter specified in the `default_format`"
                        " attribute will be used instead."
                    ),
                    DeprecationWarning,
                )
            if not mspec and default_mspec:
                warnings.warn(
                    (
                        "The given format spec does not contain a magnitude formatter."
                        " Falling back to the builtin defaults, but in the future"
                        " the magnitude formatter specified in the `default_format`"
                        " attribute will be used instead."
                    ),
                    DeprecationWarning,
                )
        elif not spec:
            mspec, uspec = default_mspec, default_uspec
    else:
        mspec = mspec or default_mspec
        uspec = uspec or default_uspec

    return mspec, uspec


def vector_to_latex(vec: Iterable[Any], fmtfun: FORMATTER = ".2f".format) -> str:
    return matrix_to_latex([vec], fmtfun)


def matrix_to_latex(matrix: ItMatrix, fmtfun: FORMATTER = ".2f".format) -> str:
    ret: list[str] = []

    for row in matrix:
        ret += [" & ".join(fmtfun(f) for f in row)]

    return r"\begin{pmatrix}%s\end{pmatrix}" % "\\\\ \n".join(ret)


def ndarray_to_latex_parts(
    ndarr, fmtfun: FORMATTER = ".2f".format, dim: tuple[int, ...] = tuple()
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
    ndarr, fmtfun: FORMATTER = ".2f".format, dim: tuple[int, ...] = tuple()
) -> str:
    return "\n".join(ndarray_to_latex_parts(ndarr, fmtfun, dim))
