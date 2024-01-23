"""
    pint.delegates.formatter._format_helpers
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Convenient functions to help string formatting operations.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Generator,
    Iterable,
    TypeVar,
    Callable,
    TYPE_CHECKING,
    Literal,
    TypedDict,
)

from locale import getlocale, setlocale, LC_NUMERIC
from contextlib import contextmanager
from warnings import warn

import locale

from pint.delegates.formatter._spec_helpers import FORMATTER, _join

from ...compat import babel_parse, ndarray
from ...util import UnitsContainer

try:
    from numpy import integer as np_integer
except ImportError:
    np_integer = None

if TYPE_CHECKING:
    from ...registry import UnitRegistry
    from ...facets.plain import PlainUnit
    from ...compat import Locale, Number

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class BabelKwds(TypedDict):
    """Babel related keywords used in formatters."""

    use_plural: bool
    length: Literal["short", "long", "narrow"] | None
    locale: Locale | str | None


def format_number(value: Any, spec: str = "") -> str:
    """Format number

    This function might disapear in the future.
    Right now is aiding backwards compatible migration.
    """
    if isinstance(value, float):
        return format(value, spec or ".16n")

    elif isinstance(value, int):
        return format(value, spec or "n")

    elif isinstance(value, ndarray) and value.ndim == 0:
        if issubclass(value.dtype.type, np_integer):
            return format(value, spec or "n")
        else:
            return format(value, spec or ".16n")
    else:
        return str(value)


def builtin_format(value: Any, spec: str = "") -> str:
    """A keyword enabled replacement for builtin format

    format has positional only arguments
    and this cannot be partialized
    and np requires a callable.
    """
    return format(value, spec)


@contextmanager
def override_locale(
    spec: str, locale: str | Locale | None
) -> Generator[Callable[[Any], str], Any, None]:
    """Given a spec a locale, yields a function to format a number.

    IMPORTANT: When the locale is not None, this function uses setlocale
    and therefore is not thread safe.
    """

    if locale is None:
        # If locale is None, just return the builtin format function.
        yield ("{:" + spec + "}").format
    else:
        # If locale is not None, change it and return the backwards compatible
        # format_number.
        prev_locale_string = getlocale(LC_NUMERIC)
        if isinstance(locale, str):
            setlocale(LC_NUMERIC, locale)
        else:
            setlocale(LC_NUMERIC, str(locale))
        yield partial(format_number, spec=spec)
        setlocale(LC_NUMERIC, prev_locale_string)


def format_unit_no_magnitude(
    measurement_unit: str,
    use_plural: bool = True,
    length: Literal["short", "long", "narrow"] = "long",
    locale: Locale | str | None = locale.LC_NUMERIC,
) -> str | None:
    """Format a value of a given unit.

    THIS IS TAKEN FROM BABEL format_unit. But
    - No magnitude is returned in the string.
    - If the unit is not found, the same is given.
    - use_plural instead of value

    Values are formatted according to the locale's usual pluralization rules
    and number formats.

    >>> format_unit(12, 'length-meter', locale='ro_RO')
    u'metri'
    >>> format_unit(15.5, 'length-mile', locale='fi_FI')
    u'mailia'
    >>> format_unit(1200, 'pressure-millimeter-ofhg', locale='nb')
    u'millimeter kvikks\\xf8lv'
    >>> format_unit(270, 'ton', locale='en')
    u'tons'
    >>> format_unit(1234.5, 'kilogram', locale='ar_EG', numbering_system='default')
    u'كيلوغرام'


    The locale's usual pluralization rules are respected.

    >>> format_unit(1, 'length-meter', locale='ro_RO')
    u'metru'
    >>> format_unit(0, 'length-mile', locale='cy')
    u'mi'
    >>> format_unit(1, 'length-mile', locale='cy')
    u'filltir'
    >>> format_unit(3, 'length-mile', locale='cy')
    u'milltir'

    >>> format_unit(15, 'length-horse', locale='fi')
    Traceback (most recent call last):
        ...
    UnknownUnitError: length-horse is not a known unit in fi

    .. versionadded:: 2.2.0

    :param value: the value to format. If this is a string, no number formatting will be attempted.
    :param measurement_unit: the code of a measurement unit.
                             Known units can be found in the CLDR Unit Validity XML file:
                             https://unicode.org/repos/cldr/tags/latest/common/validity/unit.xml
    :param length: "short", "long" or "narrow"
    :param format: An optional format, as accepted by `format_decimal`.
    :param locale: the `Locale` object or locale identifier
    :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                             The special value "default" will use the default numbering system of the locale.
    :raise `UnsupportedNumberingSystemError`: If the numbering system is not supported by the locale.
    """
    locale = babel_parse(locale)
    from babel.units import _find_unit_pattern, get_unit_name

    q_unit = _find_unit_pattern(measurement_unit, locale=locale)
    if not q_unit:
        return measurement_unit

    unit_patterns = locale._data["unit_patterns"][q_unit].get(length, {})

    if use_plural:
        plural_form = "other"
    else:
        plural_form = "one"

    if plural_form in unit_patterns:
        return unit_patterns[plural_form].format("").replace("\xa0", "").strip()

    # Fall back to a somewhat bad representation.
    # nb: This is marked as no-cover, as the current CLDR seemingly has no way for this to happen.
    fallback_name = get_unit_name(
        measurement_unit, length=length, locale=locale
    )  # pragma: no cover
    return f"{fallback_name or measurement_unit}"  # pragma: no cover


def map_keys(
    func: Callable[
        [
            T,
        ],
        U,
    ],
    items: Iterable[tuple[T, V]],
) -> Iterable[tuple[U, V]]:
    """Map dict keys given an items view."""
    return map(lambda el: (func(el[0]), el[1]), items)


def short_form(
    units: Iterable[tuple[str, T]],
    registry: UnitRegistry,
) -> Iterable[tuple[str, T]]:
    """Replace each unit by its short form."""
    return map_keys(registry._get_symbol, units)


def localized_form(
    units: Iterable[tuple[str, T]],
    use_plural: bool,
    length: Literal["short", "long", "narrow"],
    locale: Locale | str,
) -> Iterable[tuple[str, T]]:
    """Replace each unit by its localized version."""
    mapper = partial(
        format_unit_no_magnitude,
        use_plural=use_plural,
        length=length,
        locale=babel_parse(locale),
    )

    return map_keys(mapper, units)


def format_compound_unit(
    unit: PlainUnit | UnitsContainer,
    spec: str = "",
    use_plural: bool = False,
    length: Literal["short", "long", "narrow"] | None = None,
    locale: Locale | str | None = None,
) -> Iterable[tuple[str, Number]]:
    """Format compound unit into unit container given
    an spec and locale.
    """

    # TODO: provisional? Should we allow unbounded units?
    # Should we allow UnitsContainer?
    registry = getattr(unit, "_REGISTRY", None)

    if isinstance(unit, UnitsContainer):
        out = unit.items()
    else:
        out = unit._units.items()

    if "~" in spec:
        if registry is None:
            raise ValueError(
                f"Can't short format a {type(unit)} without a registry."
                " This is usually triggered when formatting a instance"
                " of the internal `UnitsContainer`."
            )
        out = short_form(out, registry)

    if locale is not None:
        out = localized_form(out, use_plural, length or "long", locale)

    return out


def dim_sort(items: Iterable[tuple[str, Number]], registry: UnitRegistry):
    """Sort a list of units by dimensional order (from `registry.formatter.dim_order`).

    Parameters
    ----------
    items : tuple
        a list of tuples containing (unit names, exponent values).
    registry : UnitRegistry
        the registry to use for looking up the dimensions of each unit.

    Returns
    -------
    list
        the list of units sorted by most significant dimension first.

    Raises
    ------
    KeyError
        If unit cannot be found in the registry.
    """

    if registry is None:
        return items
    ret_dict = dict()
    dim_order = registry.formatter.dim_order
    for unit_name, unit_exponent in items:
        cname = registry.get_name(unit_name)
        if not cname:
            continue
        cname_dims = registry.get_dimensionality(cname)
        if len(cname_dims) == 0:
            cname_dims = {"[]": None}
        dim_types = iter(dim_order)
        while True:
            try:
                dim = next(dim_types)
                if dim in cname_dims:
                    if dim not in ret_dict:
                        ret_dict[dim] = list()
                    ret_dict[dim].append(
                        (
                            unit_name,
                            unit_exponent,
                        )
                    )
                    break
            except StopIteration:
                raise KeyError(
                    f"Unit {unit_name} (aka {cname}) has no recognized dimensions"
                )

    ret = sum([ret_dict[dim] for dim in dim_order if dim in ret_dict], [])
    return ret


def formatter(
    items: Iterable[tuple[str, Number]],
    as_ratio: bool = True,
    single_denominator: bool = False,
    product_fmt: str = " * ",
    division_fmt: str = " / ",
    power_fmt: str = "{} ** {}",
    parentheses_fmt: str = "({0})",
    exp_call: FORMATTER = "{:n}".format,
    sort: bool | None = None,
    sort_func: Callable[
        [
            Iterable[tuple[str, Number]],
        ],
        Iterable[tuple[str, Number]],
    ]
    | None = sorted,
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
    exp_call : callable
         (Default value = lambda x: f"{x:n}")
    sort : bool, optional
        True to sort the formatted units alphabetically (Default value = True)
    sort_func : callable
        If not None, `sort_func` returns its sorting of the formatted units

    Returns
    -------
    str
        the formula as a string.

    """

    if sort is False:
        warn(
            "The boolean `sort` argument is deprecated. "
            "Use `sort_func` to specify the sorting function (default=sorted) "
            "or None to keep units in the original order."
        )
        sort_func = None
    elif sort is True:
        warn(
            "The boolean `sort` argument is deprecated. "
            "Use `sort_func` to specify the sorting function (default=sorted) "
            "or None to keep units in the original order."
        )
        sort_func = sorted

    if sort_func is None:
        items = tuple(items)
    else:
        items = sort_func(items)

    if not items:
        return ""

    if as_ratio:
        fun = lambda x: exp_call(abs(x))
    else:
        fun = exp_call

    pos_terms, neg_terms = [], []

    for key, value in items:
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
