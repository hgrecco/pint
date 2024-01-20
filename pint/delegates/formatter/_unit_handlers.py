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

import locale

from ...compat import babel_parse, ndarray

try:
    from numpy import integer as np_integer
except ImportError:
    np_integer = None

if TYPE_CHECKING:
    from ...registry import UnitRegistry
    from ...facets.plain import PlainUnit
    from ...compat import Locale, Number

T = TypeVar("T")


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


def _unit_mapper(
    units: Iterable[tuple[str, T]],
    shortener: Callable[
        [
            str,
        ],
        str,
    ],
) -> Iterable[tuple[str, T]]:
    return map(lambda el: (shortener(el[0]), el[1]), units)


def short_form(
    units: Iterable[tuple[str, T]],
    registry: UnitRegistry,
) -> Iterable[tuple[str, T]]:
    return _unit_mapper(units, registry.get_symbol)


def localized_form(
    units: Iterable[tuple[str, T]],
    use_plural: bool,
    length: Literal["short", "long", "narrow"],
    locale: Locale | str,
) -> Iterable[tuple[str, T]]:
    mapper = partial(
        format_unit_no_magnitude,
        use_plural=use_plural,
        length=length,
        locale=babel_parse(locale),
    )

    return _unit_mapper(units, mapper)


class BabelKwds(TypedDict):
    use_plural: bool
    length: Literal["short", "long", "narrow"] | None
    locale: Locale | str | None


def format_compound_unit(
    unit: PlainUnit,
    spec: str = "",
    use_plural: bool = False,
    length: Literal["short", "long", "narrow"] | None = None,
    locale: Locale | str | None = None,
) -> Iterable[tuple[str, Number]]:
    registry = unit._REGISTRY

    out = unit._units.items()

    if "~" in spec:
        out = short_form(out, registry)

    if locale is not None:
        out = localized_form(out, use_plural, length or "long", locale)

    return out


def format_number(value: Any, spec: str = "") -> str:
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


# TODO: ugly, ugly
# format has positional only arguments
# and this cannot be partialized
# and np requires a callable. We could create a lambda
def builtin_format(value: Any, spec: str = "") -> str:
    return format(value, spec)


@contextmanager
def override_locale(
    spec: str, locale: str | Locale | None
) -> Generator[Callable[[Any], str], Any, None]:
    if locale is None:
        yield ("{:" + spec + "}").format
    else:
        prev_locale_string = getlocale(LC_NUMERIC)
        if isinstance(locale, str):
            setlocale(LC_NUMERIC, locale)
        else:
            setlocale(LC_NUMERIC, str(locale))
        yield partial(format_number, spec=spec)
        setlocale(LC_NUMERIC, prev_locale_string)
