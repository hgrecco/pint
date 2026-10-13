"""
pint.errors
~~~~~~~~~~~

Functions and classes related to unit definitions and conversions.

:copyright: 2016 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import tokenize
import typing as ty
from io import BytesIO

if ty.TYPE_CHECKING:
    from .facets.plain.unit import UnitsContainer as UnitsContainerT

OFFSET_ERROR_DOCS_HTML = "https://pint.readthedocs.io/en/stable/user/nonmult.html"
LOG_ERROR_DOCS_HTML = "https://pint.readthedocs.io/en/stable/user/log_units.html"

MSG_INVALID_UNIT_NAME = "is not a valid unit name (must follow Python identifier rules)"
MSG_INVALID_UNIT_SYMBOL = "is not a valid unit symbol (must not contain spaces)"
MSG_INVALID_UNIT_ALIAS = (
    "is not a valid unit alias (must be a single name without whitespace or operators)"
)

MSG_INVALID_PREFIX_NAME = (
    "is not a valid prefix name (must follow Python identifier rules)"
)
MSG_INVALID_PREFIX_SYMBOL = "is not a valid prefix symbol (must not contain spaces)"
MSG_INVALID_PREFIX_ALIAS = "is not a valid prefix alias (must not contain spaces)"

MSG_INVALID_DIMENSION_NAME = "is not a valid dimension name (must follow Python identifier rules and enclosed by square brackets)"
MSG_INVALID_CONTEXT_NAME = (
    "is not a valid context name (must follow Python identifier rules)"
)
MSG_INVALID_GROUP_NAME = "is not a valid group name (must not contain spaces)"
MSG_INVALID_SYSTEM_NAME = (
    "is not a valid system name (must follow Python identifier rules)"
)


def is_dim(name: str) -> bool:
    """Return True if the name is flanked by square brackets `[` and `]`."""
    return name[0] == "[" and name[-1] == "]"


def is_valid_prefix_name(name: str) -> bool:
    """Return True if the name is a valid python identifier or empty."""
    return str.isidentifier(name) or name == ""


is_valid_unit_name = is_valid_system_name = is_valid_context_name = str.isidentifier


def _no_space(name: str) -> bool:
    """Return False if the name contains a space in any position."""
    return name.strip() == name and " " not in name


is_valid_group_name = _no_space

is_valid_prefix_alias = is_valid_unit_symbol = is_valid_prefix_symbol = _no_space


def is_valid_unit_alias(name: str) -> bool:
    """Return True if the alias is a single name in a unit expression."""
    if name in ("%", "‰"):
        return True

    # These characters are expanded into expressions by Pint's preprocessors,
    # even when Python's tokenizer considers them part of a name.
    if not name or any(char.isspace() or char in "%‰×±·⋅⁰¹²³⁴⁵⁶⁷⁸⁹" for char in name):
        return False

    try:
        tokens = tokenize.tokenize(BytesIO(name.encode("utf-8")).readline)
        next(tokens)  # Skip the encoding token, as in pint_eval.plain_tokenizer.
        token = next(tokens)
    except (tokenize.TokenError, SyntaxError, UnicodeError):
        return False

    # Accept Unicode unit names such as °C, which are not Python identifiers.
    return token.type == tokenize.NAME and token.string == name


def is_valid_dimension_name(name: str) -> bool:
    """Return True if the name is consistent with a dimension name.

    - flanked by square brackets.
    - empty dimension name or identifier.
    """

    # TODO: shall we check also fro spaces?
    return name == "[]" or (
        len(name) > 1 and is_dim(name) and str.isidentifier(name[1:-1])
    )


class WithDefErr:
    """Mixing class to make some classes more readable."""

    def def_err(self, msg: str):
        return DefinitionError(self.name, self.__class__, msg)


class PintError(Exception):
    """Base exception for all Pint errors."""


class DefinitionError(ValueError, PintError):
    """Raised when a definition is not properly constructed."""

    name: str
    definition_type: type
    msg: str

    def __init__(self, name: str, definition_type: type, msg: str):
        self.name = name
        self.definition_type = definition_type
        self.msg = msg

    def __str__(self):
        msg = f"Cannot define '{self.name}' ({self.definition_type}): {self.msg}"
        return msg

    def __reduce__(self):
        return self.__class__, (self.name, self.definition_type, self.msg)


class DefinitionSyntaxError(ValueError, PintError):
    """Raised when a textual definition has a syntax error."""

    msg: str

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg

    def __reduce__(self):
        return self.__class__, (self.msg,)


class RedefinitionError(ValueError, PintError):
    """Raised when a unit or prefix is redefined."""

    name: str
    definition_type: type

    def __init__(self, name: str, definition_type: type):
        self.name = name
        self.definition_type = definition_type

    def __str__(self):
        msg = f"Cannot redefine '{self.name}' ({self.definition_type})"
        return msg

    def __reduce__(self):
        return self.__class__, (self.name, self.definition_type)


class UndefinedUnitError(AttributeError, PintError):
    """Raised when the units are not defined in the unit registry."""

    unit_names: tuple[str, ...]

    def __init__(self, unit_names: str | ty.Iterable[str]):
        if isinstance(unit_names, str):
            self.unit_names = (unit_names,)
        else:
            self.unit_names = tuple(unit_names)

    def __str__(self):
        if len(self.unit_names) == 1:
            return f"'{tuple(self.unit_names)[0]}' is not defined in the unit registry"
        return f"{tuple(self.unit_names)} are not defined in the unit registry"

    def __reduce__(self):
        return self.__class__, (self.unit_names,)


class PintTypeError(TypeError, PintError):
    pass


class DimensionalityError(PintTypeError):
    """Raised when trying to convert between incompatible units."""

    units1: ty.Any
    units2: ty.Any
    dim1: "str | UnitsContainerT" = ""
    dim2: "str | UnitsContainerT" = ""
    extra_msg: str = ""

    def __init__(
        self,
        units1: ty.Any,
        units2: ty.Any,
        dim1: "str | UnitsContainerT" = "",
        dim2: "str | UnitsContainerT" = "",
        extra_msg: str = "",
    ) -> None:
        self.units1 = units1
        self.units2 = units2
        self.dim1 = dim1
        self.dim2 = dim2
        self.extra_msg = extra_msg

    def __str__(self):
        if self.dim1 or self.dim2:
            dim1 = f" ({self.dim1})"
            dim2 = f" ({self.dim2})"
        else:
            dim1 = ""
            dim2 = ""

        return (
            f"Cannot convert from '{self.units1}'{dim1} to "
            f"'{self.units2}'{dim2}{self.extra_msg}"
        )

    def __reduce__(self):
        return self.__class__, (
            self.units1,
            self.units2,
            self.dim1,
            self.dim2,
            self.extra_msg,
        )


class OffsetUnitCalculusError(PintTypeError):
    """Raised on ambiguous operations with offset units."""

    units1: ty.Any
    units2: ty.Optional[ty.Any] = None

    def __init__(self, units1: ty.Any, units2: ty.Optional[ty.Any] = None) -> None:
        self.units1 = units1
        self.units2 = units2

    def yield_units(self):
        yield self.units1
        if self.units2:
            yield self.units2

    def __str__(self):
        return (
            "Ambiguous operation with offset unit (%s)."
            % ", ".join(str(u) for u in self.yield_units())
            + " See "
            + OFFSET_ERROR_DOCS_HTML
            + " for guidance."
        )

    def __reduce__(self):
        return self.__class__, (self.units1, self.units2)


class LogarithmicUnitCalculusError(PintTypeError):
    """Raised on inappropriate operations with logarithmic units."""

    units1: ty.Any
    units2: ty.Optional[ty.Any] = None

    def __init__(self, units1: ty.Any, units2: ty.Optional[ty.Any] = None) -> None:
        self.units1 = units1
        self.units2 = units2

    def yield_units(self):
        yield self.units1
        if self.units2:
            yield self.units2

    def __str__(self):
        return (
            "Ambiguous operation with logarithmic unit (%s)."
            % ", ".join(str(u) for u in self.yield_units())
            + " See "
            + LOG_ERROR_DOCS_HTML
            + " for guidance."
        )

    def __reduce__(self):
        return self.__class__, (self.units1, self.units2)


class UnitStrippedWarning(UserWarning, PintError):
    msg: str

    def __init__(self, msg: str):
        self.msg = msg

    def __reduce__(self):
        return self.__class__, (self.msg,)


class UnexpectedScaleInContainer(Exception):
    pass


class UndefinedBehavior(UserWarning, PintError):
    msg: str

    def __init__(self, msg: str):
        self.msg = msg

    def __reduce__(self):
        return self.__class__, (self.msg,)
