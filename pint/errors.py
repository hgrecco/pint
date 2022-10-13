"""
    pint.errors
    ~~~~~~~~~~~

    Functions and classes related to unit definitions and conversions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import typing as ty
from dataclasses import dataclass, fields

OFFSET_ERROR_DOCS_HTML = "https://pint.readthedocs.io/en/latest/nonmult.html"
LOG_ERROR_DOCS_HTML = "https://pint.readthedocs.io/en/latest/nonmult.html"

MSG_INVALID_UNIT_NAME = "is not a valid unit name (must follow Python identifier rules)"
MSG_INVALID_UNIT_SYMBOL = "is not a valid unit symbol (must not contain spaces)"
MSG_INVALID_UNIT_ALIAS = "is not a valid unit alias (must not contain spaces)"

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


def is_dim(name):
    return name[0] == "[" and name[-1] == "]"


def is_valid_prefix_name(name):
    return str.isidentifier(name) or name == ""


is_valid_unit_name = is_valid_system_name = is_valid_context_name = str.isidentifier


def _no_space(name):
    return name.strip() == name and " " not in name


is_valid_group_name = _no_space

is_valid_unit_alias = (
    is_valid_prefix_alias
) = is_valid_unit_symbol = is_valid_prefix_symbol = _no_space


def is_valid_dimension_name(name):
    return name == "[]" or (
        len(name) > 1 and is_dim(name) and str.isidentifier(name[1:-1])
    )


class WithDefErr:
    """Mixing class to make some classes more readable."""

    def def_err(self, msg):
        return DefinitionError(self.name, self.__class__.__name__, msg)


@dataclass(frozen=False)
class PintError(Exception):
    """Base exception for all Pint errors."""


@dataclass(frozen=False)
class DefinitionError(ValueError, PintError):
    """Raised when a definition is not properly constructed."""

    name: str
    definition_type: ty.Type
    msg: str

    def __str__(self):
        msg = f"Cannot define '{self.name}' ({self.definition_type}): {self.msg}"
        return msg

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class DefinitionSyntaxError(ValueError, PintError):
    """Raised when a textual definition has a syntax error."""

    msg: str

    def __str__(self):
        return self.msg

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class RedefinitionError(ValueError, PintError):
    """Raised when a unit or prefix is redefined."""

    name: str
    definition_type: ty.Type

    def __str__(self):
        msg = f"Cannot redefine '{self.name}' ({self.definition_type})"
        return msg

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class UndefinedUnitError(AttributeError, PintError):
    """Raised when the units are not defined in the unit registry."""

    unit_names: ty.Union[str, ty.Tuple[str, ...]]

    def __str__(self):
        if isinstance(self.unit_names, str):
            return f"'{self.unit_names}' is not defined in the unit registry"
        if (
            isinstance(self.unit_names, (tuple, list, set))
            and len(self.unit_names) == 1
        ):
            return f"'{tuple(self.unit_names)[0]}' is not defined in the unit registry"
        return f"{tuple(self.unit_names)} are not defined in the unit registry"

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class PintTypeError(TypeError, PintError):
    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class DimensionalityError(PintTypeError):
    """Raised when trying to convert between incompatible units."""

    units1: ty.Any
    units2: ty.Any
    dim1: str = ""
    dim2: str = ""
    extra_msg: str = ""

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
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class OffsetUnitCalculusError(PintTypeError):
    """Raised on ambiguous operations with offset units."""

    units1: ty.Any
    units2: ty.Optional[ty.Any] = None

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
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class LogarithmicUnitCalculusError(PintTypeError):
    """Raised on inappropriate operations with logarithmic units."""

    units1: ty.Any
    units2: ty.Optional[ty.Any] = None

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
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class UnitStrippedWarning(UserWarning, PintError):

    msg: str

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class UnexpectedScaleInContainer(Exception):
    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))
