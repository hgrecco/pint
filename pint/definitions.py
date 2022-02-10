"""
    pint.definitions
    ~~~~~~~~~~~~~~~~

    Functions and classes related to unit definitions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Union

from .converters import Converter, LogarithmicConverter, OffsetConverter, ScaleConverter
from .errors import DefinitionSyntaxError
from .util import ParserHelper, UnitsContainer, _is_dim


@dataclass(frozen=True)
class PreprocessedDefinition:
    """Splits a definition into the constitutive parts.

    A definition is given as a string with equalities in a single line::

        ---------------> rhs
        a = b = c = d = e
        |   |   |   -------> aliases (optional)
        |   |   |
        |   |   -----------> symbol (use "_" for no symbol)
        |   |
        |   ---------------> value
        |
        -------------------> name
    """

    name: str
    symbol: Optional[str]
    aliases: Tuple[str, ...]
    value: str
    rhs_parts: Tuple[str, ...]

    @classmethod
    def from_string(cls, definition: str) -> PreprocessedDefinition:
        name, definition = definition.split("=", 1)
        name = name.strip()

        rhs_parts = tuple(res.strip() for res in definition.split("="))

        value, aliases = rhs_parts[0], tuple([x for x in rhs_parts[1:] if x != ""])
        symbol, aliases = (aliases[0], aliases[1:]) if aliases else (None, aliases)
        if symbol == "_":
            symbol = None
        aliases = tuple([x for x in aliases if x != "_"])

        return cls(name, symbol, aliases, value, rhs_parts)


class _NotNumeric(Exception):
    """Internal exception. Do not expose outside Pint"""

    def __init__(self, value):
        self.value = value


def numeric_parse(s: str, non_int_type: type = float):
    """Try parse a string into a number (without using eval).

    Parameters
    ----------
    s : str
    non_int_type : type

    Returns
    -------
    Number

    Raises
    ------
    _NotNumeric
        If the string cannot be parsed as a number.
    """
    ph = ParserHelper.from_string(s, non_int_type)

    if len(ph):
        raise _NotNumeric(s)

    return ph.scale


@dataclass(frozen=True)
class Definition:
    """Base class for definitions.

    Parameters
    ----------
    name : str
        Canonical name of the unit/prefix/etc.
    defined_symbol : str or None
        A short name or symbol for the definition.
    aliases : iterable of str
        Other names for the unit/prefix/etc.
    converter : callable or Converter or None
    """

    name: str
    defined_symbol: Optional[str]
    aliases: Tuple[str, ...]
    converter: Optional[Union[Callable, Converter]]

    def __post_init__(self):
        if isinstance(self.converter, str):
            raise TypeError(
                "The converter parameter cannot be an instance of `str`. Use `from_string` method"
            )

    @property
    def is_multiplicative(self) -> bool:
        return self.converter.is_multiplicative

    @property
    def is_logarithmic(self) -> bool:
        return self.converter.is_logarithmic

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> Definition:
        """Parse a definition.

        Parameters
        ----------
        definition : str or PreprocessedDefinition
        non_int_type : type

        Returns
        -------
        Definition or subclass of Definition
        """

        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        if definition.name.startswith("["):
            return DimensionDefinition.from_string(definition, non_int_type)
        elif definition.name.endswith("-"):
            return PrefixDefinition.from_string(definition, non_int_type)
        else:
            return UnitDefinition.from_string(definition, non_int_type)

    @property
    def symbol(self) -> str:
        return self.defined_symbol or self.name

    @property
    def has_symbol(self) -> bool:
        return bool(self.defined_symbol)

    def add_aliases(self, *alias: str) -> None:
        raise Exception("Cannot add aliases, definitions are inmutable.")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PrefixDefinition(Definition):
    """Definition of a prefix::

        <prefix>- = <amount> [= <symbol>] [= <alias>] [ = <alias> ] [...]

    Example::

        deca- =  1e+1  = da- = deka-
    """

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> PrefixDefinition:
        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        aliases = tuple(alias.strip("-") for alias in definition.aliases)
        if definition.symbol:
            symbol = definition.symbol.strip("-")
        else:
            symbol = definition.symbol

        try:
            converter = ScaleConverter(numeric_parse(definition.value, non_int_type))
        except _NotNumeric as ex:
            raise ValueError(
                f"Prefix definition ('{definition.name}') must contain only numbers, not {ex.value}"
            )

        return cls(definition.name.rstrip("-"), symbol, aliases, converter)


@dataclass(frozen=True)
class UnitDefinition(Definition):
    """Definition of a unit::

        <canonical name> = <relation to another unit or dimension> [= <symbol>] [= <alias>] [ = <alias> ] [...]

    Example::

        millennium = 1e3 * year = _ = millennia

    Parameters
    ----------
    reference : UnitsContainer
        Reference units.
    is_base : bool
        Indicates if it is a base unit.

    """

    reference: Optional[UnitsContainer] = None
    is_base: bool = False

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> "UnitDefinition":
        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        if ";" in definition.value:
            [converter, modifiers] = definition.value.split(";", 1)

            try:
                modifiers = dict(
                    (key.strip(), numeric_parse(value, non_int_type))
                    for key, value in (part.split(":") for part in modifiers.split(";"))
                )
            except _NotNumeric as ex:
                raise ValueError(
                    f"Unit definition ('{definition.name}') must contain only numbers in modifier, not {ex.value}"
                )

        else:
            converter = definition.value
            modifiers = {}

        converter = ParserHelper.from_string(converter, non_int_type)

        if not any(_is_dim(key) for key in converter.keys()):
            is_base = False
        elif all(_is_dim(key) for key in converter.keys()):
            is_base = True
        else:
            raise DefinitionSyntaxError(
                "Cannot mix dimensions and units in the same definition. "
                "Base units must be referenced only to dimensions. "
                "Derived units must be referenced only to units."
            )
        reference = UnitsContainer(converter)

        if not modifiers:
            converter = ScaleConverter(converter.scale)

        elif "offset" in modifiers:
            if modifiers.get("offset", 0.0) != 0.0:
                converter = OffsetConverter(converter.scale, modifiers["offset"])
            else:
                converter = ScaleConverter(converter.scale)

        elif "logbase" in modifiers and "logfactor" in modifiers:
            converter = LogarithmicConverter(
                converter.scale, modifiers["logbase"], modifiers["logfactor"]
            )

        else:
            raise DefinitionSyntaxError("Unable to assign a converter to the unit")

        return cls(
            definition.name,
            definition.symbol,
            definition.aliases,
            converter,
            reference,
            is_base,
        )


@dataclass(frozen=True)
class DimensionDefinition(Definition):
    """Definition of a dimension::

        [dimension name] = <relation to other dimensions>

    Example::

        [density] = [mass] / [volume]
    """

    reference: Optional[UnitsContainer] = None
    is_base: bool = False

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> DimensionDefinition:
        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        converter = ParserHelper.from_string(definition.value, non_int_type)

        if not converter:
            is_base = True
        elif all(_is_dim(key) for key in converter.keys()):
            is_base = False
        else:
            raise DefinitionSyntaxError(
                "Base dimensions must be referenced to None. "
                "Derived dimensions must only be referenced "
                "to dimensions."
            )
        reference = UnitsContainer(converter, non_int_type=non_int_type)

        return cls(
            definition.name,
            definition.symbol,
            definition.aliases,
            converter,
            reference,
            is_base,
        )


class AliasDefinition(Definition):
    """Additional alias(es) for an already existing unit::

        @alias <canonical name or previous alias> = <alias> [ = <alias> ] [...]

    Example::

        @alias meter = my_meter
    """

    def __init__(self, name: str, aliases: Iterable[str]) -> None:
        super().__init__(
            name=name, defined_symbol=None, aliases=aliases, converter=None
        )

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> AliasDefinition:

        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        name = definition.name[len("@alias ") :].lstrip()
        return AliasDefinition(name, tuple(definition.rhs_parts))
