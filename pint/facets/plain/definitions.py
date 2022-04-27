"""
    pint.facets.plain.definitions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Base unit converting capabilites.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

from ...converters import Converter
from ...definitions import Definition, PreprocessedDefinition
from ...errors import DefinitionSyntaxError
from ...util import ParserHelper, SourceIterator, UnitsContainer, _is_dim


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
class PrefixDefinition(Definition):
    """Definition of a prefix::

        <prefix>- = <amount> [= <symbol>] [= <alias>] [ = <alias> ] [...]

    Example::

        deca- =  1e+1  = da- = deka-
    """

    @classmethod
    def accept_to_parse(cls, preprocessed: PreprocessedDefinition):
        return preprocessed.name.endswith("-")

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
class UnitDefinition(Definition, default=True):
    """Definition of a unit::

        <canonical name> = <relation to another unit or dimension> [= <symbol>] [= <alias>] [ = <alias> ] [...]

    Example::

        millennium = 1e3 * year = _ = millennia

    Parameters
    ----------
    reference : UnitsContainer
        Reference units.
    is_base : bool
        Indicates if it is a plain unit.

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

        try:
            converter = Converter.from_arguments(scale=converter.scale, **modifiers)
        except Exception as ex:
            raise DefinitionSyntaxError(
                "Unable to assign a converter to the unit"
            ) from ex

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
    def accept_to_parse(cls, preprocessed: PreprocessedDefinition):
        return preprocessed.name.startswith("[")

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


@dataclass(frozen=True)
class DefaultsDefinition:
    """Definition for the @default directive"""

    content: Tuple[Tuple[str, str], ...]

    @classmethod
    def from_lines(cls, lines, non_int_type=float) -> DefaultsDefinition:
        source_iterator = SourceIterator(lines)
        next(source_iterator)
        out = []
        for lineno, part in source_iterator:
            k, v = part.split("=")
            out.append((k.strip(), v.strip()))

        return DefaultsDefinition(tuple(out))


@dataclass(frozen=True)
class ScaleConverter(Converter):
    """A linear transformation."""

    scale: float

    def to_reference(self, value, inplace=False):
        if inplace:
            value *= self.scale
        else:
            value = value * self.scale

        return value

    def from_reference(self, value, inplace=False):
        if inplace:
            value /= self.scale
        else:
            value = value / self.scale

        return value
