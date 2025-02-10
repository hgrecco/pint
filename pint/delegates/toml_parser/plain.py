"""
    pint.delegates.txt_defparser.plain
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Definitions for parsing:
    - Equality
    - CommentDefinition
    - PrefixDefinition
    - UnitDefinition
    - DimensionDefinition
    - DerivedDimensionDefinition
    - AliasDefinition

    Notices that some of the checks are done within the
    format agnostic parent definition class.

    See each one for a slighly longer description of the
    syntax.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from dataclasses import dataclass

import flexparser as fp

from ...converters import Converter
from ...facets.plain import definitions as definitions_
from ...facets.group import GroupDefinition as GroupDefinition_
from ...util import UnitsContainer
from ..base_defparser import ParserConfig
from . import common

import copy

@dataclass(frozen=True)
class Equality(definitions_.Equality):
    """An equality statement contains a left and right hand separated

    lhs and rhs should be space stripped.
    """

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[Equality]:
        if "=" not in s:
            return None
        parts = [p.strip() for p in s.split("=")]
        if len(parts) != 2:
            return common.DefinitionSyntaxError(
                f"Exactly two terms expected, not {len(parts)} (`{s}`)"
            )
        return cls(*parts)


@dataclass(frozen=True)
class CommentDefinition(definitions_.CommentDefinition):
    """Comments start with a # character.

        # This is a comment.
        ## This is also a comment.

    Captured value does not include the leading # character and space stripped.
    """

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[CommentDefinition]:
        if not s.startswith("#"):
            return None
        return cls(s[1:].strip())


@dataclass(frozen=True)
class PrefixDefinition(definitions_.PrefixDefinition):
    """Definition of a prefix::

        <prefix>- = <value> [= <symbol>] [= <alias>] [ = <alias> ] [...]

    Example::

        deca- =  1e+1  = da- = deka-
    """

    @classmethod
    def from_dict_and_config(
        cls, d: from_dict, config: ParserConfig
    ) -> PrefixDefinition:
        name = d["name"]
        value = d["value"]
        defined_symbol = d.get("defined_symbol", None)
        aliases = d.get("aliases", [])

        name = name.strip()
        # if not name.endswith("-"):
        #     return None

        # name = name.rstrip("-")
        # aliases = tuple(alias.strip().rstrip("-") for alias in aliases)

        # defined_symbol = None
        # if aliases:
        #     if aliases[0] == "_":
        #         aliases = aliases[1:]
        #     else:
        #         defined_symbol, *aliases = aliases

        #     aliases = tuple(alias for alias in aliases if alias not in ("", "_"))

        try:
            value = config.to_number(value)
        except definitions_.NotNumeric as ex:
            return common.DefinitionSyntaxError(
                f"Prefix definition ('{name}') must contain only numbers, not {ex.value}"
            )

        try:
            return cls(name, value, defined_symbol, aliases)
        except Exception as exc:
            return common.DefinitionSyntaxError(str(exc))


@dataclass(frozen=True)
class UnitDefinition(definitions_.UnitDefinition):
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

    @classmethod
    def from_dict_and_config(
        cls, d: from_dict, config: ParserConfig
    ) -> UnitDefinition:
        name = d["name"]
        value = d["value"]
        aliases = d.get("aliases", [])
        defined_symbol = d.get("defined_symbol", None)
        
        if ";" in value:
            [converter, modifiers] = value.split(";", 1)

            try:
                modifiers = {
                    key.strip(): config.to_number(value)
                    for key, value in (part.split(":") for part in modifiers.split(";"))
                }
            except definitions_.NotNumeric as ex:
                return common.DefinitionSyntaxError(
                    f"Unit definition ('{name}') must contain only numbers in modifier, not {ex.value}"
                )

        else:
            converter = value
            modifiers = {}

        converter = config.to_scaled_units_container(converter)

        try:
            reference = UnitsContainer(converter)
            # reference = converter.to_units_container()
        except common.DefinitionSyntaxError as ex:
            return common.DefinitionSyntaxError(f"While defining {name}: {ex}")

        try:
            converter = Converter.from_arguments(scale=converter.scale, **modifiers)
        except Exception as ex:
            return common.DefinitionSyntaxError(
                f"Unable to assign a converter to the unit {ex}"
            )

        try:
            return cls(name, defined_symbol, tuple(aliases), converter, reference)
        except Exception as ex:
            return common.DefinitionSyntaxError(str(ex))


@dataclass(frozen=True)
class DimensionDefinition(definitions_.DimensionDefinition):
    """Definition of a root dimension::

        [dimension name]

    Example::

        [volume]
    """

    @classmethod
    def from_dict_and_config(
        cls, d: from_dict, config: ParserConfig
    ) -> DimensionDefinition:

        return cls(**d)


@dataclass(frozen=True)
class DerivedDimensionDefinition(
    definitions_.DerivedDimensionDefinition
):
    """Definition of a derived dimension::

        [dimension name] = <relation to other dimensions>

    Example::

        [density] = [mass] / [volume]
    """

    @classmethod
    def from_dict_and_config(
        cls, d: from_dict, config: ParserConfig
    ) -> DerivedDimensionDefinition:
        name = d["name"]
        value = d["value"]

        try:
            reference = config.to_dimension_container(value)
        except common.DefinitionSyntaxError as exc:
            return common.DefinitionSyntaxError(
                f"In {name} derived dimensions must only be referenced "
                f"to dimensions. {exc}"
            )

        try:
            return cls(name.strip(), reference)
        except Exception as exc:
            return common.DefinitionSyntaxError(str(exc))


@dataclass(frozen=True)
class AliasDefinition(definitions_.AliasDefinition):
    """Additional alias(es) for an already existing unit::

        @alias <canonical name or previous alias> = <alias> [ = <alias> ] [...]

    Example::

        @alias meter = my_meter
    """

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[AliasDefinition]:
        if not s.startswith("@alias "):
            return None
        name, *aliases = s[len("@alias ") :].split("=")

        try:
            return cls(name.strip(), tuple(alias.strip() for alias in aliases))
        except Exception as exc:
            return common.DefinitionSyntaxError(str(exc))


@dataclass(frozen=True)
class GroupDefinition(
    GroupDefinition_
):
    """Definition of a group.

        @group <name> [using <group 1>, ..., <group N>]
            <definition 1>
            ...
            <definition N>
        @end

    See UnitDefinition and Comment for more parsing related information.

    Example::

        @group AvoirdupoisUS using Avoirdupois
            US_hundredweight = hundredweight = US_cwt
            US_ton = ton
            US_force_ton = force_ton = _ = US_ton_force
        @end

    """

    @classmethod
    def from_dict_and_config(cls, d, config) -> GroupDefinition:
        name = d["name"]
        using_group_names = d.get("using_group_names", ())
        definitions = []
        for key, value in d["definitions"].items():
            dat=copy.copy(value)
            dat['name']=key
            definitions.append(UnitDefinition.from_dict_and_config(dat, config))
        
        return cls(
            name, using_group_names, definitions
        )
