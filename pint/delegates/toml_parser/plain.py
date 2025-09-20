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

import copy
from dataclasses import dataclass

from ...converters import Converter
from ...facets import context, group, system
from ...facets.plain import definitions as definitions_
from ...util import UnitsContainer, to_units_container
from ..base_defparser import ParserConfig
from . import common


@dataclass(frozen=True)
class PrefixDefinition(definitions_.PrefixDefinition):
    """Definition of a prefix::

        [<prefix>.<name>]
        value = "<value>"
        defined_symbol = "<symbol>"
        aliases = [
            "<alias 1>",
            ...
            "<alias N>",
        ]

    Example::

        [prefix.micro]
        value = "1e-6"
        defined_symbol = "µ"
        aliases = [
            "μ",
            "u",
            "mu",
            "mc",
        ]
    """

    @classmethod
    def from_dict_and_config(cls, d: dict, config: ParserConfig) -> PrefixDefinition:
        name = d["name"].strip()
        value = d["value"]
        defined_symbol = d.get("defined_symbol", None)
        aliases = d.get("aliases", [])

        try:
            value_number = config.to_number(value)
        except definitions_.NotNumeric as ex:
            return common.DefinitionSyntaxError(
                f"Prefix definition ('{name}') must contain only numbers, not {ex.value}"
            )

        try:
            return cls(name, value.strip(), value_number, defined_symbol, aliases)
        except Exception as exc:
            return common.DefinitionSyntaxError(str(exc))


@dataclass(frozen=True)
class UnitDefinition(definitions_.UnitDefinition):
    """Definition of a unit::

        [unit.<name>]
        aliases = [
            "<alias 1>",
            ...
            "<alias N>",
        ]
        defined_symbol = "<symbol>"
        value = "<relation to another unit>"

    Example::

        [unit.meter]
        defined_symbol = "m"
        aliases = [
            "metre",
        ]
        value = "[length]"

        [unit.kelvin]
        defined_symbol = "K"
        aliases = [
            "degK",
            "°K",
            "degree_Kelvin",
            "degreeK",
        ]
        value = "[temperature]; offset: 0"

        [unit.radian]
        defined_symbol = "rad"
        value = "[]"


        [unit.pi]
        defined_symbol = "π"
        value = "3.1415926535897932384626433832795028841971693993751"

    Parameters
    ----------
    reference : UnitsContainer
        Reference units.
    is_base : bool
        Indicates if it is a base unit.

    """

    @classmethod
    def from_dict_and_config(cls, d: dict, config: ParserConfig) -> UnitDefinition:
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
            return cls(name, value, defined_symbol, tuple(aliases), converter, reference)
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
    def from_dict_and_config(cls, d: dict, config: ParserConfig) -> DimensionDefinition:
        return cls(**d)


@dataclass(frozen=True)
class DerivedDimensionDefinition(definitions_.DerivedDimensionDefinition):
    """Definition of a derived dimension::

        [dimension.<name>]
        value = "<relation to another dimension>"

    Example::

        [dimension.density]
        value = "[mass] / [volume]"

    """

    @classmethod
    def from_dict_and_config(
        cls, d: dict, config: ParserConfig
    ) -> DerivedDimensionDefinition:
        name = "[" + d["name"] + "]"
        value = d["value"]

        try:
            reference = config.to_dimension_container(value)
        except common.DefinitionSyntaxError as exc:
            return common.DefinitionSyntaxError(
                f"In {name} derived dimensions must only be referenced "
                f"to dimensions. {exc}"
            )

        try:
            return cls(name.strip(), value.strip(), reference)
        except Exception as exc:
            return common.DefinitionSyntaxError(str(exc))


@dataclass(frozen=True)
class GroupDefinition(group.GroupDefinition):
    """Definition of a group. Can be composed of other groups.

        [group.<name>]
        using_group_names = [
            "<group 1>",
            ...
            "<group N>",
        ]
        # Definitions are the same as unit definitions
        [group.<name>.definitions.<unit name>]
        aliases = [
            "<alias 1>",
            ...
            "<alias N>",
        ]
        defined_symbol = "<symbol>"
        value = "<relation to another unit>"

    Example::


    [group.AvoirdupoisUK]
    using_group_names = [
        "Avoirdupois",
    ]

    [group.AvoirdupoisUK.definitions.UK_hundredweight]
    defined_symbol = "UK_cwt"
    value = "long_hundredweight"

    [group.AvoirdupoisUK.definitions.UK_ton]
    value = "long_ton"


    """

    @classmethod
    def from_dict_and_config(cls, d, config) -> GroupDefinition:
        name = d["name"]
        using_group_names = d.get("using_group_names", ())

        definitions = []
        for key, value in d["definitions"].items():
            dat = copy.copy(value)
            dat["name"] = key
            definitions.append(UnitDefinition.from_dict_and_config(dat, config))

        return cls(name, using_group_names, definitions)


@dataclass(frozen=True)
class SystemDefinition(system.SystemDefinition):
    """Definition of a system.

        [system.<name>]
        using = [
            "<group 1>",
            ...
            "<group N>",
        ]
        rules = [
            "<unit 1>: <base unit 1>",
            ...
            "<unit N>: <base unit N>",
        ]

    The syntax for the rule is:

        new_unit_name : old_unit_name

    where:
        - old_unit_name: a root unit part which is going to be removed from the system.
        - new_unit_name: a non root unit which is going to replace the old_unit.

    If the new_unit_name and the old_unit_name, the later and the colon can be omitted.

    See Rule for more parsing related information.

    Example::

    [system.Planck]
    using_group_names = [
        "international",
    ]
    rules = [
        "planck_length: meter",
        "planck_mass: gram",
        "planck_time: second",
        "planck_current: ampere",
        "planck_temperature: kelvin",
    ]

    """

    @classmethod
    def from_dict_and_config(cls, d, config) -> SystemDefinition:
        name = d["name"]
        using_group_names = d.get("using_group_names", ())

        rules = []
        for dat in d["rules"]:
            dat = [item.strip() for item in dat.split(":")]
            rules.append(system.BaseUnitRule(*dat))

        return cls(name, using_group_names, rules)


@dataclass(frozen=True)
class ContextDefinition(context.ContextDefinition):
    """Definition of a context.

        [context.<name>]
        aliases = [
            "<alias 1>",
            ...
            "<alias N>",
        ]
        relations = [
            # can establish unidirectional relationships between dimensions:
            "[dimension 1] -> [dimension A]: <equation 1>",
            # can establish bidirectional relationships between dimensions:
            "[dimension 2] <-> [dimension B]: <equation 2>",
            ...
            "[dimension N] -> [dimension Z]: <equation N>",
        ]
        [context.<name>.defaults]
        # default parameters can be defined for use in equations
        <parameter 1> = "<value 1>"
        ...
        <parameter N> = "<value N>"

    See ForwardRelation and BidirectionalRelation for more parsing related information.

    Example::

        [context.chemistry]
        aliases = [
            "chem",
        ]
        relations = [
            "[substance] -> [mass]: value * mw",
            "[mass] -> [substance]: value / mw",
            "[substance] / [volume] -> [mass] / [volume]: value * mw",
            "[mass] / [volume] -> [substance] / [volume]: value / mw",
            "[substance] / [mass] -> [mass] / [mass]: value * mw",
            "[mass] / [mass] -> [substance] / [mass]: value / mw",
            "[substance] / [volume] -> [substance]: value * volume",
            "[substance] -> [substance] / [volume]: value / volume",
            "[substance] / [mass] -> [substance]: value * solvent_mass",
            "[substance] -> [substance] / [mass]: value / solvent_mass",
            "[substance] / [mass] -> [substance]/[volume]: value * solvent_mass / volume",
            "[substance] / [volume] -> [substance] / [mass]: value / solvent_mass * volume",
        ]

        [context.chemistry.defaults]
        mw = "0"
        volume = "0"
        solvent_mass = "0"

    """

    @classmethod
    def from_dict_and_config(cls, d, config) -> ContextDefinition:
        name = d["name"]
        aliases = d.get("aliases", ())
        defaults = d.get("defaults", {})
        relations = []

        for relation in d["relations"]:
            dat, eqn = relation.split(":")
            if "<->" in dat:
                src, dst = dat.split("<->")
                obj = context.BidirectionalRelation
            else:
                src, dst = dat.split("->")
                obj = context.ForwardRelation
            src = to_units_container(src)
            dst = to_units_container(dst)
            relations.append(obj(relation, src, dst, eqn))
        redefinitions = d.get("redefinitions", {})

        return cls(name, aliases, defaults, relations, redefinitions)
