"""
pint.delegates.toml_parser.plain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build definitions from the tables of a TOML definitions file::

    [defaults]
    group = "international"
    system = "mks"

    [prefix]
    micro = { value = "1e-6", defined_symbol = "µ", aliases = ["μ", "u"] }

    [dimension]
    density = { value = "[mass] / [volume]" }

    [unit]
    meter = { value = "[length]", defined_symbol = "m", aliases = ["metre"] }
    degree_Celsius = { value = "kelvin; offset: 273.15", defined_symbol = "°C" }

    [[group]]
    name = "AvoirdupoisUK"
    using_group_names = ["Avoirdupois"]
    [group.definitions]
    UK_hundredweight = { value = "long_hundredweight", defined_symbol = "UK_cwt" }

    [[system]]
    name = "Planck"
    using_group_names = ["international"]
    rules = ["planck_length: meter", "planck_mass: gram"]

    [[context]]
    name = "spectroscopy"
    aliases = ["sp"]
    defaults = { n = 1 }
    relations = ["[length] <-> [frequency]: speed_of_light / n / value"]
    [context.redefinitions]
    # same format as [unit], without symbol or aliases

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import Any

from ...converters import Converter
from ...errors import DefinitionSyntaxError
from ...facets import context, group, system
from ...facets.plain import definitions
from ..base_defparser import ParserConfig


def _tuple(d: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(d.get(key, ()))


def prefix(name: str, d: dict[str, Any], config: ParserConfig):
    value = d["value"]
    try:
        value_number = config.to_number(value)
    except definitions.NotNumeric as ex:
        raise DefinitionSyntaxError(
            f"Prefix definition ('{name}') must contain only numbers, not {ex.value}"
        )
    return definitions.PrefixDefinition(
        name,
        value_number,
        d.get("defined_symbol"),
        _tuple(d, "aliases"),
        value_text=value,
    )


def unit(name: str, d: dict[str, Any], config: ParserConfig):
    value = d["value"]
    converter, *modifier_parts = value.split(";")
    try:
        modifiers = {}
        for part in modifier_parts:
            key, number = part.split(":")
            modifiers[key.strip()] = config.to_number(number)
    except definitions.NotNumeric as ex:
        raise DefinitionSyntaxError(
            f"Unit definition ('{name}') must contain only numbers in modifier, not {ex.value}"
        )
    except ValueError:
        raise DefinitionSyntaxError(
            f"Unit definition ('{name}') has a malformed modifier in '{value}'"
        )

    scaled = config.to_scaled_units_container(converter)
    try:
        converter = Converter.from_arguments(scale=scaled.scale, **modifiers)
    except Exception as ex:
        raise DefinitionSyntaxError(
            f"Unable to assign a converter to the unit {name}: {ex}"
        )

    return definitions.UnitDefinition(
        name,
        d.get("defined_symbol"),
        _tuple(d, "aliases"),
        converter,
        definitions.UnitsContainer(scaled),
        value_text=value,
    )


def dimension(name: str, d: dict[str, Any], config: ParserConfig):
    value = d["value"]
    return definitions.DerivedDimensionDefinition(
        f"[{name}]", config.to_dimension_container(value), value_text=value
    )


def defaults(d: dict[str, Any], config: ParserConfig):
    unknown = set(d) - {"group", "system"}
    if unknown:
        raise DefinitionSyntaxError(
            f"Invalid keys in [defaults]: {', '.join(sorted(unknown))}"
        )
    return definitions.DefaultsDefinition(d.get("group"), d.get("system"))


def group_(d: dict[str, Any], config: ParserConfig):
    return group.GroupDefinition(
        d["name"],
        _tuple(d, "using_group_names"),
        tuple(unit(k, v, config) for k, v in d.get("definitions", {}).items()),
    )


def system_(d: dict[str, Any], config: ParserConfig):
    rules = []
    for rule in d.get("rules", ()):
        parts = [p.strip() for p in rule.split(":")]
        if len(parts) > 2:
            raise DefinitionSyntaxError(
                f"Exactly two terms expected for rule, not {len(parts)} (`{rule}`)"
            )
        rules.append(system.BaseUnitRule(*parts))
    return system.SystemDefinition(
        d["name"], _tuple(d, "using_group_names"), tuple(rules)
    )


def _relation(s: str, config: ParserConfig):
    if ":" not in s:
        raise DefinitionSyntaxError(f"Missing ':' in context relation `{s}`")
    rel, eq = s.split(":", 1)
    if "<->" in rel:
        cls, separator = context.BidirectionalRelation, "<->"
    elif "->" in rel:
        cls, separator = context.ForwardRelation, "->"
    else:
        raise DefinitionSyntaxError(f"Missing '->' or '<->' in context relation `{s}`")
    src, dst = (config.to_dimension_container(p) for p in rel.split(separator))
    return cls(src, dst, eq.strip(), value_text=s)


def context_(d: dict[str, Any], config: ParserConfig):
    return context.ContextDefinition(
        d["name"],
        _tuple(d, "aliases"),
        {k: config.to_number(str(v)) for k, v in d.get("defaults", {}).items()},
        tuple(_relation(r, config) for r in d.get("relations", ())),
        tuple(unit(k, v, config) for k, v in d.get("redefinitions", {}).items()),
    )
