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

    [alias]
    # extra aliases for units defined elsewhere, e.g. prefixed units
    kilometer = { aliases = ["kilometro"] }

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


def _check_keys(d: dict[str, Any], *allowed: str) -> None:
    unknown = set(d) - set(allowed)
    if unknown:
        raise DefinitionSyntaxError(
            f"unknown keys {', '.join(sorted(unknown))}; expected {', '.join(allowed)}"
        )


def _tuple(d: dict[str, Any], key: str) -> tuple[str, ...]:
    value = d.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise DefinitionSyntaxError(f"'{key}' must be a list of strings, not {value!r}")
    return tuple(value)


def _str(d: dict[str, Any], key: str) -> str:
    if key not in d:
        raise DefinitionSyntaxError(f"missing '{key}'")
    return _optional_str(d, key)


def _optional_str(d: dict[str, Any], key: str) -> str | None:
    value = d.get(key)
    if value is not None and not isinstance(value, str):
        raise DefinitionSyntaxError(f"'{key}' must be a string, not {value!r}")
    return value


def _table(d: dict[str, Any], key: str) -> dict[str, Any]:
    value = d.get(key, {})
    if not isinstance(value, dict):
        raise DefinitionSyntaxError(f"'{key}' must be a table, not {value!r}")
    return value


def prefix(name: str, d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "value", "defined_symbol", "aliases")
    value = _str(d, "value")
    try:
        value_number = config.to_number(value)
    except definitions.NotNumeric as ex:
        raise DefinitionSyntaxError(
            f"Prefix definition ('{name}') must contain only numbers, not {ex.value}"
        )
    return definitions.PrefixDefinition(
        name,
        value_number,
        _optional_str(d, "defined_symbol"),
        _tuple(d, "aliases"),
        value_text=value,
    )


def unit(name: str, d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "value", "defined_symbol", "aliases")
    value = _str(d, "value")
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
        _optional_str(d, "defined_symbol"),
        _tuple(d, "aliases"),
        converter,
        definitions.UnitsContainer(scaled),
        value_text=value,
    )


def dimension(name: str, d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "value")
    value = _str(d, "value")
    return definitions.DerivedDimensionDefinition(
        f"[{name}]", config.to_dimension_container(value), value_text=value
    )


def alias(name: str, d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "aliases")
    return definitions.AliasDefinition(name, _tuple(d, "aliases"))


def defaults(d: dict[str, Any], config: ParserConfig):
    unknown = set(d) - {"group", "system"}
    if unknown:
        raise DefinitionSyntaxError(
            f"Invalid keys in [defaults]: {', '.join(sorted(unknown))}"
        )
    return definitions.DefaultsDefinition(
        _optional_str(d, "group"), _optional_str(d, "system")
    )


def group_(d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "name", "using_group_names", "definitions")
    return group.GroupDefinition(
        _str(d, "name"),
        _tuple(d, "using_group_names"),
        tuple(unit(k, v, config) for k, v in _table(d, "definitions").items()),
    )


def system_(d: dict[str, Any], config: ParserConfig):
    _check_keys(d, "name", "using_group_names", "rules")
    rules = []
    for rule in _tuple(d, "rules"):
        parts = [p.strip() for p in rule.split(":")]
        if len(parts) > 2:
            raise DefinitionSyntaxError(
                f"Exactly two terms expected for rule, not {len(parts)} (`{rule}`)"
            )
        rules.append(system.BaseUnitRule(*parts))
    return system.SystemDefinition(
        _str(d, "name"), _tuple(d, "using_group_names"), tuple(rules)
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
    _check_keys(d, "name", "aliases", "defaults", "relations", "redefinitions")
    return context.ContextDefinition(
        _str(d, "name"),
        _tuple(d, "aliases"),
        {k: config.to_number(str(v)) for k, v in _table(d, "defaults").items()},
        tuple(_relation(r, config) for r in _tuple(d, "relations")),
        tuple(unit(k, v, config) for k, v in _table(d, "redefinitions").items()),
    )
