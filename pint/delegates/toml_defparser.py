"""
pint.delegates.toml_defparser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parser for TOML definition files, e.g. ``default_en.toml``::

    version = "1.0"

    [defaults]
    group = "international"
    system = "mks"

    [prefix]
    kilo = { value = 1e3, symbol = "k" }

    [dimension]
    density = { value = "[mass] / [volume]" }

    [unit]
    meter = { value = "[length]", symbol = "m", aliases = ["metre"] }
    degree_Celsius = { value = "kelvin", offset = 273.15, symbol = "°C" }
    decibel = { value = "1", logbase = 10, logfactor = 10, symbol = "dB" }

    [[group]]
    name = "Avoirdupois"
    [group.definitions]
    pound = { value = "7000 * grain", symbol = "lb" }

    [[system]]
    name = "mks"
    using_group_names = ["international"]
    rules = ["meter", "kilogram", "second"]

    [[context]]
    name = "spectroscopy"
    aliases = ["sp"]
    defaults = { n = 1 }
    relations = ["[length] <-> [frequency]: speed_of_light / n / value"]

Values use the same syntax as the text format. Numbers (prefix values and the
``offset``, ``logbase`` and ``logfactor`` of a unit) may be TOML numbers or
strings such as ``"2**10"``; both are read with the registry's number type.
``version`` is the version of this format; a file for another major version is
rejected.

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

from ..converters import Converter
from ..errors import DefinitionSyntaxError, PintError
from ..facets.context.definitions import (
    BidirectionalRelation,
    ContextDefinition,
    ForwardRelation,
)
from ..facets.group.definitions import GroupDefinition
from ..facets.plain import definitions
from ..facets.system.definitions import BaseUnitRule, SystemDefinition
from .base_defparser import ParserConfig


def _number(value: Any, cfg: ParserConfig):
    # floats arrive as their source text (parse_float=str), keeping all digits
    return cfg.to_number(str(value))


def _prefix(name: str, d: dict[str, Any], cfg: ParserConfig):
    return definitions.PrefixDefinition(
        name,
        _number(d["value"], cfg),
        d.get("symbol"),
        tuple(d.get("aliases", ())),
    )


def _unit(name: str, d: dict[str, Any], cfg: ParserConfig):
    modifiers = {
        key: _number(d[key], cfg)
        for key in ("offset", "logbase", "logfactor")
        if key in d
    }
    scaled = cfg.to_scaled_units_container(str(d["value"]))
    return definitions.UnitDefinition(
        name,
        d.get("symbol"),
        tuple(d.get("aliases", ())),
        Converter.from_arguments(scale=scaled.scale, **modifiers),
        definitions.UnitsContainer(scaled),
    )


def _dimension(name: str, d: dict[str, Any], cfg: ParserConfig):
    return definitions.DerivedDimensionDefinition(
        f"[{name}]", cfg.to_dimension_container(d["value"])
    )


def _group(d: dict[str, Any], cfg: ParserConfig):
    return GroupDefinition(
        d["name"],
        tuple(d.get("using_group_names", ())),
        tuple(_unit(k, v, cfg) for k, v in d.get("definitions", {}).items()),
    )


def _system(d: dict[str, Any], cfg: ParserConfig):
    rules = tuple(
        BaseUnitRule(*(p.strip() for p in rule.split(":")))
        for rule in d.get("rules", ())
    )
    return SystemDefinition(d["name"], tuple(d.get("using_group_names", ())), rules)


def _relation(s: str, cfg: ParserConfig):
    rel, equation = s.split(":", 1)
    cls, separator = (
        (BidirectionalRelation, "<->") if "<->" in rel else (ForwardRelation, "->")
    )
    src, dst = (cfg.to_dimension_container(p) for p in rel.split(separator))
    return cls(src, dst, equation.strip())


def _context(d: dict[str, Any], cfg: ParserConfig):
    return ContextDefinition(
        d["name"],
        tuple(d.get("aliases", ())),
        {k: _number(v, cfg) for k, v in d.get("defaults", {}).items()},
        tuple(_relation(r, cfg) for r in d.get("relations", ())),
        tuple(_unit(k, v, cfg) for k, v in d.get("redefinitions", {}).items()),
    )


#: Version of the TOML definitions format read by this parser.
FORMAT_VERSION = "1.0"

_ERRORS = (PintError, definitions.NotNumeric, ValueError, TypeError, KeyError)


def _build(where: str, build, *args):
    try:
        return build(*args)
    except _ERRORS as ex:
        raise DefinitionSyntaxError(f"in {where}: {ex}") from ex


def _defaults(data: dict[str, Any]) -> list:
    if "defaults" not in data:
        return []
    d = data["defaults"]
    return [definitions.DefaultsDefinition(d.get("group"), d.get("system"))]


def _prefixes(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[prefix] {name}", _prefix, name, d, cfg)
        for name, d in data.get("prefix", {}).items()
    ]


def _dimensions(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[dimension] {name}", _dimension, name, d, cfg)
        for name, d in data.get("dimension", {}).items()
    ]


def _units(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[unit] {name}", _unit, name, d, cfg)
        for name, d in data.get("unit", {}).items()
    ]


def _groups(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[[group]] {d.get('name')}", _group, d, cfg)
        for d in data.get("group", [])
    ]


def _systems(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[[system]] {d.get('name')}", _system, d, cfg)
        for d in data.get("system", [])
    ]


def _contexts(data: dict[str, Any], cfg: ParserConfig) -> list:
    return [
        _build(f"[[context]] {d.get('name')}", _context, d, cfg)
        for d in data.get("context", [])
    ]


def _check_version(data: dict[str, Any]) -> None:
    required = f'version = "{FORMAT_VERSION}"'
    major = FORMAT_VERSION.split(".")[0]
    version = data.get("version")
    if version is None:
        raise DefinitionSyntaxError(
            f"missing the definitions format version; add {required} "
            "at the top of the file, before any [section]"
        )
    if str(version).split(".")[0] != major:
        raise DefinitionSyntaxError(
            f'unsupported definitions format version "{version}"; this version '
            f"of Pint reads version {major}.x files, e.g. {required}"
        )


_KEYS = {
    "version",
    "defaults",
    "prefix",
    "dimension",
    "unit",
    "group",
    "system",
    "context",
}


class TomlParser:
    def __init__(self, default_config: ParserConfig):
        self._default_config = default_config

    def parse_file(self, filename: pathlib.Path | str) -> list:
        """Return the definitions in a TOML file, in the order they are added.

        Groups come before systems, which use them.
        """
        with open(filename, "rb") as f:
            data = tomllib.load(f, parse_float=str)
        cfg = self._default_config
        try:
            _check_version(data)
            unknown = set(data) - _KEYS
            if unknown:
                raise DefinitionSyntaxError(
                    f"unknown sections {', '.join(sorted(unknown))}"
                )
            return [
                *_defaults(data),
                *_prefixes(data, cfg),
                *_dimensions(data, cfg),
                *_units(data, cfg),
                *_groups(data, cfg),
                *_systems(data, cfg),
                *_contexts(data, cfg),
            ]
        except DefinitionSyntaxError as ex:
            raise DefinitionSyntaxError(f"{filename}: {ex}") from ex
