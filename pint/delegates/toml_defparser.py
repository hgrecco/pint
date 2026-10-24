"""
pint.delegates.toml_defparser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parser for TOML definition files, e.g. ``default_en.toml``::

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

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
import tomllib
from collections.abc import Iterator
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


#: Sections in the order they are added, so groups exist before systems use
#: them. Tables map names to definitions; arrays hold named tables.
_SECTIONS = {
    "prefix": (dict, _prefix),
    "dimension": (dict, _dimension),
    "unit": (dict, _unit),
    "group": (list, _group),
    "system": (list, _system),
    "context": (list, _context),
}


class TomlParser:
    def __init__(self, default_config: ParserConfig):
        self._default_config = default_config

    def parse_file(self, filename: pathlib.Path | str) -> Iterator[Any]:
        """Yield the definitions in a TOML file."""
        with open(filename, "rb") as f:
            data = tomllib.load(f, parse_float=str)
        cfg = self._default_config

        unknown = set(data) - {"defaults", *_SECTIONS}
        if unknown:
            raise DefinitionSyntaxError(
                f"{filename}: unknown sections {', '.join(sorted(unknown))}"
            )
        if "defaults" in data:
            yield definitions.DefaultsDefinition(
                data["defaults"].get("group"), data["defaults"].get("system")
            )
        for key, (kind, build) in _SECTIONS.items():
            if kind is dict:
                items = [(f"[{key}] {n}", n, d) for n, d in data.get(key, {}).items()]
            else:
                items = [
                    (f"[[{key}]] {d.get('name')}", None, d) for d in data.get(key, [])
                ]
            for where, name, d in items:
                try:
                    yield build(d, cfg) if name is None else build(name, d, cfg)
                except (
                    PintError,
                    definitions.NotNumeric,
                    ValueError,
                    TypeError,
                    KeyError,
                ) as ex:
                    raise DefinitionSyntaxError(f"{filename}: in {where}: {ex}") from ex
