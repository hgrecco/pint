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

#: Version of the TOML definitions format read by this parser.
FORMAT_VERSION = "1.0"


_ERRORS = (PintError, definitions.NotNumeric, ValueError, TypeError, KeyError)

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
    """Parser for TOML definition files.

    ``load_*`` methods read a whole section of the file, and ``parse_*``
    methods read one entry of a section.
    """

    def __init__(self, default_config: ParserConfig):
        self._default_config = default_config

    def parse_file(self, filename: pathlib.Path | str) -> list:
        """Read the definitions in a TOML file.

        Parameters
        ----------
        filename :
            path of the TOML definitions file.

        Returns
        -------
        list
            definitions in the order they are added to the registry: defaults,
            prefixes, dimensions, units, groups, systems, then contexts.

        Raises
        ------
        DefinitionSyntaxError
            if the version is missing or unsupported, a section is unknown, or
            an entry is invalid. The message starts with the file name.
        tomllib.TOMLDecodeError
            if the file is not valid TOML.
        """
        with open(filename, "rb") as f:
            data = tomllib.load(f, parse_float=str)
        try:
            self._check_version(data)
            unknown = set(data) - _KEYS
            if unknown:
                raise DefinitionSyntaxError(
                    f"unknown sections {', '.join(sorted(unknown))}"
                )
            return [
                *self.load_defaults(data),
                *self.load_prefixes(data),
                *self.load_dimensions(data),
                *self.load_units(data),
                *self.load_groups(data),
                *self.load_systems(data),
                *self.load_contexts(data),
            ]
        except DefinitionSyntaxError as ex:
            raise DefinitionSyntaxError(f"{filename}: {ex}") from ex

    @staticmethod
    def _check_version(data: dict[str, Any]) -> None:
        version = data.get("version")
        if version is None:
            raise DefinitionSyntaxError(
                f'missing the definitions format version; add version = "{FORMAT_VERSION}" '
                "at the top of the file, before any [section]"
            )
        if str(version).split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise DefinitionSyntaxError(
                f"unsupported version {version}; this Pint reads version {FORMAT_VERSION}"
            )

    @staticmethod
    def _parse_entry(where: str, parser_function, *args):
        """Parse one entry, reporting errors as a syntax error naming the entry.

        Parameters
        ----------
        where :
            label of the entry for error messages, e.g. ``"[unit] meter"``.
        parser_function :
            entry parser, e.g. ``self.parse_unit``, that returns a definition.
        *args :
            arguments for ``parser_function``: ``(name, d)`` for prefixes,
            dimensions and units, ``(d,)`` for groups, systems and contexts.
        """
        try:
            return parser_function(*args)
        except _ERRORS as ex:
            raise DefinitionSyntaxError(f"in {where}: {ex}") from ex

    # Sections

    def load_defaults(self, data: dict[str, Any]) -> list:
        if "defaults" not in data:
            return []
        d = data["defaults"]
        return [definitions.DefaultsDefinition(d.get("group"), d.get("system"))]

    def load_prefixes(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[prefix] {name}", self.parse_prefix, name, d)
            for name, d in data.get("prefix", {}).items()
        ]

    def load_dimensions(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[dimension] {name}", self.parse_dimension, name, d)
            for name, d in data.get("dimension", {}).items()
        ]

    def load_units(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[unit] {name}", self.parse_unit, name, d)
            for name, d in data.get("unit", {}).items()
        ]

    def load_groups(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[[group]] {d.get('name')}", self.parse_group, d)
            for d in data.get("group", [])
        ]

    def load_systems(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[[system]] {d.get('name')}", self.parse_system, d)
            for d in data.get("system", [])
        ]

    def load_contexts(self, data: dict[str, Any]) -> list:
        return [
            self._parse_entry(f"[[context]] {d.get('name')}", self.parse_context, d)
            for d in data.get("context", [])
        ]

    # Entries

    def _number(self, value: Any):
        # floats arrive as their source text (parse_float=str), keeping all digits
        return self._default_config.to_number(str(value))

    def parse_prefix(self, name: str, d: dict[str, Any]):
        return definitions.PrefixDefinition(
            name,
            self._number(d["value"]),
            d.get("symbol"),
            tuple(d.get("aliases", ())),
            value_text=str(d["value"]),
        )

    def parse_dimension(self, name: str, d: dict[str, Any]):
        return definitions.DerivedDimensionDefinition(
            f"[{name}]",
            self._default_config.to_dimension_container(d["value"]),
            value_text=d["value"],
        )

    def parse_unit(self, name: str, d: dict[str, Any]):
        modifiers = {
            key: self._number(d[key])
            for key in ("offset", "logbase", "logfactor")
            if key in d
        }
        scaled = self._default_config.to_scaled_units_container(str(d["value"]))
        return definitions.UnitDefinition(
            name,
            d.get("symbol"),
            tuple(d.get("aliases", ())),
            Converter.from_arguments(scale=scaled.scale, **modifiers),
            definitions.UnitsContainer(scaled),
            value_text=str(d["value"]),
        )

    def parse_group(self, d: dict[str, Any]):
        return GroupDefinition(
            d["name"],
            tuple(d.get("using_group_names", ())),
            tuple(self.parse_unit(k, v) for k, v in d.get("definitions", {}).items()),
        )

    def parse_system(self, d: dict[str, Any]):
        rules = tuple(
            BaseUnitRule(*(p.strip() for p in rule.split(":")))
            for rule in d.get("rules", ())
        )
        return SystemDefinition(d["name"], tuple(d.get("using_group_names", ())), rules)

    def parse_relation(self, text: str):
        """Parse a context relation, e.g. ``"[length] <-> [frequency]: c / value"``."""
        dimensions, equation = text.split(":", 1)
        arrow = "<->" if "<->" in dimensions else "->"
        source, destination = dimensions.split(arrow)
        source = self._default_config.to_dimension_container(source)
        destination = self._default_config.to_dimension_container(destination)
        if arrow == "<->":
            return BidirectionalRelation(source, destination, equation.strip())
        return ForwardRelation(source, destination, equation.strip())

    def parse_context(self, d: dict[str, Any]):
        return ContextDefinition(
            d["name"],
            tuple(d.get("aliases", ())),
            {k: self._number(v) for k, v in d.get("defaults", {}).items()},
            tuple(self.parse_relation(r) for r in d.get("relations", ())),
            tuple(self.parse_unit(k, v) for k, v in d.get("redefinitions", {}).items()),
        )
