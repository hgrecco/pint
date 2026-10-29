"""
pint.delegates.toml_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~

Writer for TOML definition files, read back by
:class:`~pint.delegates.toml_defparser.TomlParser`.

Only defaults, prefixes, derived dimensions and units are written. Values are
written from their source text (``value_text``), or rebuilt for definitions
made in code, e.g. ``5 / 9 * kelvin`` becomes ``0.5555555555555556 * kelvin``.

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import tomllib
from collections.abc import Iterable
from typing import Any

from ..facets.plain import definitions
from .toml_defparser import FORMAT_VERSION

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


class _Number(str):
    """Source text of a number, e.g. ``1e3`` or ``2**10``."""


class TomlWriter:
    """Writer for TOML definition files.

    ``dump_*`` methods write a whole section of the file, and ``format_*``
    methods write one entry of a section.
    """

    def write_file(self, filename: pathlib.Path | str, defs: Iterable) -> None:
        """Write definitions to a TOML file.

        Parameters
        ----------
        filename :
            path of the TOML definitions file.
        defs :
            DefaultsDefinition, PrefixDefinition, DerivedDimensionDefinition
            and UnitDefinition objects, e.g. from ``TomlParser.parse_file``.

        Raises
        ------
        TypeError
            if a definition of another kind is given.
        """
        pathlib.Path(filename).write_text(self.dumps(defs), encoding="utf-8")

    def dumps(self, defs: Iterable) -> str:
        """Return the TOML text for definitions, see :meth:`write_file`."""
        sections: dict[type, list] = {
            definitions.DefaultsDefinition: [],
            definitions.PrefixDefinition: [],
            definitions.DerivedDimensionDefinition: [],
            definitions.UnitDefinition: [],
        }
        for d in defs:
            # subclasses too, e.g. the definitions of the text parser
            kind = next((k for k in sections if isinstance(d, k)), None)
            if kind is None:
                raise TypeError(f"cannot write {type(d).__name__} to TOML")
            sections[kind].append(d)

        lines = [f'version = "{FORMAT_VERSION}"']
        lines += self.dump_defaults(sections[definitions.DefaultsDefinition])
        lines += self.dump_prefixes(sections[definitions.PrefixDefinition])
        lines += self.dump_dimensions(sections[definitions.DerivedDimensionDefinition])
        lines += self.dump_units(sections[definitions.UnitDefinition])
        return "\n".join(lines) + "\n"

    # Sections

    def dump_defaults(self, defs: list) -> list[str]:
        if not defs:
            return []
        d = defs[-1]
        values = {"group": d.group, "system": d.system}
        return ["", "[defaults]"] + [
            f"{key} = {self._string(value)}" for key, value in values.items() if value
        ]

    def dump_prefixes(self, defs: list) -> list[str]:
        return self._section("prefix", [self.format_prefix(d) for d in defs])

    def dump_dimensions(self, defs: list) -> list[str]:
        return self._section("dimension", [self.format_dimension(d) for d in defs])

    def dump_units(self, defs: list) -> list[str]:
        return self._section("unit", [self.format_unit(d) for d in defs])

    # Entries

    def format_prefix(self, d) -> str:
        value = _Number(d.value_text) if d.value_text else d.value
        return self._entry(d.name, {"value": value, **self._names(d)})

    def format_dimension(self, d) -> str:
        return self._entry(d.name[1:-1], {"value": d.value_text or str(d.reference)})

    def format_unit(self, d) -> str:
        converter = d.converter
        reference = str(d.reference) if d.reference else ""
        if d.value_text:
            value = d.value_text
        elif d.is_base or converter.scale == 1:
            value = reference or "1"
        else:
            value = " * ".join(filter(None, (str(converter.scale), reference)))
        # offset, logbase and logfactor of non-multiplicative units
        modifiers = {
            f.name: getattr(converter, f.name)
            for f in dataclasses.fields(converter)
            if f.name != "scale"
        }
        return self._entry(d.name, {"value": value, **modifiers, **self._names(d)})

    # Formatting

    @staticmethod
    def _names(d) -> dict[str, Any]:
        names: dict[str, Any] = {}
        if d.defined_symbol:
            names["symbol"] = d.defined_symbol
        if d.aliases:
            names["aliases"] = list(d.aliases)
        return names

    @staticmethod
    def _section(name: str, entries: list[str]) -> list[str]:
        return ["", f"[{name}]", *entries] if entries else []

    def _entry(self, name: str, values: dict[str, Any]) -> str:
        pairs = ", ".join(f"{key} = {self._value(v)}" for key, v in values.items())
        return f"{self._key(name)} = {{ {pairs} }}"

    def _key(self, name: str) -> str:
        return name if _BARE_KEY.fullmatch(name) else self._string(name)

    @staticmethod
    def _string(text: str) -> str:
        # JSON strings are valid TOML basic strings
        return json.dumps(text, ensure_ascii=False)

    def _value(self, value: Any) -> str:
        if isinstance(value, _Number):
            try:
                tomllib.loads(f"v = {value}", parse_float=str)
                return value
            except tomllib.TOMLDecodeError:
                return self._string(value)
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, list):
            return "[" + ", ".join(map(self._value, value)) + "]"
        if isinstance(value, bool):
            raise TypeError(f"cannot write {value!r} to TOML")
        if isinstance(value, (int, float)):
            return repr(value)
        # other number types, e.g. Decimal, are read back with to_number
        return self._string(str(value))
