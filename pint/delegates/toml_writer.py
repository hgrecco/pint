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
import pathlib
from collections.abc import Iterable
from typing import Any

from ..facets.plain import definitions
from .toml_defparser import FORMAT_VERSION

try:
    import tomlkit
except ImportError:  # pragma: no cover
    tomlkit = None


class TomlWriter:
    """Writer for TOML definition files. Requires tomlkit.

    ``dump_*`` methods write a whole section of the file, and ``format_*``
    methods write one entry of a section.
    """

    def __init__(self):
        if tomlkit is None:
            raise ImportError(
                "tomlkit is required to write TOML definitions; "
                "install it with pip install pint[toml]"
            )

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

        doc = tomlkit.document()
        doc.add("version", FORMAT_VERSION)
        for name, table in (
            ("defaults", self.dump_defaults(sections[definitions.DefaultsDefinition])),
            ("prefix", self.dump_prefixes(sections[definitions.PrefixDefinition])),
            (
                "dimension",
                self.dump_dimensions(sections[definitions.DerivedDimensionDefinition]),
            ),
            ("unit", self.dump_units(sections[definitions.UnitDefinition])),
        ):
            if table:
                doc.add(name, table)
        return tomlkit.dumps(doc)

    # Sections

    def dump_defaults(self, defs: list):
        table = tomlkit.table()
        if defs:
            d = defs[-1]
            for key, value in d.items():
                table.add(key, value)
        return table

    def dump_prefixes(self, defs: list):
        return self._build_section_table({d.name: self.format_prefix(d) for d in defs})

    def dump_dimensions(self, defs: list):
        return self._build_section_table(
            {d.name[1:-1]: self.format_dimension(d) for d in defs}
        )

    def dump_units(self, defs: list):
        return self._build_section_table({d.name: self.format_unit(d) for d in defs})

    # Entries

    def format_prefix(self, d):
        value = self._build_number(d.value_text) if d.value_text else d.value
        return self._build_inline_table(
            {"value": value, **self._symbol_and_aliases_dict(d)}
        )

    def format_dimension(self, d):
        return self._build_inline_table({"value": d.value_text or str(d.reference)})

    def format_unit(self, d):
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
        return self._build_inline_table(
            {"value": value, **modifiers, **self._symbol_and_aliases_dict(d)}
        )

    @staticmethod
    def _symbol_and_aliases_dict(d) -> dict[str, Any]:
        names: dict[str, Any] = {}
        if d.defined_symbol:
            names["symbol"] = d.defined_symbol
        if d.aliases:
            names["aliases"] = list(d.aliases)
        return names

    # Building TOML items

    @staticmethod
    def _build_section_table(entries: dict[str, Any]):
        table = tomlkit.table()
        for name, entry in entries.items():
            table.add(name, entry)
        return table

    def _build_inline_table(self, values: dict[str, Any]):
        entry = tomlkit.inline_table()
        for key, value in values.items():
            entry.add(key, self._build_toml_value(value))
        return entry

    @staticmethod
    def _build_number(text: str):
        """The TOML representation of a number. Returns an int, float or string.

        Uses a string only when the text is not a TOML number, e.g. ``2**10``.
        """
        try:
            item = tomlkit.value(text)
        except tomlkit.exceptions.ParseError:
            return text
        return item if isinstance(item, (int, float)) else text

    @staticmethod
    def _build_toml_value(value: Any):
        if isinstance(value, (str, list, int, float)):
            return value
        # other number types, e.g. Decimal, are read back with to_number
        return str(value)
