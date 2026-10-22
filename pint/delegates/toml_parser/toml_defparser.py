"""
pint.delegates.toml_parser.toml_defparser
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parser for TOML definition files.

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
import tomllib
from collections.abc import Generator
from typing import Any

from ...errors import DefinitionSyntaxError, PintError
from ...facets.plain.definitions import NotNumeric
from ..base_defparser import ParserConfig
from . import plain

#: Definitions are added in this order, so groups exist before systems use them.
_NAMED_TABLES = {
    "prefix": plain.prefix,
    "dimension": plain.dimension,
    "unit": plain.unit,
}
_ARRAYS_OF_TABLES = {
    "group": plain.group_,
    "system": plain.system_,
    "context": plain.context_,
}


def _section(parsed_project: dict[str, Any], key: str, kind: type):
    value = parsed_project.get(key, kind())
    if not isinstance(value, kind):
        expected = "a table" if kind is dict else "an array of tables"
        raise DefinitionSyntaxError(f"[{key}] must be {expected}")
    return value


def _build(where: str, build, d):
    if not isinstance(d, dict):
        raise DefinitionSyntaxError(f"{where} must be a table, not {d!r}")
    try:
        return build(d)
    except (PintError, NotNumeric, ValueError, TypeError, KeyError) as ex:
        raise DefinitionSyntaxError(f"In {where}: {ex}") from ex


class TomlParser:
    def __init__(self, default_config: ParserConfig):
        self._default_config = default_config

    def iter_parsed_project(
        self, parsed_project: dict[str, Any], cfg: ParserConfig | None = None
    ) -> Generator[Any, None, None]:
        cfg = cfg or self._default_config

        unknown = set(parsed_project) - {"defaults", *_NAMED_TABLES, *_ARRAYS_OF_TABLES}
        if unknown:
            raise DefinitionSyntaxError(
                f"Unknown sections in TOML definitions: {', '.join(sorted(unknown))}"
            )

        if "defaults" in parsed_project:
            yield _build(
                "[defaults]",
                lambda d: plain.defaults(d, cfg),
                parsed_project["defaults"],
            )
        for key, build in _NAMED_TABLES.items():
            for name, d in _section(parsed_project, key, dict).items():
                yield _build(f"[{key}] {name}", lambda d: build(name, d, cfg), d)
        for key, build in _ARRAYS_OF_TABLES.items():
            for i, d in enumerate(_section(parsed_project, key, list)):
                name = d.get("name", i) if isinstance(d, dict) else i
                yield _build(f"[[{key}]] {name}", lambda d: build(d, cfg), d)

    def parse_file(self, filename: pathlib.Path | str) -> dict[str, Any]:
        with open(filename, "rb") as f:
            return tomllib.load(f)

    def parse_string(self, content: str) -> dict[str, Any]:
        return tomllib.loads(content)
