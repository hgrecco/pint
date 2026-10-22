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

from ...errors import DefinitionSyntaxError
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
            yield plain.defaults(parsed_project["defaults"], cfg)
        for key, build in _NAMED_TABLES.items():
            for name, d in parsed_project.get(key, {}).items():
                yield build(name, d, cfg)
        for key, build in _ARRAYS_OF_TABLES.items():
            for d in parsed_project.get(key, ()):
                yield build(d, cfg)

    def parse_file(self, filename: pathlib.Path | str) -> dict[str, Any]:
        with open(filename, "rb") as f:
            return tomllib.load(f)

    def parse_string(self, content: str) -> dict[str, Any]:
        return tomllib.loads(content)
