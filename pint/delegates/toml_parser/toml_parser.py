from __future__ import annotations

import copy
import pathlib

import flexcache as fc
import tomllib

from ..base_defparser import ParserConfig
from . import plain


class TomlParser:
    def __init__(self, default_config: ParserConfig, diskcache: fc.DiskCache):
        self._default_config = default_config
        self._diskcache = diskcache

    def iter_parsed_project(self, parsed_project: dict):
        stmts = {
            "unit": plain.UnitDefinition,
            "prefix": plain.PrefixDefinition,
            "dimension": plain.DerivedDimensionDefinition,
            "system": plain.SystemDefinition,
            "context": plain.ContextDefinition,
            "group": plain.GroupDefinition,
        }
        for definition_type in parsed_project.keys():
            for key, value in parsed_project[definition_type].items():
                d = copy.copy(value)
                d["name"] = key
                stmt = stmts[definition_type].from_dict_and_config(
                    d, self._default_config
                )
                yield stmt

    def parse_file(
        self, filename: pathlib.Path | str, cfg: ParserConfig | None = None
    ) -> dict:
        with open(filename, "rb") as f:
            data = tomllib.load(f)
        return data

    # def parse_string(self, content: str, cfg: ParserConfig | None = None):
    #     return fp.parse_bytes(
    #         content.encode("utf-8"),
    #         _PintParser,
    #         cfg or self._default_config,
    #         diskcache=self._diskcache,
    #         strip_spaces=True,
    #         delimiters=_PintParser._delimiters,
    #     )
