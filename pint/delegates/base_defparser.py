"""
    pint.delegates.base_defparser
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Common class and function for all parsers.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import functools
import itertools
import numbers
import pathlib
import typing as ty
from dataclasses import dataclass, field

from pint import errors
from pint.facets.plain.definitions import NotNumeric
from pint.util import ParserHelper, UnitsContainer

from .._vendor import flexcache as fc
from .._vendor import flexparser as fp


@dataclass(frozen=True)
class ParserConfig:
    """Configuration used by the parser."""

    #: Indicates the output type of non integer numbers.
    non_int_type: ty.Type[numbers.Number] = float

    def to_scaled_units_container(self, s: str):
        return ParserHelper.from_string(s, self.non_int_type)

    def to_units_container(self, s: str):
        v = self.to_scaled_units_container(s)
        if v.scale != 1:
            raise errors.UnexpectedScaleInContainer(str(v.scale))
        return UnitsContainer(v)

    def to_dimension_container(self, s: str):
        v = self.to_units_container(s)
        invalid = tuple(itertools.filterfalse(errors.is_valid_dimension_name, v.keys()))
        if invalid:
            raise errors.DefinitionSyntaxError(
                f"Cannot build a dimension container with {', '.join(invalid)} that "
                + errors.MSG_INVALID_DIMENSION_NAME
            )
        return v

    def to_number(self, s: str) -> numbers.Number:
        """Try parse a string into a number (without using eval).

        The string can contain a number or a simple equation (3 + 4)

        Raises
        ------
        _NotNumeric
            If the string cannot be parsed as a number.
        """
        val = self.to_scaled_units_container(s)
        if len(val):
            raise NotNumeric(s)
        return val.scale


@functools.lru_cache()
def build_disk_cache_class(non_int_type: type):
    """Build disk cache class, taking into account the non_int_type."""

    @dataclass(frozen=True)
    class PintHeader(fc.InvalidateByExist, fc.NameByFields, fc.BasicPythonHeader):

        from .. import __version__

        pint_version: str = __version__
        non_int_type: str = field(default_factory=lambda: non_int_type.__qualname__)

    class PathHeader(fc.NameByFileContent, PintHeader):
        pass

    class ParsedProjecHeader(fc.NameByHashIter, PintHeader):
        @classmethod
        def from_parsed_project(cls, pp: fp.ParsedProject, reader_id):
            tmp = []
            for stmt in pp.iter_statements():
                if isinstance(stmt, fp.BOS):
                    tmp.append(
                        stmt.content_hash.algorithm_name
                        + ":"
                        + stmt.content_hash.hexdigest
                    )

            return cls(tuple(tmp), reader_id)

    class PintDiskCache(fc.DiskCache):

        _header_classes = {
            pathlib.Path: PathHeader,
            str: PathHeader.from_string,
            fp.ParsedProject: ParsedProjecHeader.from_parsed_project,
        }

    return PintDiskCache
