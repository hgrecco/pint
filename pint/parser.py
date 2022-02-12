"""
    pint.parser
    ~~~~~~~~~~~

    Classes and methods to parse a definition text file into a DefinitionFile.

    :copyright: 2019 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from io import StringIO
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

from . import diskcache
from .definitions import Definition
from .errors import DefinitionSyntaxError
from .util import SourceIterator, logger

_BLOCK_RE = re.compile(r"[ (]")

ParserFuncT = Callable[[SourceIterator, type], Any]


@dataclass(frozen=True)
class DefinitionFile:
    """Represents a definition file after parsing."""

    # Fullpath of the original file, None if a text was provided
    filename: Optional[pathlib.Path]
    is_resource: bool

    # Modification time of the file or None.
    mtime: Optional[float]

    # collection of line number and corresponding definition.
    parsed_lines: Tuple[Tuple[int, Any], ...]

    def filter_by(self, *klass):
        yield from (
            (lineno, d) for lineno, d in self.parsed_lines if isinstance(d, klass)
        )

    @cached_property
    def errors(self):
        return tuple(self.filter_by(Exception))

    def has_errors(self):
        return bool(self.errors)


@dataclass(frozen=True)
class ImportDefinition:
    """Definition for the @import directive"""

    path: str

    @classmethod
    def from_string(
        cls, definition: str, non_int_type: type = float
    ) -> ImportDefinition:
        return ImportDefinition(definition[7:].strip())


class Parser:
    """Class to parse a definition file into an intermediate object representation.

    non_int_type
        numerical type used for non integer values. (Default: float)
    raise_on_error
        if True, an exception will be raised as soon as a Definition Error it is found.
        if False, the exception will be added to the ParedDefinitionFile
    """

    #: Map context prefix to function
    _directives: Dict[str, ParserFuncT]

    handled_classes = (ImportDefinition,)

    def __init__(self, non_int_type=float, raise_on_error=True, use_cache=True):
        self._directives = {}
        self._non_int_type = non_int_type
        self._raise_on_error = raise_on_error
        self.register_class("@import", ImportDefinition)
        self._use_cache = use_cache

    def register_directive(
        self, prefix: str, parserfunc: ParserFuncT, single_line: bool
    ):
        """Register a parser for a given @ directive..

        Parameters
        ----------
        prefix
            string identifying the section (e.g. @context)
        parserfunc
            function that is able to parse a definition into a DefinitionObject
        single_line
            indicates that the directive spans in a single line, i.e. and @end is not required.
        """
        if prefix and prefix[0] == "@":
            if single_line:
                self._directives[prefix] = lambda si, non_int_type: parserfunc(
                    si.last[1], non_int_type
                )
            else:
                self._directives[prefix] = lambda si, non_int_type: parserfunc(
                    si.block_iter(), non_int_type
                )
        else:
            raise ValueError("Prefix directives must start with '@'")

    def register_class(self, prefix: str, klass):
        """Register a definition class for a directive and try to guess
        if it is a line or block directive from the signature.
        """
        if hasattr(klass, "from_string"):
            self.register_directive(prefix, klass.from_string, True)
        elif hasattr(klass, "from_lines"):
            self.register_directive(prefix, klass.from_lines, False)
        else:
            raise ValueError(
                f"While registering {prefix}, {klass} does not have `from_string` or from_lines` method"
            )

    def parse(self, file, is_resource: bool = False) -> Tuple[DefinitionFile, ...]:
        """Parse a file or resource into a collection of DefinitionFile.

        Parameters
        ----------
        file
            definitions or file containing definition.
        is_resource
            indicates that the file is a resource file
            and therefore should be loaded from the package.
            (Default value = False)
        """

        if is_resource:
            parsed = self.parse_single_resource(file)
        else:
            path = pathlib.Path(file)
            if self._use_cache:
                parsed = self.parse_single_cache(path)
            else:
                parsed = self.parse_single(path)

        out = [parsed]
        for lineno, content in parsed.filter_by(ImportDefinition):
            if parsed.is_resource:
                path = content.path
            else:
                try:
                    basedir = parsed.filename.parent
                except AttributeError:
                    basedir = pathlib.Path.cwd()
                path = basedir.joinpath(content.path)
            out.extend(self.parse(path, parsed.is_resource))
        return tuple(out)

    def parse_single_resource(self, resource_name: str) -> DefinitionFile:
        """Parse a resource in the package into a DefinitionFile.

        This method will try to load it first as a regular file
        (with a path and mtime) to allow caching.
        If this files (i.e. the resource is not filesystem file)
        it will use python importlib.resources.read_binary
        """

        with resources.path(__package__, resource_name) as p:
            filepath = p.resolve()

        # This is the only way I could come up to decide if the
        # resource is a real file in the filesystem.
        # It will not allow me to cache a resource inside a Zip file
        # (which is logical fine)
        if filepath.exists():
            return self.parse_single(filepath)

        logger.debug("Cannot use_cache resource without a real path")
        return self._parse_single_resource(resource_name)

    def _parse_single_resource(self, resource_name: str) -> DefinitionFile:
        rbytes = resources.read_binary(__package__, resource_name)
        si = SourceIterator(
            StringIO(rbytes.decode("utf-8")), resource_name, is_resource=True
        )
        parsed_lines = tuple(self.yield_from_source_iterator(si))
        return DefinitionFile(pathlib.Path(resource_name), True, None, parsed_lines)

    def parse_single(self, filepath: pathlib.Path) -> DefinitionFile:
        """Parse a filepath without nesting into dependent files.

        Parameters
        ----------
        filepath
            definitions or file containing definition.
        """
        with filepath.open(encoding="utf-8") as fp:
            si = SourceIterator(fp, filepath, is_resource=False)
            parsed_lines = tuple(self.yield_from_source_iterator(si))

        filename = filepath.resolve()
        mtime = filepath.stat().st_mtime

        return DefinitionFile(filename, False, mtime, parsed_lines)

    def parse_single_cache(self, filepath: pathlib.Path) -> DefinitionFile:
        """Parse a filepath into a DefinitionFile object
        without importing dependent files.

        A cached version is used if available.

        Parameters
        ----------
        filepath
            definitions or file containing definition.
        """
        content = diskcache.load(filepath)
        if content:
            return content
        content = self.parse_single(filepath)
        diskcache.save(content, filepath)
        return content

    def parse_lines(self, lines: Iterable[str]) -> DefinitionFile:
        """Parse an iterable of strings into a dependent file"""
        si = SourceIterator(lines, None, False)
        parsed_lines = tuple(self.yield_from_source_iterator(si))
        df = DefinitionFile(None, False, None, parsed_lines=parsed_lines)
        if any(df.filter_by(ImportDefinition)):
            raise ValueError(
                "Cannot use the @import directive when parsing "
                "an iterable of strings."
            )
        return df

    def yield_from_source_iterator(
        self, source_iterator: SourceIterator
    ) -> Generator[Tuple[int, Any]]:
        """Iterates through the source iterator, yields line numbers and
        the coresponding parsed definition object.

        Parameters
        ----------
        source_iterator
        """
        for lineno, line in source_iterator:
            try:
                if line.startswith("@"):
                    # Handle @ directives dispatching to the appropriate parsers
                    parts = _BLOCK_RE.split(line)

                    subparser = self._directives.get(parts[0], None)

                    if subparser is None:
                        raise DefinitionSyntaxError(
                            "Unknown directive %s" % line, lineno=lineno
                        )

                    d = subparser(source_iterator, self._non_int_type)
                    yield lineno, d
                else:
                    yield lineno, Definition.from_string(line, self._non_int_type)
            except DefinitionSyntaxError as ex:
                if ex.lineno is None:
                    ex.lineno = lineno
                if self._raise_on_error:
                    raise ex
                yield lineno, ex
            except Exception as ex:
                logger.error("In line {}, cannot add '{}' {}".format(lineno, line, ex))
                raise ex
