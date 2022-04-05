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
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from io import StringIO
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple

from ._vendor import flexcache as fc
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

    # SHA-1 hash
    content_hash: Optional[str]

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


class DefinitionFiles(tuple):
    """Wrapper class that allows handling a tuple containing DefinitionFile."""

    @staticmethod
    def _iter_definitions(
        pending_files: list[DefinitionFile],
    ) -> Generator[Tuple[int, Definition]]:
        """Internal method to iterate definitions.

        pending_files is a mutable list of definitions files
        and elements are being removed as they are yielded.
        """
        if not pending_files:
            return
        current_file = pending_files.pop(0)
        for lineno, definition in current_file.parsed_lines:
            if isinstance(definition, ImportDefinition):
                if not pending_files:
                    raise ValueError(
                        f"No more files while trying to import {definition.path}."
                    )

                if not str(pending_files[0].filename).endswith(definition.path):
                    raise ValueError(
                        "The order of the files do not match. "
                        f"(expected: {definition.path}, "
                        f"found {pending_files[0].filename})"
                    )

                yield from DefinitionFiles._iter_definitions(pending_files)
            else:
                yield lineno, definition

    def iter_definitions(self):
        """Iter all definitions in the order they appear,
        going into the included files.

        Important: This assumes that the order of the imported files
        is the one that they will appear in the definitions.
        """
        yield from self._iter_definitions(list(self))


def build_disk_cache_class(non_int_type: type):
    """Build disk cache class, taking into account the non_int_type."""

    @dataclass(frozen=True)
    class PintHeader(fc.InvalidateByExist, fc.NameByFields, fc.BasicPythonHeader):

        from . import __version__

        pint_version: str = __version__
        non_int_type: str = field(default_factory=lambda: non_int_type.__qualname__)

    class PathHeader(fc.NameByFileContent, PintHeader):
        pass

    class DefinitionFilesHeader(fc.NameByHashIter, PintHeader):
        @classmethod
        def from_definition_files(cls, dfs: DefinitionFiles, reader_id):
            return cls(tuple(df.content_hash for df in dfs), reader_id)

    class PintDiskCache(fc.DiskCache):

        _header_classes = {
            pathlib.Path: PathHeader,
            str: PathHeader.from_string,
            DefinitionFiles: DefinitionFilesHeader.from_definition_files,
        }

    return PintDiskCache


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

    _diskcache: fc.DiskCache

    handled_classes = (ImportDefinition,)

    def __init__(self, non_int_type=float, raise_on_error=True, cache_folder=None):
        self._directives = {}
        self._non_int_type = non_int_type
        self._raise_on_error = raise_on_error
        self.register_class("@import", ImportDefinition)

        if isinstance(cache_folder, (str, pathlib.Path)):
            self._diskcache = build_disk_cache_class(non_int_type)(cache_folder)
        else:
            self._diskcache = cache_folder

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

    def parse(self, file, is_resource: bool = False) -> DefinitionFiles:
        """Parse a file or resource into a collection of DefinitionFile that will
        include all other files imported.

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
            if self._diskcache is None:
                parsed = self.parse_single(path, None)
            else:
                parsed, content_hash = self._diskcache.load(
                    path, self.parse_single, True
                )

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
        return DefinitionFiles(out)

    def parse_single_resource(self, resource_name: str) -> DefinitionFile:
        """Parse a resource in the package into a DefinitionFile.

        Imported files will appear as ImportDefinition objects and
        will not be followed.

        This method will try to load it first as a regular file
        (with a path and mtime) to allow caching.
        If this files (i.e. the resource is not filesystem file)
        it will use python importlib.resources.read_binary
        """

        with resources.path(__package__, resource_name) as p:
            filepath = p.resolve()

        if filepath.exists():
            if self._diskcache is None:
                return self.parse_single(filepath, None)
            else:
                definition_file, content_hash = self._diskcache.load(
                    filepath, self.parse_single, True
                )
                return definition_file

        logger.debug("Cannot use_cache resource (yet) without a real path")
        return self._parse_single_resource(resource_name)

    def _parse_single_resource(self, resource_name: str) -> DefinitionFile:
        rbytes = resources.read_binary(__package__, resource_name)
        if self._diskcache:
            hdr = self._diskcache.PathHeader(rbytes)
            content_hash = self._diskcache.cache_stem_for(hdr)
        else:
            content_hash = None

        si = SourceIterator(
            StringIO(rbytes.decode("utf-8")), resource_name, is_resource=True
        )
        parsed_lines = tuple(self.yield_from_source_iterator(si))
        return DefinitionFile(
            filename=pathlib.Path(resource_name),
            is_resource=True,
            mtime=None,
            content_hash=content_hash,
            parsed_lines=parsed_lines,
        )

    def parse_single(
        self, filepath: pathlib.Path, content_hash: Optional[str]
    ) -> DefinitionFile:
        """Parse a filepath without nesting into dependent files.

        Imported files will appear as ImportDefinition objects and
        will not be followed.

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

        return DefinitionFile(
            filename=filename,
            is_resource=False,
            mtime=mtime,
            content_hash=content_hash,
            parsed_lines=parsed_lines,
        )

    def parse_lines(self, lines: Iterable[str]) -> DefinitionFile:
        """Parse an iterable of strings into a dependent file"""
        si = SourceIterator(lines, None, False)
        parsed_lines = tuple(self.yield_from_source_iterator(si))
        df = DefinitionFile(None, False, None, "", parsed_lines=parsed_lines)
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
