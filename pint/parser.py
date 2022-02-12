"""
    pint.parser
    ~~~~~~~~~~~

    Classes and methods to parse a definition text file into a DefinitionFile.

    :copyright: 2019 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from io import StringIO
from typing import Any, Callable, Dict, Generator, Optional, Tuple

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

    def __init__(self, non_int_type=float, raise_on_error=True):
        self._directives = {}
        self._non_int_type = non_int_type
        self._raise_on_error = raise_on_error
        self.register_class("@import", ImportDefinition)

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
        if hasattr(klass, "from_string"):
            self.register_directive(prefix, klass.from_string, True)
        elif hasattr(klass, "from_lines"):
            self.register_directive(prefix, klass.from_lines, False)
        else:
            raise ValueError(
                f"While registering {prefix}, {klass} does not have `from_string` or from_lines` method"
            )

    def parse(self, file, is_resource: bool = False) -> Tuple[DefinitionFile, ...]:
        """Parse a file, resource, iterable of strings or SourceIterator
        into a collection of DefinitionFile.

        Parameters
        ----------
        file
            definitions or file containing definition.
        is_resource
            indicates that the file is a resource file
            and therefore should be loaded from the package.
            (Default value = False)
        """
        out = []
        parsed = self.parse_single(file, is_resource)
        out.append(parsed)
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

    def parse_single(self, file, is_resource: bool = False) -> DefinitionFile:
        """Parse a file, resource, iterable of strings or SourceIterator
        without nesting into dependent files.

        Parameters
        ----------
        file
            definitions or file containing definition.
        is_resource
            indicates that the file is a resource file
            and therefore should be loaded from the package.
            (Default value = False)
        """
        # Permit both filenames and line-iterables

        try:
            if not isinstance(file, (str, pathlib.Path)):
                si = SourceIterator(file)
                parsed_lines = tuple(self._parse(si))

                filename = None
                mtime = None  # TODO What is the right value here?
            elif is_resource:
                rbytes = resources.read_binary(__package__, file)
                si = SourceIterator(
                    StringIO(rbytes.decode("utf-8")), file, is_resource=True
                )
                parsed_lines = tuple(self._parse(si))

                filename = file
                mtime = None
                try:
                    with resources.path(__package__, file) as p:
                        filename = p.resolve()
                        mtime = p.stat().st_mtime  # Will this always work:
                except Exception:
                    pass
            else:
                with open(file, encoding="utf-8") as fp:
                    si = SourceIterator(fp, file, is_resource=False)
                    parsed_lines = tuple(self._parse(si))

                filename = pathlib.Path(file).resolve()
                mtime = os.stat(file).st_mtime
        except Exception as e:
            msg = getattr(e, "message", "") or str(e)
            raise ValueError("While opening {}\n{}".format(file, msg))

        return DefinitionFile(filename, is_resource, mtime, parsed_lines)

    def _parse(self, source_iterator: SourceIterator) -> Generator[Tuple[int, Any]]:
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
