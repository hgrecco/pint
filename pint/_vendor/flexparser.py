"""
    flexparser.flexparser
    ~~~~~~~~~~~~~~~~~~~~~

    Classes and functions to create parsers.

    The idea is quite simple. You write a class for every type of content
    (called here ``ParsedStatement``) you need to parse. Each class should
    have a ``from_string`` constructor. We used extensively the ``typing``
    module to make the output structure easy to use and less error prone.

    For more information, take a look at https://github.com/hgrecco/flexparser

    :copyright: 2022 by flexparser Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import functools
import hashlib
import hmac
import inspect
import logging
import pathlib
import re
import sys
import typing as ty
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from typing import Optional, Tuple, Type

_LOGGER = logging.getLogger("flexparser")

_SENTINEL = object()


################
# Exceptions
################


@dataclass(frozen=True)
class Statement:
    """Base class for parsed elements within a source file."""

    start_line: int = dataclasses.field(init=False, default=None)
    start_col: int = dataclasses.field(init=False, default=None)

    end_line: int = dataclasses.field(init=False, default=None)
    end_col: int = dataclasses.field(init=False, default=None)

    raw: str = dataclasses.field(init=False, default=None)

    @classmethod
    def from_statement(cls, statement: Statement):
        out = cls()
        out.set_position(*statement.get_position())
        out.set_raw(statement.raw)
        return out

    @classmethod
    def from_statement_iterator_element(cls, values: ty.Tuple[int, int, int, int, str]):
        out = cls()
        out.set_position(*values[:-1])
        out.set_raw(values[-1])
        return out

    @property
    def format_position(self):
        if self.start_line is None:
            return "N/A"
        return "%d,%d-%d,%d" % self.get_position()

    @property
    def raw_strip(self):
        return self.raw.strip()

    def get_position(self):
        return self.start_line, self.start_col, self.end_line, self.end_col

    def set_position(self, start_line, start_col, end_line, end_col):
        object.__setattr__(self, "start_line", start_line)
        object.__setattr__(self, "start_col", start_col)
        object.__setattr__(self, "end_line", end_line)
        object.__setattr__(self, "end_col", end_col)
        return self

    def set_raw(self, raw):
        object.__setattr__(self, "raw", raw)
        return self

    def set_simple_position(self, line, col, width):
        return self.set_position(line, col, line, col + width)


@dataclass(frozen=True)
class ParsingError(Statement, Exception):
    """Base class for all parsing exceptions in this package."""

    def __str__(self):
        return Statement.__str__(self)


@dataclass(frozen=True)
class UnknownStatement(ParsingError):
    """A string statement could not bee parsed."""

    def __str__(self):
        return f"Could not parse '{self.raw}' ({self.format_position})"


@dataclass(frozen=True)
class UnhandledParsingError(ParsingError):
    """Base class for all parsing exceptions in this package."""

    ex: Exception

    def __str__(self):
        return f"Unhandled exception while parsing '{self.raw}' ({self.format_position}): {self.ex}"


@dataclass(frozen=True)
class UnexpectedEOF(ParsingError):
    """End of file was found within an open block."""


#############################
# Useful methods and classes
#############################


@dataclass(frozen=True)
class Hash:
    algorithm_name: str
    hexdigest: str

    def __eq__(self, other: Hash):
        return (
            isinstance(other, Hash)
            and self.algorithm_name != ""
            and self.algorithm_name == other.algorithm_name
            and hmac.compare_digest(self.hexdigest, other.hexdigest)
        )

    @classmethod
    def from_bytes(cls, algorithm, b: bytes):
        hasher = algorithm(b)
        return cls(hasher.name, hasher.hexdigest())

    @classmethod
    def from_file_pointer(cls, algorithm, fp: ty.BinaryIO):
        return cls.from_bytes(algorithm, fp.read())

    @classmethod
    def nullhash(cls):
        return cls("", "")


def _yield_types(
    obj, valid_subclasses=(object,), recurse_origin=(tuple, list, ty.Union)
):
    """Recursively transverse type annotation if the
    origin is any of the types in `recurse_origin`
    and yield those type which are subclasses of `valid_subclasses`.

    """
    if ty.get_origin(obj) in recurse_origin:
        for el in ty.get_args(obj):
            yield from _yield_types(el, valid_subclasses, recurse_origin)
    else:
        if inspect.isclass(obj) and issubclass(obj, valid_subclasses):
            yield obj


class classproperty:  # noqa N801
    """Decorator for a class property

    In Python 3.9+ can be replaced by

        @classmethod
        @property
        def myprop(self):
            return 42

    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def is_relative_to(self, *other):
    """Return True if the path is relative to another path or False.

    In Python 3.9+ can be replaced by

        path.is_relative_to(other)
    """
    try:
        self.relative_to(*other)
        return True
    except ValueError:
        return False


class DelimiterInclude(enum.IntEnum):
    """Specifies how to deal with delimiters while parsing."""

    #: Split at delimiter, not including in any string
    SPLIT = enum.auto()

    #: Split after, keeping the delimiter with previous string.
    SPLIT_AFTER = enum.auto()

    #: Split before, keeping the delimiter with next string.
    SPLIT_BEFORE = enum.auto()

    #: Do not split at delimiter.
    DO_NOT_SPLIT = enum.auto()


class DelimiterAction(enum.IntEnum):
    """Specifies how to deal with delimiters while parsing."""

    #: Continue parsing normally.
    CONTINUE = enum.auto()

    #: Capture everything til end of line as a whole.
    CAPTURE_NEXT_TIL_EOL = enum.auto()

    #: Stop parsing line and move to next.
    STOP_PARSING_LINE = enum.auto()

    #: Stop parsing content.
    STOP_PARSING = enum.auto()


DO_NOT_SPLIT_EOL = {
    "\r\n": (DelimiterInclude.DO_NOT_SPLIT, DelimiterAction.CONTINUE),
    "\n": (DelimiterInclude.DO_NOT_SPLIT, DelimiterAction.CONTINUE),
    "\r": (DelimiterInclude.DO_NOT_SPLIT, DelimiterAction.CONTINUE),
}

SPLIT_EOL = {
    "\r\n": (DelimiterInclude.SPLIT, DelimiterAction.CONTINUE),
    "\n": (DelimiterInclude.SPLIT, DelimiterAction.CONTINUE),
    "\r": (DelimiterInclude.SPLIT, DelimiterAction.CONTINUE),
}

_EOLs_set = set(DO_NOT_SPLIT_EOL.keys())


@functools.lru_cache
def _build_delimiter_pattern(delimiters: ty.Tuple[str, ...]) -> re.Pattern:
    """Compile a tuple of delimiters into a regex expression with a capture group
    around the delimiter.
    """
    return re.compile("|".join(f"({re.escape(el)})" for el in delimiters))


############
# Iterators
############

DelimiterDictT = ty.Dict[str, ty.Tuple[DelimiterInclude, DelimiterAction]]


class Spliter:
    """Content iterator splitting according to given delimiters.

    The pattern can be changed dynamically sending a new pattern to the generator,
    see DelimiterInclude and DelimiterAction for more information.

    The current scanning position can be changed at any time.

    Parameters
    ----------
    content : str
    delimiters : ty.Dict[str, ty.Tuple[DelimiterInclude, DelimiterAction]]

    Yields
    ------
    start_line : int
        line number of the start of the content (zero-based numbering).
    start_col : int
        column number of the start of the content (zero-based numbering).
    end_line : int
        line number of the end of the content (zero-based numbering).
    end_col : int
        column number of the end of the content (zero-based numbering).
    part : str
        part of the text between delimiters.
    """

    _pattern: ty.Optional[re.Pattern]
    _delimiters: DelimiterDictT

    __stop_searching_in_line = False

    __pending = ""
    __first_line_col = None

    __lines = ()
    __lineno = 0
    __colno = 0

    def __init__(self, content: str, delimiters: DelimiterDictT):
        self.set_delimiters(delimiters)
        self.__lines = content.splitlines(keepends=True)

    def set_position(self, lineno: int, colno: int):
        self.__lineno, self.__colno = lineno, colno

    def set_delimiters(self, delimiters: DelimiterDictT):
        for k, v in delimiters.items():
            if v == (DelimiterInclude.DO_NOT_SPLIT, DelimiterAction.STOP_PARSING):
                raise ValueError(
                    f"The delimiter action for {k} is not a valid combination ({v})"
                )
        # Build a pattern but removing eols
        _pat_dlm = tuple(set(delimiters.keys()) - _EOLs_set)
        if _pat_dlm:
            self._pattern = _build_delimiter_pattern(_pat_dlm)
        else:
            self._pattern = None
        # We add the end of line as delimiters if not present.
        self._delimiters = {**DO_NOT_SPLIT_EOL, **delimiters}

    def __iter__(self):
        return self

    def __next__(self):
        if self.__lineno >= len(self.__lines):
            raise StopIteration

        while True:
            if self.__stop_searching_in_line:
                # There must be part of a line pending to parse
                # due to stop
                line = self.__lines[self.__lineno]
                mo = None
                self.__stop_searching_in_line = False
            else:
                # We get the current line and the find the first delimiter.
                line = self.__lines[self.__lineno]
                if self._pattern is None:
                    mo = None
                else:
                    mo = self._pattern.search(line, self.__colno)

            if mo is None:
                # No delimiter was found,
                # which should happen at end of the content or end of line
                for k in DO_NOT_SPLIT_EOL.keys():
                    if line.endswith(k):
                        dlm = line[-len(k) :]
                        end_col, next_col = len(line) - len(k), 0
                        break
                else:
                    # No EOL found, this is end of content
                    dlm = None
                    end_col, next_col = len(line), 0

                next_line = self.__lineno + 1

            else:
                next_line = self.__lineno
                end_col, next_col = mo.span()
                dlm = mo.group()

            part = line[self.__colno : end_col]

            include, action = self._delimiters.get(
                dlm, (DelimiterInclude.SPLIT, DelimiterAction.STOP_PARSING)
            )

            if include == DelimiterInclude.SPLIT:
                next_pending = ""
            elif include == DelimiterInclude.SPLIT_AFTER:
                end_col += len(dlm)
                part = part + dlm
                next_pending = ""
            elif include == DelimiterInclude.SPLIT_BEFORE:
                next_pending = dlm
            elif include == DelimiterInclude.DO_NOT_SPLIT:
                self.__pending += line[self.__colno : end_col] + dlm
                next_pending = ""
            else:
                raise ValueError(f"Unknown action {include}.")

            if action == DelimiterAction.STOP_PARSING:
                # this will raise a StopIteration in the next call.
                next_line = len(self.__lines)
            elif action == DelimiterAction.STOP_PARSING_LINE:
                next_line = self.__lineno + 1
                next_col = 0

            start_line = self.__lineno
            start_col = self.__colno
            end_line = self.__lineno

            self.__lineno = next_line
            self.__colno = next_col

            if action == DelimiterAction.CAPTURE_NEXT_TIL_EOL:
                self.__stop_searching_in_line = True

            if include == DelimiterInclude.DO_NOT_SPLIT:
                self.__first_line_col = start_line, start_col
            else:
                if self.__first_line_col is None:
                    out = (
                        start_line,
                        start_col - len(self.__pending),
                        end_line,
                        end_col,
                        self.__pending + part,
                    )
                else:
                    out = (
                        *self.__first_line_col,
                        end_line,
                        end_col,
                        self.__pending + part,
                    )
                    self.__first_line_col = None
                self.__pending = next_pending
                return out


class StatementIterator:
    """Content peekable iterator splitting according to given delimiters.

    The pattern can be changed dynamically sending a new pattern to the generator,
    see DelimiterInclude and DelimiterAction for more information.

    Parameters
    ----------
    content : str
    delimiters : dict[str, ty.Tuple[DelimiterInclude, DelimiterAction]]

    Yields
    ------
    Statement
    """

    _cache: ty.Deque[Statement]

    def __init__(
        self, content: str, delimiters: DelimiterDictT, strip_spaces: bool = True
    ):
        self._cache = collections.deque()
        self._spliter = Spliter(content, delimiters)
        self._strip_spaces = strip_spaces

    def __iter__(self):
        return self

    def set_delimiters(self, delimiters: DelimiterDictT):
        self._spliter.set_delimiters(delimiters)
        if self._cache:
            value = self.peek()
            # Elements are 1 based indexing, while splitter is 0 based.
            self._spliter.set_position(value.start_line - 1, value.start_col)
            self._cache.clear()

    def _get_next_strip(self) -> Statement:
        part = ""
        while not part:
            start_line, start_col, end_line, end_col, part = next(self._spliter)
            lo = len(part)
            part = part.lstrip()
            start_col += lo - len(part)

            lo = len(part)
            part = part.rstrip()
            end_col -= lo - len(part)

        return Statement.from_statement_iterator_element(
            (start_line + 1, start_col, end_line + 1, end_col, part)
        )

    def _get_next(self) -> Statement:
        if self._strip_spaces:
            return self._get_next_strip()

        part = ""
        while not part:
            start_line, start_col, end_line, end_col, part = next(self._spliter)

        return Statement.from_statement_iterator_element(
            (start_line + 1, start_col, end_line + 1, end_col, part)
        )

    def peek(self, default=_SENTINEL) -> Statement:
        """Return the item that will be next returned from ``next()``.

        Return ``default`` if there are no items left. If ``default`` is not
        provided, raise ``StopIteration``.

        """
        if not self._cache:
            try:
                self._cache.append(self._get_next())
            except StopIteration:
                if default is _SENTINEL:
                    raise
                return default
        return self._cache[0]

    def __next__(self) -> Statement:
        if self._cache:
            return self._cache.popleft()
        else:
            return self._get_next()


###########
# Parsing
###########

# Configuration type
CT = ty.TypeVar("CT")
PST = ty.TypeVar("PST", bound="ParsedStatement")
LineColStr = Tuple[int, int, str]
FromString = ty.Union[None, PST, ParsingError]
Consume = ty.Union[PST, ParsingError]
NullableConsume = ty.Union[None, PST, ParsingError]

Single = ty.Union[PST, ParsingError]
Multi = ty.Tuple[ty.Union[PST, ParsingError], ...]


@dataclass(frozen=True)
class ParsedStatement(ty.Generic[CT], Statement):
    """A single parsed statement.

    In order to write your own, you need to subclass it as a
    frozen dataclass and implement the parsing logic by overriding
    `from_string` classmethod.

    Takes two arguments: the string to parse and an object given
    by the parser which can be used to store configuration information.

    It should return an instance of this class if parsing
    was successful or None otherwise
    """

    @classmethod
    def from_string(cls: Type[PST], s: str) -> FromString[PST]:
        """Parse a string into a ParsedStatement.

        Return files and their meaning:
        1. None: the string cannot be parsed with this class.
        2. A subclass of ParsedStatement: the string was parsed successfully
        3. A subclass of ParsingError the string could be parsed with this class but there is
           an error.
        """
        raise NotImplementedError(
            "ParsedStatement subclasses must implement "
            "'from_string' or 'from_string_and_config'"
        )

    @classmethod
    def from_string_and_config(cls: Type[PST], s: str, config: CT) -> FromString[PST]:
        """Parse a string into a ParsedStatement.

        Return files and their meaning:
        1. None: the string cannot be parsed with this class.
        2. A subclass of ParsedStatement: the string was parsed successfully
        3. A subclass of ParsingError the string could be parsed with this class but there is
           an error.
        """
        return cls.from_string(s)

    @classmethod
    def from_statement_and_config(
        cls: Type[PST], statement: Statement, config: CT
    ) -> FromString[PST]:
        try:
            out = cls.from_string_and_config(statement.raw, config)
        except Exception as ex:
            out = UnhandledParsingError(ex)

        if out is None:
            return None

        out.set_position(*statement.get_position())
        out.set_raw(statement.raw)
        return out

    @classmethod
    def consume(
        cls: Type[PST], statement_iterator: StatementIterator, config: CT
    ) -> NullableConsume[PST]:
        """Peek into the iterator and try to parse.

        Return files and their meaning:
        1. None: the string cannot be parsed with this class, the iterator is kept an the current place.
        2. a subclass of ParsedStatement: the string was parsed successfully, advance the iterator.
        3. a subclass of ParsingError: the string could be parsed with this class but there is
           an error, advance the iterator.
        """
        statement = statement_iterator.peek()
        parsed_statement = cls.from_statement_and_config(statement, config)
        if parsed_statement is None:
            return None
        next(statement_iterator)
        return parsed_statement


OPST = ty.TypeVar("OPST", bound="ParsedStatement")
IPST = ty.TypeVar("IPST", bound="ParsedStatement")
CPST = ty.TypeVar("CPST", bound="ParsedStatement")
BT = ty.TypeVar("BT", bound="Block")
RBT = ty.TypeVar("RBT", bound="RootBlock")


@dataclass(frozen=True)
class Block(ty.Generic[OPST, IPST, CPST, CT]):
    """A sequence of statements with an opening, body and closing."""

    opening: Consume[OPST]
    body: Tuple[Consume[IPST], ...]
    closing: Consume[CPST]

    delimiters = {}

    @property
    def start_line(self):
        return self.opening.start_line

    @property
    def start_col(self):
        return self.opening.start_col

    @property
    def end_line(self):
        return self.closing.end_line

    @property
    def end_col(self):
        return self.closing.end_col

    def get_position(self):
        return self.start_line, self.start_col, self.end_line, self.end_col

    @property
    def format_position(self):
        if self.start_line is None:
            return "N/A"
        return "%d,%d-%d,%d" % self.get_position()

    @classmethod
    def subclass_with(cls, *, opening=None, body=None, closing=None):
        @dataclass(frozen=True)
        class CustomBlock(Block):
            pass

        if opening:
            CustomBlock.__annotations__["opening"] = Single[ty.Union[opening]]
        if body:
            CustomBlock.__annotations__["body"] = Multi[ty.Union[body]]
        if closing:
            CustomBlock.__annotations__["closing"] = Single[ty.Union[closing]]

        return CustomBlock

    def __iter__(self) -> Iterator[Statement]:
        yield self.opening
        for el in self.body:
            if isinstance(el, Block):
                yield from el
            else:
                yield el
        yield self.closing

    def iter_blocks(self) -> Iterator[ty.Union[Block, Statement]]:
        yield self.opening
        yield from self.body
        yield self.closing

    ###################################################
    # Convenience methods to iterate parsed statements
    ###################################################

    _ElementT = ty.TypeVar("_ElementT", bound=Statement)

    def filter_by(self, *klass: Type[_ElementT]) -> Iterator[_ElementT]:
        """Yield elements of a given class or classes."""
        yield from (el for el in self if isinstance(el, klass))  # noqa Bug in pycharm.

    @cached_property
    def errors(self) -> ty.Tuple[ParsingError, ...]:
        """Tuple of errors found."""
        return tuple(self.filter_by(ParsingError))

    @property
    def has_errors(self) -> bool:
        """True if errors were found during parsing."""
        return bool(self.errors)

    ####################
    # Statement classes
    ####################

    @classproperty
    def opening_classes(cls) -> Iterator[Type[OPST]]:
        """Classes representing any of the parsed statement that can open this block."""
        opening = ty.get_type_hints(cls)["opening"]
        yield from _yield_types(opening, ParsedStatement)

    @classproperty
    def body_classes(cls) -> Iterator[Type[IPST]]:
        """Classes representing any of the parsed statement that can be in the body."""
        body = ty.get_type_hints(cls)["body"]
        yield from _yield_types(body, (ParsedStatement, Block))

    @classproperty
    def closing_classes(cls) -> Iterator[Type[CPST]]:
        """Classes representing any of the parsed statement that can close this block."""
        closing = ty.get_type_hints(cls)["closing"]
        yield from _yield_types(closing, ParsedStatement)

    ##########
    # Consume
    ##########

    @classmethod
    def consume_opening(
        cls: Type[BT], statement_iterator: StatementIterator, config: CT
    ) -> NullableConsume[OPST]:
        """Peek into the iterator and try to parse with any of the opening classes.

        See `ParsedStatement.consume` for more details.
        """
        for c in cls.opening_classes:
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        return None

    @classmethod
    def consume_body(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> Consume[IPST]:
        """Peek into the iterator and try to parse with any of the body classes.

        If the statement cannot be parsed, a UnknownStatement is returned.
        """
        for c in cls.body_classes:
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        el = next(statement_iterator)
        return UnknownStatement.from_statement(el)

    @classmethod
    def consume_closing(
        cls: Type[BT], statement_iterator: StatementIterator, config: CT
    ) -> NullableConsume[CPST]:
        """Peek into the iterator and try to parse with any of the opening classes.

        See `ParsedStatement.consume` for more details.
        """
        for c in cls.closing_classes:
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        return None

    @classmethod
    def consume_body_closing(
        cls: Type[BT], opening: OPST, statement_iterator: StatementIterator, config: CT
    ) -> BT:
        body = []
        closing = None
        last_line = opening.end_line
        while closing is None:
            try:
                closing = cls.consume_closing(statement_iterator, config)
                if closing is not None:
                    continue
                el = cls.consume_body(statement_iterator, config)
                body.append(el)
                last_line = el.end_line
            except StopIteration:
                closing = cls.on_stop_iteration(config)
                closing.set_position(last_line + 1, 0, last_line + 1, 0)

        return cls(opening, tuple(body), closing)

    @classmethod
    def consume(
        cls: Type[BT], statement_iterator: StatementIterator, config: CT
    ) -> Optional[BT]:
        """Try consume the block.

        Possible outcomes:
        1. The opening was not matched, return None.
        2. A subclass of Block, where body and closing migh contain errors.
        """
        opening = cls.consume_opening(statement_iterator, config)
        if opening is None:
            return None

        return cls.consume_body_closing(opening, statement_iterator, config)

    @classmethod
    def on_stop_iteration(cls, config):
        return UnexpectedEOF()


@dataclass(frozen=True)
class BOS(ParsedStatement[CT]):
    """Beginning of source."""

    # Hasher algorithm name and hexdigest
    content_hash: Hash

    @classmethod
    def from_string_and_config(cls: Type[PST], s: str, config: CT) -> FromString[PST]:
        raise RuntimeError("BOS cannot be constructed from_string_and_config")

    @property
    def location(self) -> SourceLocationT:
        return "<undefined>"


@dataclass(frozen=True)
class BOF(BOS):
    """Beginning of file."""

    path: pathlib.Path

    # Modification time of the file.
    mtime: float

    @property
    def location(self) -> SourceLocationT:
        return self.path


@dataclass(frozen=True)
class BOR(BOS):
    """Beginning of resource."""

    package: str
    resource_name: str

    @property
    def location(self) -> SourceLocationT:
        return self.package, self.resource_name


@dataclass(frozen=True)
class EOS(ParsedStatement[CT]):
    """End of sequence."""

    @classmethod
    def from_string_and_config(cls: Type[PST], s: str, config: CT) -> FromString[PST]:
        return cls()


class RootBlock(ty.Generic[IPST, CT], Block[BOS, IPST, EOS, CT]):
    """A sequence of statement flanked by the beginning and ending of stream."""

    opening: Single[BOS]
    closing: Single[EOS]

    @classmethod
    def subclass_with(cls, *, body=None):
        @dataclass(frozen=True)
        class CustomRootBlock(RootBlock):
            pass

        if body:
            CustomRootBlock.__annotations__["body"] = Multi[ty.Union[body]]

        return CustomRootBlock

    @classmethod
    def consume_opening(
        cls: Type[RBT], statement_iterator: StatementIterator, config: CT
    ) -> NullableConsume[BOS]:
        raise RuntimeError(
            "Implementation error, 'RootBlock.consume_opening' should never be called"
        )

    @classmethod
    def consume(
        cls: Type[RBT], statement_iterator: StatementIterator, config: CT
    ) -> RBT:
        block = super().consume(statement_iterator, config)
        if block is None:
            raise RuntimeError(
                "Implementation error, 'RootBlock.consume' should never return None"
            )
        return block

    @classmethod
    def consume_closing(
        cls: Type[RBT], statement_iterator: StatementIterator, config: CT
    ) -> NullableConsume[EOS]:
        return None

    @classmethod
    def on_stop_iteration(cls, config):
        return EOS()


#################
# Source parsing
#################

ResourceT = ty.Tuple[str, str]  # package name, resource name
StrictLocationT = ty.Union[pathlib.Path, ResourceT]
SourceLocationT = ty.Union[str, StrictLocationT]


@dataclass(frozen=True)
class ParsedSource(ty.Generic[RBT, CT]):

    parsed_source: RBT

    # Parser configuration.
    config: CT

    @property
    def location(self) -> StrictLocationT:
        return self.parsed_source.opening.location

    @cached_property
    def has_errors(self) -> bool:
        return self.parsed_source.has_errors

    def errors(self):
        yield from self.parsed_source.errors


@dataclass(frozen=True)
class CannotParseResourceAsFile(Exception):
    """The requested python package resource cannot be located as a file
    in the file system.
    """

    package: str
    resource_name: str


class Parser(ty.Generic[RBT, CT]):
    """Parser class."""

    #: class to iterate through statements in a source unit.
    _statement_iterator_class: Type[StatementIterator] = StatementIterator

    #: Delimiters.
    _delimiters: DelimiterDictT = SPLIT_EOL

    _strip_spaces: bool = True

    #: root block class containing statements and blocks can be parsed.
    _root_block_class: Type[RBT]

    #: source file text encoding.
    _encoding = "utf-8"

    #: configuration passed to from_string functions.
    _config: CT

    #: try to open resources as files.
    _prefer_resource_as_file: bool

    #: parser algorithm to us. Must be a callable member of hashlib
    _hasher = hashlib.blake2b

    def __init__(self, config: CT, prefer_resource_as_file=True):
        self._config = config
        self._prefer_resource_as_file = prefer_resource_as_file

    def parse(self, source_location: SourceLocationT) -> ParsedSource[RBT, CT]:
        """Parse a file into a ParsedSourceFile or ParsedResource.

        Parameters
        ----------
        source_location:
            if str or pathlib.Path is interpreted as a file.
            if (str, str) is interpreted as (package, resource) using the resource python api.
        """
        if isinstance(source_location, tuple) and len(source_location) == 2:
            if self._prefer_resource_as_file:
                try:
                    return self.parse_resource_from_file(*source_location)
                except CannotParseResourceAsFile:
                    pass
            return self.parse_resource(*source_location)

        if isinstance(source_location, str):
            return self.parse_file(pathlib.Path(source_location))

        if isinstance(source_location, pathlib.Path):
            return self.parse_file(source_location)

        raise TypeError(
            f"Unknown type {type(source_location)}, "
            "use str or pathlib.Path for files or "
            "(package: str, resource_name: str) tuple "
            "for a resource."
        )

    def parse_bytes(self, b: bytes, bos: BOS = None) -> ParsedSource[RBT, CT]:
        if bos is None:
            bos = BOS(Hash.from_bytes(self._hasher, b)).set_simple_position(0, 0, 0)

        sic = self._statement_iterator_class(
            b.decode(self._encoding), self._delimiters, self._strip_spaces
        )

        parsed = self._root_block_class.consume_body_closing(bos, sic, self._config)

        return ParsedSource(
            parsed,
            self._config,
        )

    def parse_file(self, path: pathlib.Path) -> ParsedSource[RBT, CT]:
        """Parse a file into a ParsedSourceFile.

        Parameters
        ----------
        path
            path of the file.
        """
        with path.open(mode="rb") as fi:
            content = fi.read()

        bos = BOF(
            Hash.from_bytes(self._hasher, content), path, path.stat().st_mtime
        ).set_simple_position(0, 0, 0)
        return self.parse_bytes(content, bos)

    def parse_resource_from_file(
        self, package: str, resource_name: str
    ) -> ParsedSource[RBT, CT]:
        """Parse a resource into a ParsedSourceFile, opening as a file.

        Parameters
        ----------
        package
            package name where the resource is located.
        resource_name
            name of the resource
        """
        if sys.version_info < (3, 9):
            # Remove when Python 3.8 is dropped
            with resources.path(package, resource_name) as p:
                path = p.resolve()
        else:
            with resources.as_file(
                resources.files(package).joinpath(resource_name)
            ) as p:
                path = p.resolve()

        if path.exists():
            return self.parse_file(path)

        raise CannotParseResourceAsFile(package, resource_name)

    def parse_resource(self, package: str, resource_name: str) -> ParsedSource[RBT, CT]:
        """Parse a resource into a ParsedResource.

        Parameters
        ----------
        package
            package name where the resource is located.
        resource_name
            name of the resource
        """
        if sys.version_info < (3, 9):
            # Remove when Python 3.8 is dropped
            with resources.open_binary(package, resource_name) as fi:
                content = fi.read()
        else:
            with resources.files(package).joinpath(resource_name).open("rb") as fi:
                content = fi.read()

        bos = BOR(
            Hash.from_bytes(self._hasher, content), package, resource_name
        ).set_simple_position(0, 0, 0)

        return self.parse_bytes(content, bos)


##########
# Project
##########


class IncludeStatement(ParsedStatement):
    """ "Include statements allow to merge files."""

    @property
    def target(self) -> str:
        raise NotImplementedError(
            "IncludeStatement subclasses must implement target property."
        )


class ParsedProject(
    ty.Dict[
        ty.Optional[ty.Tuple[StrictLocationT, str]],
        ParsedSource,
    ]
):
    """Collection of files, independent or connected via IncludeStatement.

    Keys are either an absolute pathname  or a tuple package name, resource name.

    None is the name of the root.

    """

    @cached_property
    def has_errors(self) -> bool:
        return any(el.has_errors for el in self.values())

    def errors(self):
        for el in self.values():
            yield from el.errors()

    def _iter_statements(self, items, seen, include_only_once):
        """Iter all definitions in the order they appear,
        going into the included files.
        """
        for source_location, parsed in items:
            seen.add(source_location)
            for parsed_statement in parsed.parsed_source:
                if isinstance(parsed_statement, IncludeStatement):
                    location = parsed.location, parsed_statement.target
                    if location in seen and include_only_once:
                        raise ValueError(f"{location} was already included.")
                    yield from self._iter_statements(
                        ((location, self[location]),), seen, include_only_once
                    )
                else:
                    yield parsed_statement

    def iter_statements(self, include_only_once=True):
        """Iter all definitions in the order they appear,
        going into the included files.

        Parameters
        ----------
        include_only_once
            if true, each file cannot be included more than once.
        """
        yield from self._iter_statements([(None, self[None])], set(), include_only_once)

    def _iter_blocks(self, items, seen, include_only_once):
        """Iter all definitions in the order they appear,
        going into the included files.
        """
        for source_location, parsed in items:
            seen.add(source_location)
            for parsed_statement in parsed.parsed_source.iter_blocks():
                if isinstance(parsed_statement, IncludeStatement):
                    location = parsed.location, parsed_statement.target
                    if location in seen and include_only_once:
                        raise ValueError(f"{location} was already included.")
                    yield from self._iter_blocks(
                        ((location, self[location]),), seen, include_only_once
                    )
                else:
                    yield parsed_statement

    def iter_blocks(self, include_only_once=True):
        """Iter all definitions in the order they appear,
        going into the included files.

        Parameters
        ----------
        include_only_once
            if true, each file cannot be included more than once.
        """
        yield from self._iter_blocks([(None, self[None])], set(), include_only_once)


def default_locator(source_location: StrictLocationT, target: str) -> StrictLocationT:
    """Return a new location from current_location and target."""

    if isinstance(source_location, pathlib.Path):
        current_location = pathlib.Path(source_location).resolve()

        if current_location.is_file():
            current_path = current_location.parent
        else:
            current_path = current_location

        target_path = pathlib.Path(target)
        if target_path.is_absolute():
            raise ValueError(
                f"Cannot refer to absolute paths in import statements ({source_location}, {target})."
            )

        tmp = (current_path / target_path).resolve()
        if not is_relative_to(tmp, current_path):
            raise ValueError(
                f"Cannot refer to locations above the current location ({source_location}, {target})"
            )

        return tmp.absolute()

    elif isinstance(source_location, tuple) and len(source_location) == 2:
        return source_location[0], target

    raise TypeError(
        f"Cannot handle type {type(source_location)}, "
        "use str or pathlib.Path for files or "
        "(package: str, resource_name: str) tuple "
        "for a resource."
    )


DefinitionT = ty.Union[ty.Type[Block], ty.Type[ParsedStatement]]

SpecT = ty.Union[
    ty.Type[Parser],
    DefinitionT,
    ty.Iterable[DefinitionT],
    ty.Type[RootBlock],
]


def build_parser_class(spec: SpecT, *, strip_spaces: bool = True, delimiters=None):
    """Build a custom parser class.

    Parameters
    ----------
    spec
        specification of the content to parse. Can be one of the following things:
        - Parser class.
        - Block or ParsedStatement derived class.
        - Iterable of Block or ParsedStatement derived class.
        - RootBlock derived class.
    strip_spaces : bool
        if True, spaces will be stripped for each statement before calling
        ``from_string_and_config``.
    delimiters : dict
        Specify how the source file is split into statements (See below).

    Delimiters dictionary
    ---------------------
        The delimiters are specified with the keys of the delimiters dict.
    The dict files can be used to further customize the iterator. Each
    consist of a tuple of two elements:
      1. A value of the DelimiterMode to indicate what to do with the
         delimiter string: skip it, attach keep it with previous or next string
      2. A boolean indicating if parsing should stop after fiSBT
         encountering this delimiter.
    """

    if delimiters is None:
        delimiters = SPLIT_EOL

    if isinstance(spec, type) and issubclass(spec, Parser):
        CustomParser = spec
    else:
        if isinstance(spec, (tuple, list)):

            for el in spec:
                if not issubclass(el, (Block, ParsedStatement)):
                    raise TypeError(
                        "Elements in root_block_class must be of type Block or ParsedStatement, "
                        f"not {el}"
                    )

            @dataclass(frozen=True)
            class CustomRootBlock(RootBlock):
                pass

            CustomRootBlock.__annotations__["body"] = Multi[ty.Union[spec]]

        elif isinstance(spec, type) and issubclass(spec, RootBlock):

            CustomRootBlock = spec

        elif isinstance(spec, type) and issubclass(spec, (Block, ParsedStatement)):

            @dataclass(frozen=True)
            class CustomRootBlock(RootBlock):
                pass

            CustomRootBlock.__annotations__["body"] = Multi[spec]

        else:
            raise TypeError(
                "`spec` must be of type RootBlock or tuple of type Block or ParsedStatement, "
                f"not {type(spec)}"
            )

        class CustomParser(Parser):

            _delimiters = delimiters
            _root_block_class = CustomRootBlock
            _strip_spaces = strip_spaces

    return CustomParser


def parse(
    entry_point: SourceLocationT,
    spec: SpecT,
    config=None,
    *,
    strip_spaces: bool = True,
    delimiters=None,
    locator: ty.Callable[[StrictLocationT, str], StrictLocationT] = default_locator,
    prefer_resource_as_file: bool = True,
    **extra_parser_kwargs,
) -> ParsedProject:
    """Parse sources into a ParsedProject dictionary.

    Parameters
    ----------
    entry_point
        file or resource, given as (package_name, resource_name).
    spec
        specification of the content to parse. Can be one of the following things:
        - Parser class.
        - Block or ParsedStatement derived class.
        - Iterable of Block or ParsedStatement derived class.
        - RootBlock derived class.
    config
        a configuration object that will be passed to `from_string_and_config`
        classmethod.
    strip_spaces : bool
        if True, spaces will be stripped for each statement before calling
        ``from_string_and_config``.
    delimiters : dict
        Specify how the source file is split into statements (See below).
    locator : Callable
        function that takes the current location and a target of an IncludeStatement
        and returns a new location.
    prefer_resource_as_file : bool
        if True, resources will try to be located in the filesystem if
        available.
    extra_parser_kwargs
        extra keyword arguments to be given to the parser.

    Delimiters dictionary
    ---------------------
        The delimiters are specified with the keys of the delimiters dict.
    The dict files can be used to further customize the iterator. Each
    consist of a tuple of two elements:
      1. A value of the DelimiterMode to indicate what to do with the
         delimiter string: skip it, attach keep it with previous or next string
      2. A boolean indicating if parsing should stop after fiSBT
         encountering this delimiter.
    """

    CustomParser = build_parser_class(
        spec, strip_spaces=strip_spaces, delimiters=delimiters
    )
    parser = CustomParser(
        config, prefer_resource_as_file=prefer_resource_as_file, **extra_parser_kwargs
    )

    pp = ParsedProject()

    # : ty.List[Optional[ty.Union[LocatorT, str]], ...]
    pending: ty.List[ty.Tuple[StrictLocationT, str]] = []
    if isinstance(entry_point, (str, pathlib.Path)):
        entry_point = pathlib.Path(entry_point)
        if not entry_point.is_absolute():
            entry_point = pathlib.Path.cwd() / entry_point

    elif not (isinstance(entry_point, tuple) and len(entry_point) == 2):
        raise TypeError(
            f"Cannot handle type {type(entry_point)}, "
            "use str or pathlib.Path for files or "
            "(package: str, resource_name: str) tuple "
            "for a resource."
        )

    pp[None] = parsed = parser.parse(entry_point)
    pending.extend(
        (parsed.location, el.target)
        for el in parsed.parsed_source.filter_by(IncludeStatement)
    )

    while pending:
        source_location, target = pending.pop(0)
        pp[(source_location, target)] = parsed = parser.parse(
            locator(source_location, target)
        )
        pending.extend(
            (parsed.location, el.target)
            for el in parsed.parsed_source.filter_by(IncludeStatement)
        )

    return pp


def parse_bytes(
    content: bytes,
    spec: SpecT,
    config=None,
    *,
    strip_spaces: bool = True,
    delimiters=None,
    **extra_parser_kwargs,
) -> ParsedProject:
    """Parse sources into a ParsedProject dictionary.

    Parameters
    ----------
    content
        bytes.
    spec
        specification of the content to parse. Can be one of the following things:
        - Parser class.
        - Block or ParsedStatement derived class.
        - Iterable of Block or ParsedStatement derived class.
        - RootBlock derived class.
    config
        a configuration object that will be passed to `from_string_and_config`
        classmethod.
    strip_spaces : bool
        if True, spaces will be stripped for each statement before calling
        ``from_string_and_config``.
    delimiters : dict
        Specify how the source file is split into statements (See below).
    """

    CustomParser = build_parser_class(
        spec, strip_spaces=strip_spaces, delimiters=delimiters
    )
    parser = CustomParser(config, prefer_resource_as_file=False, **extra_parser_kwargs)

    pp = ParsedProject()

    pp[None] = parsed = parser.parse_bytes(content)

    if any(parsed.parsed_source.filter_by(IncludeStatement)):
        raise ValueError("parse_bytes does not support using an IncludeStatement")

    return pp
