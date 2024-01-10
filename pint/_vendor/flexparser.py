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

import sys
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
import typing as ty
from dataclasses import dataclass
from functools import cached_property
from importlib import resources
from typing import Any, Union, Optional, no_type_check

if sys.version_info >= (3, 10):
    from typing import TypeAlias  # noqa
else:
    from typing_extensions import TypeAlias  # noqa


if sys.version_info >= (3, 11):
    from typing import Self  # noqa
else:
    from typing_extensions import Self  # noqa


_LOGGER = logging.getLogger("flexparser")

_SENTINEL = object()


class HasherProtocol(ty.Protocol):
    @property
    def name(self) -> str:
        ...

    def hexdigest(self) -> str:
        ...


class GenericInfo:
    _specialized: Optional[
        dict[type, Optional[list[tuple[type, dict[ty.TypeVar, type]]]]]
    ] = None

    @staticmethod
    def _summarize(d: dict[ty.TypeVar, type]) -> dict[ty.TypeVar, type]:
        d = d.copy()
        while True:
            for k, v in d.items():
                if isinstance(v, ty.TypeVar):
                    d[k] = d[v]
                    break
            else:
                return d

            del d[v]

    @classmethod
    def _specialization(cls) -> dict[ty.TypeVar, type]:
        if cls._specialized is None:
            return dict()

        out: dict[ty.TypeVar, type] = {}
        specialized = cls._specialized[cls]

        if specialized is None:
            return {}

        for parent, content in specialized:
            for tvar, typ in content.items():
                out[tvar] = typ
                origin = getattr(parent, "__origin__", None)
                if origin is not None and origin in cls._specialized:
                    out = {**origin._specialization(), **out}

        return out

    @classmethod
    def specialization(cls) -> dict[ty.TypeVar, type]:
        return GenericInfo._summarize(cls._specialization())

    def __init_subclass__(cls) -> None:
        if cls._specialized is None:
            cls._specialized = {GenericInfo: None}

        tv: list[ty.TypeVar] = []
        entries: list[tuple[type, dict[ty.TypeVar, type]]] = []

        for par in getattr(cls, "__parameters__", ()):
            if isinstance(par, ty.TypeVar):
                tv.append(par)

        for b in getattr(cls, "__orig_bases__", ()):
            for k in cls._specialized.keys():
                if getattr(b, "__origin__", None) is k:
                    entries.append((b, {k: v for k, v in zip(tv, b.__args__)}))
                    break

        cls._specialized[cls] = entries

        return super().__init_subclass__()


################
# Exceptions
################


@dataclass(frozen=True)
class Statement:
    """Base class for parsed elements within a source file."""

    is_position_set: bool = dataclasses.field(init=False, default=False, repr=False)

    start_line: int = dataclasses.field(init=False, default=0)
    start_col: int = dataclasses.field(init=False, default=0)

    end_line: int = dataclasses.field(init=False, default=0)
    end_col: int = dataclasses.field(init=False, default=0)

    raw: Optional[str] = dataclasses.field(init=False, default=None)

    @classmethod
    def from_statement(cls, statement: Statement) -> Self:
        out = cls()
        if statement.is_position_set:
            out.set_position(*statement.get_position())
        if statement.raw is not None:
            out.set_raw(statement.raw)
        return out

    @classmethod
    def from_statement_iterator_element(
        cls, values: tuple[int, int, int, int, str]
    ) -> Self:
        out = cls()
        out.set_position(*values[:-1])
        out.set_raw(values[-1])
        return out

    @property
    def format_position(self) -> str:
        if not self.is_position_set:
            return "N/A"
        return "%d,%d-%d,%d" % self.get_position()

    @property
    def raw_strip(self) -> Optional[str]:
        if self.raw is None:
            return None
        return self.raw.strip()

    def get_position(self) -> tuple[int, int, int, int]:
        if self.is_position_set:
            return self.start_line, self.start_col, self.end_line, self.end_col
        return 0, 0, 0, 0

    def set_position(
        self: Self, start_line: int, start_col: int, end_line: int, end_col: int
    ) -> Self:
        object.__setattr__(self, "is_position_set", True)
        object.__setattr__(self, "start_line", start_line)
        object.__setattr__(self, "start_col", start_col)
        object.__setattr__(self, "end_line", end_line)
        object.__setattr__(self, "end_col", end_col)
        return self

    def set_raw(self: Self, raw: str) -> Self:
        object.__setattr__(self, "raw", raw)
        return self

    def set_simple_position(self: Self, line: int, col: int, width: int) -> Self:
        return self.set_position(line, col, line, col + width)


@dataclass(frozen=True)
class ParsingError(Statement, Exception):
    """Base class for all parsing exceptions in this package."""

    def __str__(self) -> str:
        return Statement.__str__(self)


@dataclass(frozen=True)
class UnknownStatement(ParsingError):
    """A string statement could not bee parsed."""

    def __str__(self) -> str:
        return f"Could not parse '{self.raw}' ({self.format_position})"


@dataclass(frozen=True)
class UnhandledParsingError(ParsingError):
    """Base class for all parsing exceptions in this package."""

    ex: Exception

    def __str__(self) -> str:
        return f"Unhandled exception while parsing '{self.raw}' ({self.format_position}): {self.ex}"


@dataclass(frozen=True)
class UnexpectedEOS(ParsingError):
    """End of file was found within an open block."""


#############################
# Useful methods and classes
#############################


@dataclass(frozen=True)
class Hash:
    algorithm_name: str
    hexdigest: str

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Hash)
            and self.algorithm_name != ""
            and self.algorithm_name == other.algorithm_name
            and hmac.compare_digest(self.hexdigest, other.hexdigest)
        )

    @classmethod
    def from_bytes(
        cls,
        algorithm: ty.Callable[
            [
                bytes,
            ],
            HasherProtocol,
        ],
        b: bytes,
    ) -> Self:
        hasher = algorithm(b)
        return cls(hasher.name, hasher.hexdigest())

    @classmethod
    def from_file_pointer(
        cls,
        algorithm: ty.Callable[
            [
                bytes,
            ],
            HasherProtocol,
        ],
        fp: ty.BinaryIO,
    ) -> Self:
        return cls.from_bytes(algorithm, fp.read())

    @classmethod
    def nullhash(cls) -> Self:
        return cls("", "")


def _yield_types(
    obj: type,
    valid_subclasses: tuple[type, ...] = (object,),
    recurse_origin: tuple[Any, ...] = (tuple, list, Union),
) -> ty.Generator[type, None, None]:
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

    def __init__(self, fget):  # type: ignore
        self.fget = fget

    def __get__(self, owner_self, owner_cls):  # type: ignore
        return self.fget(owner_cls)  # type: ignore


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
def _build_delimiter_pattern(delimiters: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a tuple of delimiters into a regex expression with a capture group
    around the delimiter.
    """
    return re.compile("|".join(f"({re.escape(el)})" for el in delimiters))


############
# Iterators
############

DelimiterDictT = dict[str, tuple[DelimiterInclude, DelimiterAction]]


class Spliter:
    """Content iterator splitting according to given delimiters.

    The pattern can be changed dynamically sending a new pattern to the ty.Generator,
    see DelimiterInclude and DelimiterAction for more information.

    The current scanning position can be changed at any time.

    Parameters
    ----------
    content : str
    delimiters : dict[str, tuple[DelimiterInclude, DelimiterAction]]

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

    _pattern: Optional[re.Pattern[str]]
    _delimiters: DelimiterDictT

    __stop_searching_in_line: bool = False

    __pending: str = ""
    __first_line_col: Optional[tuple[int, int]] = None

    __lines: list[str]
    __lineno: int = 0
    __colno: int = 0

    def __init__(self, content: str, delimiters: DelimiterDictT):
        self.set_delimiters(delimiters)
        self.__lines = content.splitlines(keepends=True)

    def set_position(self, lineno: int, colno: int) -> None:
        self.__lineno, self.__colno = lineno, colno

    def set_delimiters(self, delimiters: DelimiterDictT) -> None:
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

    def __iter__(self) -> Spliter:
        return self

    def __next__(self) -> tuple[int, int, int, int, str]:
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

            if dlm is None:
                include, action = DelimiterInclude.SPLIT, DelimiterAction.STOP_PARSING
            else:
                include, action = self._delimiters[dlm]

            if include == DelimiterInclude.SPLIT:
                next_pending = ""
            else:
                # When dlm is None, DelimiterInclude.SPLIT
                assert isinstance(dlm, str)
                if include == DelimiterInclude.SPLIT_AFTER:
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

    The pattern can be changed dynamically sending a new pattern to the ty.Generator,
    see DelimiterInclude and DelimiterAction for more information.

    Parameters
    ----------
    content : str
    delimiters : dict[str, tuple[DelimiterInclude, DelimiterAction]]

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

    def set_delimiters(self, delimiters: DelimiterDictT) -> None:
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
            (start_line + 1, start_col, end_line + 1, end_col, part)  # type: ignore
        )

    def _get_next(self) -> Statement:
        if self._strip_spaces:
            return self._get_next_strip()

        part = ""
        while not part:
            start_line, start_col, end_line, end_col, part = next(self._spliter)

        return Statement.from_statement_iterator_element(
            (start_line + 1, start_col, end_line + 1, end_col, part)  # type: ignore
        )

    def peek(self, default: Any = _SENTINEL) -> Statement:
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
        return self._get_next()


###########
# Parsing
###########

# Configuration type
T = ty.TypeVar("T")
CT = ty.TypeVar("CT")
PST = ty.TypeVar("PST", bound="ParsedStatement[Any]")
LineColStr: TypeAlias = tuple[int, int, str]

ParsedResult: TypeAlias = Union[T, ParsingError]
NullableParsedResult: TypeAlias = Union[T, ParsingError, None]


class ConsumeProtocol(ty.Protocol):
    @property
    def is_position_set(self) -> bool:
        ...

    @property
    def start_line(self) -> int:
        ...

    @property
    def start_col(self) -> int:
        ...

    @property
    def end_line(self) -> int:
        ...

    @property
    def end_col(self) -> int:
        ...

    @classmethod
    def consume(
        cls, statement_iterator: StatementIterator, config: Any
    ) -> NullableParsedResult[Self]:
        ...


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
    def from_string(cls, s: str) -> NullableParsedResult[Self]:
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
    def from_string_and_config(cls, s: str, config: CT) -> NullableParsedResult[Self]:
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
        cls, statement: Statement, config: CT
    ) -> NullableParsedResult[Self]:
        raw = statement.raw
        if raw is None:
            return None

        try:
            out = cls.from_string_and_config(raw, config)
        except Exception as ex:
            out = UnhandledParsingError(ex)

        if out is None:
            return None

        out.set_position(*statement.get_position())
        out.set_raw(raw)
        return out

    @classmethod
    def consume(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> NullableParsedResult[Self]:
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


OPST = ty.TypeVar("OPST", bound="ParsedStatement[Any]")
BPST = ty.TypeVar(
    "BPST", bound="Union[ParsedStatement[Any], Block[Any, Any, Any, Any]]"
)
CPST = ty.TypeVar("CPST", bound="ParsedStatement[Any]")
RBT = ty.TypeVar("RBT", bound="RootBlock[Any, Any]")


@dataclass(frozen=True)
class Block(ty.Generic[OPST, BPST, CPST, CT], GenericInfo):
    """A sequence of statements with an opening, body and closing."""

    opening: ParsedResult[OPST]
    body: tuple[ParsedResult[BPST], ...]
    closing: Union[ParsedResult[CPST], EOS[CT]]

    delimiters: DelimiterDictT = dataclasses.field(default_factory=dict, init=False)

    def is_closed(self) -> bool:
        return not isinstance(self.closing, EOS)

    @property
    def is_position_set(self) -> bool:
        return self.opening.is_position_set

    @property
    def start_line(self) -> int:
        return self.opening.start_line

    @property
    def start_col(self) -> int:
        return self.opening.start_col

    @property
    def end_line(self) -> int:
        return self.closing.end_line

    @property
    def end_col(self) -> int:
        return self.closing.end_col

    def get_position(self) -> tuple[int, int, int, int]:
        return self.start_line, self.start_col, self.end_line, self.end_col

    @property
    def format_position(self) -> str:
        if not self.is_position_set:
            return "N/A"
        return "%d,%d-%d,%d" % self.get_position()

    def __iter__(
        self,
    ) -> ty.Generator[
        ParsedResult[Union[OPST, BPST, Union[CPST, EOS[CT]]]], None, None
    ]:
        yield self.opening
        for el in self.body:
            if isinstance(el, Block):
                yield from el
            else:
                yield el
        yield self.closing

    def iter_blocks(
        self,
    ) -> ty.Generator[ParsedResult[Union[OPST, BPST, CPST]], None, None]:
        # raise RuntimeError("Is this used?")
        yield self.opening
        yield from self.body
        yield self.closing

    ###################################################
    # Convenience methods to iterate parsed statements
    ###################################################

    _ElementT = ty.TypeVar("_ElementT", bound=Statement)

    def filter_by(
        self, klass1: type[_ElementT], *klass: type[_ElementT]
    ) -> ty.Generator[_ElementT, None, None]:
        """Yield elements of a given class or classes."""
        yield from (el for el in self if isinstance(el, (klass1,) + klass))  # type: ignore[misc]

    @cached_property
    def errors(self) -> tuple[ParsingError, ...]:
        """Tuple of errors found."""
        return tuple(self.filter_by(ParsingError))

    @property
    def has_errors(self) -> bool:
        """True if errors were found during parsing."""
        return bool(self.errors)

    ####################
    # Statement classes
    ####################

    @classmethod
    def opening_classes(cls) -> ty.Generator[type[OPST], None, None]:
        """Classes representing any of the parsed statement that can open this block."""
        try:
            opening = cls.specialization()[OPST]  # type: ignore[misc]
        except KeyError:
            opening: type = ty.get_type_hints(cls)["opening"]  # type: ignore[no-redef]
        yield from _yield_types(opening, ParsedStatement)  # type: ignore

    @classmethod
    def body_classes(cls) -> ty.Generator[type[BPST], None, None]:
        """Classes representing any of the parsed statement that can be in the body."""
        try:
            body = cls.specialization()[BPST]  # type: ignore[misc]
        except KeyError:
            body: type = ty.get_type_hints(cls)["body"]  # type: ignore[no-redef]
        yield from _yield_types(body, (ParsedStatement, Block))  # type: ignore

    @classmethod
    def closing_classes(cls) -> ty.Generator[type[CPST], None, None]:
        """Classes representing any of the parsed statement that can close this block."""
        try:
            closing = cls.specialization()[CPST]  # type: ignore[misc]
        except KeyError:
            closing: type = ty.get_type_hints(cls)["closing"]  # type: ignore[no-redef]
        yield from _yield_types(closing, ParsedStatement)  # type: ignore

    ##########
    # ParsedResult
    ##########

    @classmethod
    def consume_opening(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> NullableParsedResult[OPST]:
        """Peek into the iterator and try to parse with any of the opening classes.

        See `ParsedStatement.consume` for more details.
        """
        for c in cls.opening_classes():
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        return None

    @classmethod
    def consume_body(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> ParsedResult[BPST]:
        """Peek into the iterator and try to parse with any of the body classes.

        If the statement cannot be parsed, a UnknownStatement is returned.
        """
        for c in cls.body_classes():
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        unkel = next(statement_iterator)
        return UnknownStatement.from_statement(unkel)

    @classmethod
    def consume_closing(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> NullableParsedResult[CPST]:
        """Peek into the iterator and try to parse with any of the opening classes.

        See `ParsedStatement.consume` for more details.
        """
        for c in cls.closing_classes():
            el = c.consume(statement_iterator, config)
            if el is not None:
                return el
        return None

    @classmethod
    def consume_body_closing(
        cls, opening: OPST, statement_iterator: StatementIterator, config: CT
    ) -> Self:
        body: list[ParsedResult[BPST]] = []
        closing: ty.Union[CPST, ParsingError, None] = None
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
                unexpected_end = cls.on_stop_iteration(config)
                unexpected_end.set_position(last_line + 1, 0, last_line + 1, 0)
                return cls(opening, tuple(body), unexpected_end)

        return cls(opening, tuple(body), closing)

    @classmethod
    def consume(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> Union[Self, None]:
        """Try consume the block.

        Possible outcomes:
        1. The opening was not matched, return None.
        2. A subclass of Block, where body and closing migh contain errors.
        """
        opening = cls.consume_opening(statement_iterator, config)
        if opening is None:
            return None

        if isinstance(opening, ParsingError):
            return None

        return cls.consume_body_closing(opening, statement_iterator, config)

    @classmethod
    def on_stop_iteration(cls, config: CT) -> ParsedResult[EOS[CT]]:
        return UnexpectedEOS()


@dataclass(frozen=True)
class BOS(ty.Generic[CT], ParsedStatement[CT]):
    """Beginning of source."""

    # Hasher algorithm name and hexdigest
    content_hash: Hash

    @classmethod
    def from_string_and_config(cls, s: str, config: CT) -> NullableParsedResult[Self]:
        raise RuntimeError("BOS cannot be constructed from_string_and_config")

    @property
    def location(self) -> SourceLocationT:
        return "<undefined>"


@dataclass(frozen=True)
class BOF(ty.Generic[CT], BOS[CT]):
    """Beginning of file."""

    path: pathlib.Path

    # Modification time of the file.
    mtime: float

    @property
    def location(self) -> SourceLocationT:
        return self.path


@dataclass(frozen=True)
class BOR(ty.Generic[CT], BOS[CT]):
    """Beginning of resource."""

    package: str
    resource_name: str

    @property
    def location(self) -> SourceLocationT:
        return self.package, self.resource_name


@dataclass(frozen=True)
class EOS(ty.Generic[CT], ParsedStatement[CT]):
    """End of sequence."""

    @classmethod
    def from_string_and_config(
        cls: type[PST], s: str, config: CT
    ) -> NullableParsedResult[PST]:
        return cls()


class RootBlock(ty.Generic[BPST, CT], Block[BOS[CT], BPST, EOS[CT], CT]):
    """A sequence of statement flanked by the beginning and ending of stream."""

    @classmethod
    def consume_opening(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> NullableParsedResult[BOS[CT]]:
        raise RuntimeError(
            "Implementation error, 'RootBlock.consume_opening' should never be called"
        )

    @classmethod
    def consume(cls, statement_iterator: StatementIterator, config: CT) -> Self:
        block = super().consume(statement_iterator, config)
        if block is None:
            raise RuntimeError(
                "Implementation error, 'RootBlock.consume' should never return None"
            )
        return block

    @classmethod
    def consume_closing(
        cls, statement_iterator: StatementIterator, config: CT
    ) -> NullableParsedResult[EOS[CT]]:
        return None

    @classmethod
    def on_stop_iteration(cls, config: CT) -> ParsedResult[EOS[CT]]:
        return EOS[CT]()


#################
# Source parsing
#################

ResourceT: TypeAlias = tuple[str, str]  # package name, resource name
StrictLocationT: TypeAlias = Union[pathlib.Path, ResourceT]
SourceLocationT: TypeAlias = Union[str, StrictLocationT]


@dataclass(frozen=True)
class ParsedSource(ty.Generic[RBT, CT]):
    parsed_source: RBT

    # Parser configuration.
    config: CT

    @property
    def location(self) -> SourceLocationT:
        if isinstance(self.parsed_source.opening, ParsingError):
            raise self.parsed_source.opening
        return self.parsed_source.opening.location

    @cached_property
    def has_errors(self) -> bool:
        return self.parsed_source.has_errors

    def errors(self) -> ty.Generator[ParsingError, None, None]:
        yield from self.parsed_source.errors


@dataclass(frozen=True)
class CannotParseResourceAsFile(Exception):
    """The requested python package resource cannot be located as a file
    in the file system.
    """

    package: str
    resource_name: str


class Parser(ty.Generic[RBT, CT], GenericInfo):
    """Parser class."""

    #: class to iterate through statements in a source unit.
    _statement_iterator_class: type[StatementIterator] = StatementIterator

    #: Delimiters.
    _delimiters: DelimiterDictT = SPLIT_EOL

    _strip_spaces: bool = True

    #: source file text encoding.
    _encoding: str = "utf-8"

    #: configuration passed to from_string functions.
    _config: CT

    #: try to open resources as files.
    _prefer_resource_as_file: bool

    #: parser algorithm to us. Must be a callable member of hashlib
    _hasher: ty.Callable[
        [
            bytes,
        ],
        HasherProtocol,
    ] = hashlib.blake2b

    def __init__(self, config: CT, prefer_resource_as_file: bool = True):
        self._config = config
        self._prefer_resource_as_file = prefer_resource_as_file

    @classmethod
    def root_boot_class(cls) -> type[RBT]:
        """Class representing the root block class."""
        try:
            return cls.specialization()[RBT]  # type: ignore[misc]
        except KeyError:
            return ty.get_type_hints(cls)["root_boot_class"]  # type: ignore[no-redef]

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

    def parse_bytes(
        self, b: bytes, bos: Optional[BOS[CT]] = None
    ) -> ParsedSource[RBT, CT]:
        if bos is None:
            bos = BOS[CT](Hash.from_bytes(self._hasher, b)).set_simple_position(0, 0, 0)

        sic = self._statement_iterator_class(
            b.decode(self._encoding), self._delimiters, self._strip_spaces
        )

        parsed = self.root_boot_class().consume_body_closing(bos, sic, self._config)

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

        bos = BOF[CT](
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
        with resources.as_file(resources.files(package).joinpath(resource_name)) as p:
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
        with resources.files(package).joinpath(resource_name).open("rb") as fi:
            content = fi.read()

        bos = BOR[CT](
            Hash.from_bytes(self._hasher, content), package, resource_name
        ).set_simple_position(0, 0, 0)

        return self.parse_bytes(content, bos)


##########
# Project
##########


class IncludeStatement(ty.Generic[CT], ParsedStatement[CT]):
    """ "Include statements allow to merge files."""

    @property
    def target(self) -> str:
        raise NotImplementedError(
            "IncludeStatement subclasses must implement target property."
        )


class ParsedProject(
    ty.Generic[RBT, CT],
    dict[
        Optional[tuple[StrictLocationT, str]],
        ParsedSource[RBT, CT],
    ],
):
    """Collection of files, independent or connected via IncludeStatement.

    Keys are either an absolute pathname  or a tuple package name, resource name.

    None is the name of the root.

    """

    @cached_property
    def has_errors(self) -> bool:
        return any(el.has_errors for el in self.values())

    def errors(self) -> ty.Generator[ParsingError, None, None]:
        for el in self.values():
            yield from el.errors()

    def _iter_statements(
        self,
        items: ty.Iterable[tuple[Any, Any]],
        seen: set[Any],
        include_only_once: bool,
    ) -> ty.Generator[ParsedStatement[CT], None, None]:
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

    def iter_statements(
        self, include_only_once: bool = True
    ) -> ty.Generator[ParsedStatement[CT], None, None]:
        """Iter all definitions in the order they appear,
        going into the included files.

        Parameters
        ----------
        include_only_once
            if true, each file cannot be included more than once.
        """
        yield from self._iter_statements([(None, self[None])], set(), include_only_once)

    def _iter_blocks(
        self,
        items: ty.Iterable[tuple[Any, Any]],
        seen: set[Any],
        include_only_once: bool,
    ) -> ty.Generator[ParsedStatement[CT], None, None]:
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

    def iter_blocks(
        self, include_only_once: bool = True
    ) -> ty.Generator[ParsedStatement[CT], None, None]:
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
        if not tmp.is_relative_to(current_path):
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


@no_type_check
def _build_root_block_class_parsed_statement(
    spec: type[ParsedStatement[CT]], config: type[CT]
) -> type[RootBlock[ParsedStatement[CT], CT]]:
    """Build root block class from a single ParsedStatement."""

    @dataclass(frozen=True)
    class CustomRootBlockA(RootBlock[spec, config]):  # type: ignore
        pass

    return CustomRootBlockA


@no_type_check
def _build_root_block_class_block(
    spec: type[Block[OPST, BPST, CPST, CT]],
    config: type[CT],
) -> type[RootBlock[Block[OPST, BPST, CPST, CT], CT]]:
    """Build root block class from a single ParsedStatement."""

    @dataclass(frozen=True)
    class CustomRootBlockA(RootBlock[spec, config]):  # type: ignore
        pass

    return CustomRootBlockA


@no_type_check
def _build_root_block_class_parsed_statement_it(
    spec: tuple[type[Union[ParsedStatement[CT], Block[OPST, BPST, CPST, CT]]]],
    config: type[CT],
) -> type[RootBlock[ParsedStatement[CT], CT]]:
    """Build root block class from iterable ParsedStatement."""

    @dataclass(frozen=True)
    class CustomRootBlockA(RootBlock[Union[spec], config]):  # type: ignore
        pass

    return CustomRootBlockA


@no_type_check
def _build_parser_class_root_block(
    spec: type[RootBlock[BPST, CT]],
    *,
    strip_spaces: bool = True,
    delimiters: Optional[DelimiterDictT] = None,
) -> type[Parser[RootBlock[BPST, CT], CT]]:
    class CustomParser(Parser[spec, spec.specialization()[CT]]):  # type: ignore
        _delimiters: DelimiterDictT = delimiters or SPLIT_EOL
        _strip_spaces: bool = strip_spaces

    return CustomParser


@no_type_check
def build_parser_class(
    spec: Union[
        type[
            Union[
                Parser[RBT, CT],
                RootBlock[BPST, CT],
                Block[OPST, BPST, CPST, CT],
                ParsedStatement[CT],
            ]
        ],
        ty.Iterable[type[ParsedStatement[CT]]],
    ],
    config: CT = None,
    strip_spaces: bool = True,
    delimiters: Optional[DelimiterDictT] = None,
) -> type[
    Union[
        Parser[RBT, CT],
        Parser[RootBlock[BPST, CT], CT],
        Parser[RootBlock[Block[OPST, BPST, CPST, CT], CT], CT],
    ]
]:
    """Build a custom parser class.

    Parameters
    ----------
    spec
        RootBlock derived class.
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

    if isinstance(spec, type):
        if issubclass(spec, Parser):
            CustomParser = spec

        elif issubclass(spec, RootBlock):
            CustomParser = _build_parser_class_root_block(
                spec, strip_spaces=strip_spaces, delimiters=delimiters
            )

        elif issubclass(spec, Block):
            CustomRootBlock = _build_root_block_class_block(spec, config.__class__)
            CustomParser = _build_parser_class_root_block(
                CustomRootBlock, strip_spaces=strip_spaces, delimiters=delimiters
            )

        elif issubclass(spec, ParsedStatement):
            CustomRootBlock = _build_root_block_class_parsed_statement(
                spec, config.__class__
            )
            CustomParser = _build_parser_class_root_block(
                CustomRootBlock, strip_spaces=strip_spaces, delimiters=delimiters
            )

        else:
            raise TypeError(
                "`spec` must be of type Parser, Block, RootBlock or tuple of type Block or ParsedStatement, "
                f"not {type(spec)}"
            )

    elif isinstance(spec, (tuple, list)):
        CustomRootBlock = _build_root_block_class_parsed_statement_it(
            spec, config.__class__
        )
        CustomParser = _build_parser_class_root_block(
            CustomRootBlock, strip_spaces=strip_spaces, delimiters=delimiters
        )

    else:
        raise

    return CustomParser


@no_type_check
def parse(
    entry_point: SourceLocationT,
    spec: Union[
        type[
            Union[
                Parser[RBT, CT],
                RootBlock[BPST, CT],
                Block[OPST, BPST, CPST, CT],
                ParsedStatement[CT],
            ]
        ],
        ty.Iterable[type[ParsedStatement[CT]]],
    ],
    config: CT = None,
    *,
    strip_spaces: bool = True,
    delimiters: Optional[DelimiterDictT] = None,
    locator: ty.Callable[[SourceLocationT, str], StrictLocationT] = default_locator,
    prefer_resource_as_file: bool = True,
    **extra_parser_kwargs: Any,
) -> Union[ParsedProject[RBT, CT], ParsedProject[RootBlock[BPST, CT], CT]]:
    """Parse sources into a ParsedProject dictionary.

    Parameters
    ----------
    entry_point
        file or resource, given as (package_name, resource_name).
    spec
        specification of the content to parse. Can be one of the following things:
        - Parser class.
        - Block or ParsedStatement derived class.
        - ty.Iterable of Block or ParsedStatement derived class.
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

    CustomParser = build_parser_class(spec, config, strip_spaces, delimiters)
    parser = CustomParser(
        config, prefer_resource_as_file=prefer_resource_as_file, **extra_parser_kwargs
    )

    pp = ParsedProject()

    pending: list[tuple[SourceLocationT, str]] = []
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


@no_type_check
def parse_bytes(
    content: bytes,
    spec: Union[
        type[
            Union[
                Parser[RBT, CT],
                RootBlock[BPST, CT],
                Block[OPST, BPST, CPST, CT],
                ParsedStatement[CT],
            ]
        ],
        ty.Iterable[type[ParsedStatement[CT]]],
    ],
    config: Optional[CT] = None,
    *,
    strip_spaces: bool,
    delimiters: Optional[DelimiterDictT],
    **extra_parser_kwargs: Any,
) -> ParsedProject[
    Union[RBT, RootBlock[BPST, CT], RootBlock[ParsedStatement[CT], CT]], CT
]:
    """Parse sources into a ParsedProject dictionary.

    Parameters
    ----------
    content
        bytes.
    spec
        specification of the content to parse. Can be one of the following things:
        - Parser class.
        - Block or ParsedStatement derived class.
        - ty.Iterable of Block or ParsedStatement derived class.
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

    CustomParser = build_parser_class(spec, config, strip_spaces, delimiters)

    parser = CustomParser(config, prefer_resource_as_file=False, **extra_parser_kwargs)

    pp = ParsedProject()

    pp[None] = parsed = parser.parse_bytes(content)

    if any(parsed.parsed_source.filter_by(IncludeStatement)):
        raise ValueError("parse_bytes does not support using an IncludeStatement")

    return pp
