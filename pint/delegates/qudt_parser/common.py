"""
    pint.delegates.txt_defparser.common
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Definitions for parsing an Import Statement

    Also DefinitionSyntaxError

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import flexparser as fp

from ... import errors
from ...util import UnitsContainer
from ..base_defparser import ParserConfig
from . import block


@dataclass(frozen=True)
class DefinitionSyntaxError(errors.DefinitionSyntaxError, fp.ParsingError):
    """A syntax error was found in a definition. Combines:

    DefinitionSyntaxError: which provides a message placeholder.
    fp.ParsingError: which provides raw text, and start and end column and row

    and an extra location attribute in which the filename or reseource is stored.
    """

    location: str = field(init=False, default="")

    def __str__(self) -> str:
        msg = (
            self.msg + "\n    " + (self.format_position or "") + " " + (self.raw or "")
        )
        if self.location:
            msg += "\n    " + self.location
        return msg

    def set_location(self, value: str) -> None:
        super().__setattr__("location", value)


@dataclass(frozen=True)
class ImportDefinition(fp.IncludeStatement[ParserConfig]):
    value: str

    @property
    def target(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[ImportDefinition]:
        if s.startswith("@import"):
            return ImportDefinition(s[len("@import") :].strip())
        return None


###########################################


@dataclass(frozen=True)
class Attribute(fp.ParsedStatement):
    """Parses the following `qudt:applicableSystem sou:CGS-EMU ;`"""

    parent: str
    attr: str
    value: str

    @classmethod
    def from_string(cls, s):
        if ":" not in s and ";" not in s:
            # This means: I do not know how to parse it
            # try with another ParsedStatement class.
            return None
        s = s[:-1].strip()
        parent, s = s.split(":", 1)
        if " " not in s:
            attr, value = s, ""
        else:
            attr, value = s.split(" ", 1)

        # if not str.isidentifier(lhs):
        #     return InvalidIdentifier(lhs)

        return cls(parent, attr, value)


class BeginObject(fp.ParsedStatement):
    _object_type = str

    @classmethod
    def from_string(cls, s):
        if s[: len(cls._object_type)] == cls._object_type:
            return cls()

        return None


class BeginHeader(fp.ParsedStatement):
    @classmethod
    def from_string(cls, s):
        if s[:9] == "# baseURI":
            return cls()

        return None


class End(fp.ParsedStatement):
    @classmethod
    def from_string(cls, s):
        if s == ".":
            return cls()

        return None


class HeaderLine(fp.ParsedStatement):
    # couldnt get BOF to work in place of this in HeaderBlock
    def from_string(cls, s):
        return cls()


@dataclass(frozen=True)
class HeaderBlock(fp.Block[BeginHeader, HeaderLine, End, ParserConfig]):
    pass


DIMENSIONS = "substance, current, length, luminousity, mass, temperature, time, dimensionless".split(
    ", "
)
BASE_UNITS = "mol, A, m, cd, kg, K, s".split(", ")


class BeginVoag(block.BeginDirectiveBlock):
    _object_type = "voag"


@dataclass(frozen=True)
class VoagDefinitionBlock(
    fp.Block[BeginVoag, Attribute, block.EndDirectiveBlock, ParserConfig]
):
    pass


class BeginVaem(block.BeginDirectiveBlock):
    _object_type = "vaem"


@dataclass(frozen=True)
class VaemDefinitionBlock(
    fp.Block[BeginVaem, Attribute, block.EndDirectiveBlock, ParserConfig]
):
    pass


class BeginHttp(block.BeginDirectiveBlock):
    _object_type = "<http"


@dataclass(frozen=True)
class HttpDefinitionBlock(
    fp.Block[BeginHttp, Attribute, block.EndDirectiveBlock, ParserConfig]
):
    pass


def make_units_container(dimensionvector: str) -> UnitsContainer:
    # use regex to split the dimension vector 'A0E1L0I0M0H0T0D0' into a dictionary of units
    if ":" in dimensionvector:
        dimensionvector = dimensionvector.split(":")[1]
    dimensionvector = re.sub(r"[a-zA-Z]", r" ", dimensionvector).split()
    units = {unit: int(exponent) for unit, exponent in zip(BASE_UNITS, dimensionvector)}
    return UnitsContainer(units)
