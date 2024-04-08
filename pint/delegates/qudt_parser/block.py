"""
    pint.delegates.txt_defparser.block
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Classes for Pint Blocks, which are defined by:

        @<block name>
            <content>
        @end

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import flexparser as fp

from ..base_defparser import ParserConfig, PintParsedStatement


@dataclass(frozen=True)
class BeginDirectiveBlock(fp.ParsedStatement):
    """An BeginDirectiveBlock is simply a "." statement.
    example statement: 'unit:A'
    example _object_type: 'unit'

    """

    _object_type = str

    @classmethod
    def from_string(cls, s):
        if s[: len(cls._object_type)] == cls._object_type:
            obj = cls()
            obj.name = s[len(cls._object_type) + 1 :]
            return obj

        return None


@dataclass(frozen=True)
class EndDirectiveBlock(PintParsedStatement):
    """An EndDirectiveBlock is simply a "." statement."""

    @classmethod
    def from_string(cls, s: str) -> fp.NullableParsedResult[EndDirectiveBlock]:
        if s == ".":
            return cls()
        return None


OPST = TypeVar("OPST", bound="PintParsedStatement")  # Opening Parsed Statement Type
IPST = TypeVar("IPST", bound="PintParsedStatement")  # ? Inner Parsed Statement Type

DefT = TypeVar("DefT")  # Definition? Type


@dataclass(frozen=True)
class DirectiveBlock(
    Generic[DefT, OPST, IPST], fp.Block[OPST, IPST, EndDirectiveBlock, ParserConfig]
):
    """Directive blocks have beginning statement Begining with a @ character.
    and ending with a "@end" (captured using a EndDirectiveBlock).

    Subclass this class for convenience.
    """

    # is this needed below?
    def derive_definition(self) -> DefT:
        ...
