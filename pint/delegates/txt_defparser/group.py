"""
    pint.delegates.txt_defparser.group
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Definitions for parsing Group and their related objects

    Notices that some of the checks are done within the
    format agnostic parent definition class.

    See each one for a slighly longer description of the
    syntax.

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import re
import typing as ty
from dataclasses import dataclass

from ..._vendor import flexparser as fp
from ...facets.group import definitions
from . import block, common, plain


@dataclass(frozen=True)
class BeginGroup(fp.ParsedStatement):
    """Being of a group directive.

    @group <name> [using <group 1>, ..., <group N>]
    """

    #: Regex to match the header parts of a definition.
    _header_re = re.compile(r"@group\s+(?P<name>\w+)\s*(using\s(?P<used_groups>.*))*")

    name: str
    using_group_names: ty.Tuple[str, ...]

    @classmethod
    def from_string(cls, s: str) -> fp.FromString[BeginGroup]:
        if not s.startswith("@group"):
            return None

        r = cls._header_re.search(s)

        if r is None:
            return common.DefinitionSyntaxError(f"Invalid Group header syntax: '{s}'")

        name = r.groupdict()["name"].strip()
        groups = r.groupdict()["used_groups"]
        if groups:
            parent_group_names = tuple(a.strip() for a in groups.split(","))
        else:
            parent_group_names = ()

        return cls(name, parent_group_names)


@dataclass(frozen=True)
class GroupDefinition(block.DirectiveBlock):
    """Definition of a group.

        @group <name> [using <group 1>, ..., <group N>]
            <definition 1>
            ...
            <definition N>
        @end

    See UnitDefinition and Comment for more parsing related information.

    Example::

        @group AvoirdupoisUS using Avoirdupois
            US_hundredweight = hundredweight = US_cwt
            US_ton = ton
            US_force_ton = force_ton = _ = US_ton_force
        @end

    """

    opening: fp.Single[BeginGroup]
    body: fp.Multi[
        ty.Union[
            plain.CommentDefinition,
            plain.UnitDefinition,
        ]
    ]

    def derive_definition(self):
        return definitions.GroupDefinition(
            self.name, self.using_group_names, self.definitions
        )

    @property
    def name(self):
        return self.opening.name

    @property
    def using_group_names(self):
        return self.opening.using_group_names

    @property
    def definitions(self) -> ty.Tuple[plain.UnitDefinition, ...]:
        return tuple(el for el in self.body if isinstance(el, plain.UnitDefinition))
