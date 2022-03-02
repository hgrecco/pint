"""
    pint.facets.group.defintions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from ...definitions import Definition
from ...errors import DefinitionSyntaxError
from ...util import SourceIterator
from ..base.definitions import UnitDefinition


@dataclass(frozen=True)
class GroupDefinition:
    """Definition of a group.

        @group <name> [using <group 1>, ..., <group N>]
            <definition 1>
            ...
            <definition N>
        @end

    Example::

        @group AvoirdupoisUS using Avoirdupois
            US_hundredweight = hundredweight = US_cwt
            US_ton = ton
            US_force_ton = force_ton = _ = US_ton_force
        @end

    """

    #: Regex to match the header parts of a definition.
    _header_re = re.compile(r"@group\s+(?P<name>\w+)\s*(using\s(?P<used_groups>.*))*")

    name: str
    units: Tuple[Tuple[int, UnitDefinition], ...]
    using_group_names: Tuple[str, ...]

    @property
    def unit_names(self) -> Tuple[str, ...]:
        return tuple(u.name for lineno, u in self.units)

    @classmethod
    def from_lines(cls, lines, non_int_type=float):
        """Return a Group object parsing an iterable of lines.

        Parameters
        ----------
        lines : list[str]
            iterable
        define_func : callable
            Function to define a unit in the registry; it must accept a single string as
            a parameter.

        Returns
        -------

        """

        lines = SourceIterator(lines)
        lineno, header = next(lines)

        r = cls._header_re.search(header)

        if r is None:
            raise ValueError("Invalid Group header syntax: '%s'" % header)

        name = r.groupdict()["name"].strip()
        groups = r.groupdict()["used_groups"]
        if groups:
            parent_group_names = tuple(a.strip() for a in groups.split(","))
        else:
            parent_group_names = ()

        units = []
        for lineno, line in lines:
            definition = Definition.from_string(line, non_int_type=non_int_type)
            if not isinstance(definition, UnitDefinition):
                raise DefinitionSyntaxError(
                    "Only UnitDefinition are valid inside _used_groups, not "
                    + str(definition),
                    lineno=lineno,
                )
            units.append((lineno, definition))

        return cls(name, tuple(units), parent_group_names)
