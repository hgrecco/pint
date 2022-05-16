"""
    pint.facets.systems.definitions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""


from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from ...util import SourceIterator


@dataclass(frozen=True)
class SystemDefinition:
    """Definition of a System:

        @system <name> [using <group 1>, ..., <group N>]
            <rule 1>
            ...
            <rule N>
        @end

    The syntax for the rule is:

        new_unit_name : old_unit_name

    where:
        - old_unit_name: a root unit part which is going to be removed from the system.
        - new_unit_name: a non root unit which is going to replace the old_unit.

    If the new_unit_name and the old_unit_name, the later and the colon can be omitted.
    """

    #: Regex to match the header parts of a context.
    _header_re = re.compile(r"@system\s+(?P<name>\w+)\s*(using\s(?P<used_groups>.*))*")

    name: str
    unit_replacements: Tuple[Tuple[int, str, str], ...]
    using_group_names: Tuple[str, ...]

    @classmethod
    def from_lines(cls, lines, non_int_type=float):
        lines = SourceIterator(lines)

        lineno, header = next(lines)

        r = cls._header_re.search(header)

        if r is None:
            raise ValueError("Invalid System header syntax '%s'" % header)

        name = r.groupdict()["name"].strip()
        groups = r.groupdict()["used_groups"]

        # If the systems has no group, it automatically uses the root group.
        if groups:
            group_names = tuple(a.strip() for a in groups.split(","))
        else:
            group_names = ("root",)

        unit_replacements = []
        for lineno, line in lines:
            line = line.strip()

            # We would identify a
            #  - old_unit: a root unit part which is going to be removed from the system.
            #  - new_unit: a non root unit which is going to replace the old_unit.

            if ":" in line:
                # The syntax is new_unit:old_unit

                new_unit, old_unit = line.split(":")
                new_unit, old_unit = new_unit.strip(), old_unit.strip()

                unit_replacements.append((lineno, new_unit, old_unit))
            else:
                # The syntax is new_unit
                # old_unit is inferred as the root unit with the same dimensionality.
                unit_replacements.append((lineno, line, None))

        return cls(name, tuple(unit_replacements), group_names)
