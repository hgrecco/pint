# -*- coding: utf-8 -*-
"""
    pint.systems
    ~~~~~~~~~~~~

    Functions and classes related to system definitions and conversions.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import

import re

from .unit import Definition, UnitDefinition, DefinitionSyntaxError


class Group(object):
    """A group is a list of units.

    Units can be added directly or by including other groups.

    Members are computed dynamically, that is if a unit is added to a group X
    all groups that include X are affected.
    """

    #: Regex to match the header parts of a context.
    _header_re = re.compile('@group\s+(?P<name>\w+)\s*(using\s(?P<groups>.*))*')

    def __init__(self, name, parent):
        """

        :type name: str
        :type parent: Group | None
        """

        # The name of the group.
        #: type: str
        self.name = name

        #: Names of the units in this group.
        #: :type: set[str]
        self._unit_names = set()

        #: Names of the groups in this group.
        #: :type: set[str]
        self._child_groups = set()

        #: Names of the groups in which this group is contained.
        #: :type: set[str]
        self._parent_groups = set()

        if parent is None:
            #: Maps group name to Group.
            #: If None, the Group has not been assigned to a Registry yet.
            #: :type: dict[str, Group] | None
            self._group_dict = {self.name: self}
        else:
            self._group_dict = parent._group_dict

            if self.name in self._group_dict:
                raise ValueError('Group name already in use')

            self._group_dict[self.name] = self
            self._parent_groups.add(parent.name)
            parent._child_groups.add(self.name)

        #: A cache of the included units.
        #: None indicates that the cache has been invalidated.
        #: :type: frozenset[str] | None
        self._computed_members = None

    @property
    def members(self):
        """Names of the units that are members of the group.

        Calculated to include to all units in all included groups.

        :rtype: frozenset[str]
        """
        if self._computed_members is None:
            self._computed_members = set(self._unit_names)

            for _, group in self.iter_child_groups():
                self._computed_members |= group.members

            self._computed_members = frozenset(self._computed_members)

        return self._computed_members

    def invalidate_members(self):
        """Invalidate computed members in this Group and all parent nodes.
        """
        self._computed_members = None
        d = self._group_dict
        for name in self._parent_groups:
            d[name].invalidate_members()

    def iter_child_groups(self):
        pending = set(self._child_groups)
        d = self._group_dict
        while pending:
            name = pending.pop()
            group = d[name]
            pending |= group._child_groups
            yield name, d[name]

    def is_child_group(self, group_name):
        for name, _ in self.iter_child_groups():
            if name == group_name:
                return True
        return False

    def add_units(self, *unit_names):
        """Add units to group.

        :type unit_names: str
        """
        for unit_name in unit_names:
            self._unit_names.add(unit_name)

        self.invalidate_members()

    def remove_units(self, *unit_names):
        """Remove units from group.

        :type unit_names: str
        """
        for unit_name in unit_names:
            self._unit_names.remove(unit_name)

        self.invalidate_members()

    def add_groups(self, *group_names):
        """Add groups to group.

        :type group_names: str
        """
        d = self._group_dict
        for group_name in group_names:
            grp = d[group_name]
            if self.is_child_group(group_name):
                raise ValueError('Cyclic relationship found between %s and %s' % (self.name, group_name))
            self._child_groups.add(group_name)

            grp._parent_groups.add(self.name)

        self.invalidate_members()

    def remove_groups(self, *group_names):
        """Remove groups from group.

        :type group_names: str
        """
        d = self._group_dict
        for group_name in group_names:
            grp = d[group_name]

            self._child_groups.remove(group_name)
            grp._parent_groups.remove(self.name)


        self.invalidate_members()

    @classmethod
    def from_lines(cls, lines, define_func, parent_group):
        """Return a Group object parsing an iterable of lines.

        :param lines: iterable
        :type lines: collections.Iterable[str]
        :param parent_group: parent group of this Group.
        :type parent_group: Group
        :param define_func: Function to define a unit in the registry.
        :type define_func: str -> None
        """
        header, lines = lines[0], lines[1:]

        r = cls._header_re.search(header)
        name = r.groupdict()['name'].strip()
        groups = r.groupdict()['groups']
        if groups:
            group_names = tuple(a.strip() for a in groups.split(','))
        else:
            group_names = ()

        unit_names = []
        for line in lines:
            if '=' in line:
                # Is a definition
                definition = Definition.from_string(line)
                if not isinstance(Definition, UnitDefinition):
                    raise DefinitionSyntaxError('Only UnitDefinitions are valid inside groups, '
                                                'not %s' % definition.__class__.__name)
                define_func(definition)
                unit_names.append(definition.name)
            else:
                unit_names.append(line.strip())

        grp = cls(name, parent_group)

        grp.add_units(*unit_names)

        if group_names:
            grp.add_groups(*group_names)

        return grp


class System(object):
    """A system is a Group plus a set of base units.

    @system <name> [using <group 1>, ..., <group N>]
        <rule 1>
        ...
        <rule N>
    @end

    """

    #: Regex to match the header parts of a context.
    _header_re = re.compile('@system\s+(?P<name>\w+)\s*(using\s(?P<groups>.*))*')

    def __init__(self, name, groups):

        #: N
        #: :type: str
        self.name = name

        #: Maps base unit names to old unit names
        #: None indicates that it replaces the standard unit.
        #: :type: dict[str, str | None]
        self.base_units = {}

        #: Derived unit names.
        #: :type: set(str)
        self.derived_units = set()

        self.groups = groups

        #: :type: frozenset | None
        self._computed_members = None

    @property
    def members(self):
        if self._computed_members is None:
            self._computed_members = set()

            for group_name in self.groups:
                self._computed_members |= _groups[group_name].members

            self._computed_members = frozenset(self._computed_members)

        return self._computed_members

    @classmethod
    def from_lines(cls, lines):
        header, lines = lines[0], lines[1:]

        r = cls._header_re.search(header)
        name = r.groupdict()['name'].strip()
        groups = r.groupdict()['groups']
        if groups:
            group_names = tuple(a.strip() for a in groups.split(','))
        else:
            group_names = ()

        base_unit_names = {}
        derived_unit_names = []
        for line in lines:
            line = line.strip()
            if line[0] == '+':
                # Type 2, derived dimension
                derived_unit_names.append(line[1:].strip())
            elif ':' in line:
                old_unit, new_unit = line.split(':')
                base_unit_names[old_unit.strip()] = new_unit.strip()
            else:
                derived_unit_names.append(line.strip())

        system = cls(name, group_names)

        system.base_units.update(**base_unit_names)
        system.derived_units |= derived_unit_names

        return system


# These dictionaries will be part of the registry

#: :type: dict[str, Group]
_groups = dict()

#: :type: dict[str, System]
_systems = dict()


# These methods will be included in the registry, upgrading the existing ones.
# current get_base_units will be renamed to get_root_units
#
# Not sure yet how to deal with the cache.
# Should we cache get_root_units, get_base_units or both?
# - get_base_units will need to be invalidated when the system is changed (How often will this happen?)
# - get_root_units will not need to be invalidated.


def get_group(registry, name, create_if_needed=True):
    """Return a Group.


    :param registry:
    :param name: Name of the group to be
    :param create_if_needed: Create a group if not Found. If False, raise an Exception.
    :return: Group
    """
    if name == 'default':
        raise ValueError('The name default is reserved.')

    try:
        return _groups[name]
    except KeyError:
        if create_if_needed:
            return Group(name, parent=_groups['default'])
        else:
            raise KeyError('No group %s found.' % name)


def get_compatible_units(registry, input_units, group_or_system=None):
    """
    :param registry:
    :param input_units:
    :param group_or_system:
    :type group_or_system: Group | System
    :return:
    """
    ret = registry.get_compatible_units(input_units)

    if not group_or_system:
        return ret

    members = _groups[group_or_system].members
    return frozenset(r for r in ret
                     if r in members)


def get_base_units(registry, input_units, check_nonmult=True, system=None):
    """
    :param registry:
    :param input_units:
    :param check_nonmult:
    :param system: System
    :return:
    """
    factor, units = registry.get_base_units(input_units, check_nonmult)

    if not system:
        return factor, units

    destination_units = {}

    bu = system.base_units
    for unit in units:
        destination_units[bu.get(unit, unit)] = 1

    base_factor = registry.convert(factor, units, destination_units)

    return base_factor, destination_units
