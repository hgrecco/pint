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
from .util import to_units_container, UnitsContainer


class Group(object):
    """A group is a list of units.

    Units can be added directly or by including other groups.

    Members are computed dynamically, that is if a unit is added to a group X
    all groups that include X are affected.
    """

    #: Regex to match the header parts of a context.
    _header_re = re.compile('@group\s+(?P<name>\w+)\s*(using\s(?P<used_groups>.*))*')

    def __init__(self, name, groups_systems):
        """
        :param name: Name of the group
        :type name: str
        :param groups_systems: dictionary containing groups and system.
                               The newly created group will be added after creation.
        :type groups_systems: dict[str, Group | System]
        """

        if name in groups_systems:
            t = 'group' if isinstance(groups_systems['name'], Group) else 'system'
            raise ValueError('The system name already in use by a %s' % t)

        # The name of the group.
        #: type: str
        self.name = name

        #: Names of the units in this group.
        #: :type: set[str]
        self._unit_names = set()

        #: Names of the groups in this group.
        #: :type: set[str]
        self._used_groups = set()

        #: Names of the groups in which this group is contained.
        #: :type: set[str]
        self._used_by = set()

        #: Maps group name to Group.
        #: :type: dict[str, Group]
        self._groups_systems = groups_systems

        self._groups_systems[self.name] = self

        if name != 'root':
            # All groups are added to root group
            groups_systems['root'].add_groups(name)

        #: A cache of the included units.
        #: None indicates that the cache has been invalidated.
        #: :type: frozenset[str] | None
        self._computed_members = None

    @property
    def members(self):
        """Names of the units that are members of the group.

        Calculated to include to all units in all included _used_groups.

        :rtype: frozenset[str]
        """
        if self._computed_members is None:
            self._computed_members = set(self._unit_names)

            for _, group in self.iter_used_groups():
                self._computed_members |= group.members

            self._computed_members = frozenset(self._computed_members)

        return self._computed_members

    def invalidate_members(self):
        """Invalidate computed members in this Group and all parent nodes.
        """
        self._computed_members = None
        d = self._groups_systems
        for name in self._used_by:
            d[name].invalidate_members()

    def iter_used_groups(self):
        pending = set(self._used_groups)
        d = self._groups_systems
        while pending:
            name = pending.pop()
            group = d[name]
            pending |= group._used_groups
            yield name, d[name]

    def is_used_group(self, group_name):
        for name, _ in self.iter_used_groups():
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
        d = self._groups_systems
        for group_name in group_names:

            grp = d[group_name]

            if grp.is_used_group(self.name):
                raise ValueError('Cyclic relationship found between %s and %s' % (self.name, group_name))

            self._used_groups.add(group_name)
            grp._used_by.add(self.name)

        self.invalidate_members()

    def remove_groups(self, *group_names):
        """Remove groups from group.

        :type group_names: str
        """
        d = self._groups_systems
        for group_name in group_names:
            grp = d[group_name]

            self._used_groups.remove(group_name)
            grp._used_by.remove(self.name)


        self.invalidate_members()

    @classmethod
    def from_lines(cls, lines, define_func, group_dict):
        """Return a Group object parsing an iterable of lines.

        :param lines: iterable
        :type lines: list[str]
        :param define_func: Function to define a unit in the registry.
        :type define_func: str -> None
        :param group_dict: Maps group name to Group.
        :type group_dict: dict[str, Group]
        """
        header, lines = lines[0], lines[1:]

        r = cls._header_re.search(header)
        name = r.groupdict()['name'].strip()
        groups = r.groupdict()['used_groups']
        if groups:
            group_names = tuple(a.strip() for a in groups.split(','))
        else:
            group_names = ()

        unit_names = []
        for line in lines:
            if '=' in line:
                # Is a definition
                definition = Definition.from_string(line)
                if not isinstance(definition, UnitDefinition):
                    raise DefinitionSyntaxError('Only UnitDefinition are valid inside _used_groups, '
                                                'not %s' % type(definition))
                define_func(definition)
                unit_names.append(definition.name)
            else:
                unit_names.append(line.strip())

        grp = cls(name, group_dict)

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
    _header_re = re.compile('@system\s+(?P<name>\w+)\s*(using\s(?P<used_groups>.*))*')

    def __init__(self, name, groups_systems):
        """
        :param name: Name of the group
        :type name: str
        :param groups_systems: dictionary containing groups and system.
                               The newly created group will be added after creation.
        :type groups_systems: dict[str, Group | System]
        """

        if name in groups_systems:
            t = 'group' if isinstance(groups_systems[name], Group) else 'system'
            raise ValueError('The system name (%s) already in use by a %s' % (name, t))

        #: Name of the system
        #: :type: str
        self.name = name

        #: Maps root unit names to a dict indicating the new unit and its exponent.
        #: :type: dict[str, dict[str, number]]]
        self.base_units = {}

        #: Derived unit names.
        #: :type: set(str)
        self.derived_units = set()

        #: Names of the _used_groups in used by this system.
        #: :type: set(str)
        self._used_groups = set()

        #: :type: frozenset | None
        self._computed_members = None

        #: Maps group name to Group.
        self._group_systems_dict = groups_systems

        self._group_systems_dict[self.name] = self

    @property
    def members(self):
        if self._computed_members is None:
            self._computed_members = set()

            for group_name in self._used_groups:
                self._computed_members |= self._group_systems_dict[group_name].members

            self._computed_members = frozenset(self._computed_members)

        return self._computed_members

    def invalidate_members(self):
        """Invalidate computed members in this Group and all parent nodes.
        """
        self._computed_members = None

    def add_groups(self, *group_names):
        """Add groups to group.

        :type group_names: str
        """
        self._used_groups |= set(group_names)

        self.invalidate_members()

    def remove_groups(self, *group_names):
        """Remove groups from group.

        :type group_names: str
        """
        self._used_groups -= set(group_names)

        self.invalidate_members()

    @classmethod
    def from_lines(cls, lines, get_root_func, group_dict):
        header, lines = lines[0], lines[1:]

        r = cls._header_re.search(header)
        name = r.groupdict()['name'].strip()
        groups = r.groupdict()['used_groups']

        # If the systems has no group, it automatically uses the root group.
        if groups:
            group_names = tuple(a.strip() for a in groups.split(','))
        else:
            group_names = ('root', )

        base_unit_names = {}
        derived_unit_names = []
        for line in lines:
            line = line.strip()

            # We would identify a
            #  - old_unit: a root unit part which is going to be removed from the system.
            #  - new_unit: a non root unit which is going to replace the old_unit.

            if ':' in line:
                # The syntax is new_unit:old_unit

                new_unit, old_unit = line.split(':')
                new_unit, old_unit = new_unit.strip(), old_unit.strip()

                # The old unit MUST be a root unit, if not raise an error.
                if old_unit != str(get_root_func(old_unit)[1]):
                    raise ValueError('In `%s`, the unit at the right of the `:` must be a root unit.' % line)

                # Here we find new_unit expanded in terms of root_units
                new_unit_expanded = to_units_container(get_root_func(new_unit)[1])

                # We require that the old unit is present in the new_unit expanded
                if old_unit not in new_unit_expanded:
                    raise ValueError('Old unit must be a component of new unit')

                # Here we invert the equation, in other words
                # we write old units in terms new unit and expansion
                new_unit_dict = dict((new_unit, -1./value)
                                     for new_unit, value in new_unit_expanded.items()
                                     if new_unit != old_unit)
                new_unit_dict[new_unit] = 1 / new_unit_expanded[old_unit]

                base_unit_names[old_unit] = new_unit_dict

            else:
                # The syntax is new_unit
                # old_unit is inferred as the root unit with the same dimensionality.

                new_unit = line
                old_unit_dict = to_units_container(get_root_func(line)[1])

                if len(old_unit_dict) != 1:
                    raise ValueError('The new base must be a root dimension if not discarded unit is specified.')

                old_unit, value = dict(old_unit_dict).popitem()

                base_unit_names[old_unit] = {new_unit: 1./value}

        system = cls(name, group_dict)

        system.add_groups(*group_names)

        system.base_units.update(**base_unit_names)
        system.derived_units |= set(derived_unit_names)

        return system


#: These dictionaries will be part of the registry
#: :type: dict[str, Group | System]
_groups_systems = dict()
_root_group = Group('root', _groups_systems)


# These methods will be included in the registry, upgrading the existing ones.

def get_group(registry, name, create_if_needed=True):
    """Return a Group.

    :param registry:
    :param name: Name of the group to be
    :param create_if_needed: Create a group if not Found. If False, raise an Exception.
    :return: Group
    """
    if name == 'root':
        raise ValueError('The name root is reserved.')

    try:
        return _groups_systems[name]
    except KeyError:
        if create_if_needed:
            return Group(name, _groups_systems)
        else:
            raise KeyError('No group %s found.' % name)


def get_system(registry, name, create_if_needed=True):
    """Return a Group.

    :param registry:
    :param name: Name of the group to be
    :param create_if_needed: Create a group if not Found. If False, raise an Exception.
    :return: System
    """

    try:
        return _groups_systems[name]
    except KeyError:
        if create_if_needed:
            return System(name, _groups_systems)
        else:
            raise KeyError('No system %s found.' % name)


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

    members = _groups_systems[group_or_system].members

    # This will not be necessary after integration with the registry as it has a strings intermediate
    members = frozenset((getattr(registry, member) for member in members))

    return ret.intersection(members)


# Current get_base_units will be renamed to get_root_units
#
# Not sure yet how to deal with the cache.
# Should we cache get_root_units, get_base_units or both?
# - get_base_units will need to be invalidated when the system is changed (How often will this happen?)
# - get_root_units will not need to be invalidated.

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

    # This will not be necessary after integration with the registry as it has a UnitsContainer intermediate
    units = to_units_container(units, registry)

    destination_units = UnitsContainer()

    bu = _groups_systems[system].base_units

    for unit, value in units.items():
        if unit in bu:
            new_unit = bu[unit]
            new_unit = to_units_container(new_unit, registry)
            destination_units *= new_unit ** value
        else:
            destination_units *= UnitsContainer({unit: value})

    base_factor = registry.convert(factor, units, destination_units)

    return base_factor, destination_units
