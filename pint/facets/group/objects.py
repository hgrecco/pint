"""
    pint.facets.group.objects
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from ...util import SharedRegistryObject, getattr_maybe_raise
from .definitions import GroupDefinition


class Group(SharedRegistryObject):
    """A group is a set of units.

    Units can be added directly or by including other groups.

    Members are computed dynamically, that is if a unit is added to a group X
    all groups that include X are affected.

    The group belongs to one Registry.

    See GroupDefinition for the definition file syntax.
    """

    def __init__(self, name):
        """
        :param name: Name of the group. If not given, a root Group will be created.
        :type name: str
        :param groups: dictionary like object groups and system.
                        The newly created group will be added after creation.
        :type groups: dict[str | Group]
        """

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

        # Add this group to the group dictionary
        self._REGISTRY._groups[self.name] = self

        if name != "root":
            # All groups are added to root group
            self._REGISTRY._groups["root"].add_groups(name)

        #: A cache of the included units.
        #: None indicates that the cache has been invalidated.
        #: :type: frozenset[str] | None
        self._computed_members = None

    @property
    def members(self):
        """Names of the units that are members of the group.

        Calculated to include to all units in all included _used_groups.

        """
        if self._computed_members is None:
            self._computed_members = set(self._unit_names)

            for _, group in self.iter_used_groups():
                self._computed_members |= group.members

            self._computed_members = frozenset(self._computed_members)

        return self._computed_members

    def invalidate_members(self):
        """Invalidate computed members in this Group and all parent nodes."""
        self._computed_members = None
        d = self._REGISTRY._groups
        for name in self._used_by:
            d[name].invalidate_members()

    def iter_used_groups(self):
        pending = set(self._used_groups)
        d = self._REGISTRY._groups
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
        """Add units to group."""
        for unit_name in unit_names:
            self._unit_names.add(unit_name)

        self.invalidate_members()

    @property
    def non_inherited_unit_names(self):
        return frozenset(self._unit_names)

    def remove_units(self, *unit_names):
        """Remove units from group."""
        for unit_name in unit_names:
            self._unit_names.remove(unit_name)

        self.invalidate_members()

    def add_groups(self, *group_names):
        """Add groups to group."""
        d = self._REGISTRY._groups
        for group_name in group_names:
            grp = d[group_name]

            if grp.is_used_group(self.name):
                raise ValueError(
                    "Cyclic relationship found between %s and %s"
                    % (self.name, group_name)
                )

            self._used_groups.add(group_name)
            grp._used_by.add(self.name)

        self.invalidate_members()

    def remove_groups(self, *group_names):
        """Remove groups from group."""
        d = self._REGISTRY._groups
        for group_name in group_names:
            grp = d[group_name]

            self._used_groups.remove(group_name)
            grp._used_by.remove(self.name)

        self.invalidate_members()

    @classmethod
    def from_lines(cls, lines, define_func, non_int_type=float) -> Group:
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
        group_definition = GroupDefinition.from_lines(lines, non_int_type)
        return cls.from_definition(group_definition, define_func)

    @classmethod
    def from_definition(
        cls, group_definition: GroupDefinition, add_unit_func=None
    ) -> Group:
        grp = cls(group_definition.name)

        add_unit_func = add_unit_func or grp._REGISTRY._add_unit

        # We first add all units defined within the group
        # to the registry.
        for definition in group_definition.definitions:
            add_unit_func(definition)

        # Then we add all units defined within the group
        # to this group (by name)
        grp.add_units(*group_definition.unit_names)

        # Finally, we add all grou0ps used by this group
        # tho this group (by name)
        if group_definition.using_group_names:
            grp.add_groups(*group_definition.using_group_names)

        return grp

    def __getattr__(self, item):
        getattr_maybe_raise(self, item)
        return self._REGISTRY
