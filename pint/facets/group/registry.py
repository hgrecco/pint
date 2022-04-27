"""
    pint.facets.group.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from typing import Dict, FrozenSet

from pint.facets.plain.unit import Unit

from ..plain.definitions import UnitDefinition
from .definitions import GroupDefinition
from .objects import Group, build_group_class


class GroupRegistry:
    """Handle of Groups.

    Group units

    Capabilities:
    - Register groups.
    - Parse @group directive.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: Map group name to group.
        #: :type: dict[ str | Group]
        self._groups: Dict[str, Group] = {}
        self._groups["root"] = self.Group("root")

    def _init_dynamic_classes(self) -> None:
        super()._init_dynamic_classes()
        self.Group = build_group_class(self)

    def _after_init(self) -> None:
        """Invoked at the end of ``__init__``.

        - Create default group and add all orphan units to it
        - Set default system
        """
        super()._after_init()

        #: Copy units not defined in any group to the default group
        if "group" in self._defaults:
            grp = self.get_group(self._defaults["group"], True)
            group_units = frozenset(
                [
                    member
                    for group in self._groups.values()
                    if group.name != "root"
                    for member in group.members
                ]
            )
            all_units = self.get_group("root", False).members
            grp.add_units(*(all_units - group_units))

    def _register_directives(self) -> None:
        super()._register_directives()
        self._register_directive(
            "@group",
            lambda gd: self.Group.from_definition(gd, self.define),
            GroupDefinition,
        )

    def get_group(self, name: str, create_if_needed: bool = True) -> Group:
        """Return a Group.

        Parameters
        ----------
        name : str
            Name of the group to be
        create_if_needed : bool
            If True, create a group if not found. If False, raise an Exception.
            (Default value = True)

        Returns
        -------
        Group
            Group
        """
        if name in self._groups:
            return self._groups[name]

        if not create_if_needed:
            raise ValueError("Unknown group %s" % name)

        return self.Group(name)

    def _define(self, definition):

        # In addition to the what is done by the PlainRegistry,
        # this adds all units to the `root` group.

        definition, d, di = super()._define(definition)

        if isinstance(definition, UnitDefinition):
            # We add all units to the root group
            self.get_group("root").add_units(definition.name)

        return definition, d, di

    def _get_compatible_units(self, input_units, group) -> FrozenSet[Unit]:

        ret = super()._get_compatible_units(input_units, group)

        if not group:
            return ret

        if group in self._groups:
            members = self._groups[group].members
        else:
            raise ValueError("Unknown Group with name '%s'" % group)
        return frozenset(ret & members)
