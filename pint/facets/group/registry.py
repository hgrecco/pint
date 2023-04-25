"""
    pint.facets.group.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, FrozenSet

from ... import errors

if TYPE_CHECKING:
    from ..._typing import Unit

from ...util import build_dependent_class, create_class_with_registry
from ..plain import PlainRegistry, UnitDefinition
from .definitions import GroupDefinition
from .objects import Group


class GroupRegistry(PlainRegistry):
    """Handle of Groups.

    Group units

    Capabilities:
    - Register groups.
    - Parse @group directive.
    """

    # TODO: Change this to Group: Group to specify class
    # and use introspection to get system class as a way
    # to enjoy typing goodies
    _group_class = Group

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: Map group name to group.
        #: :type: dict[ str | Group]
        self._groups: Dict[str, Group] = {}
        self._groups["root"] = self.Group("root")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.Group = build_dependent_class(cls, "Group", "_group_class")

    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""
        super()._init_dynamic_classes()
        self.Group = create_class_with_registry(self, self.Group)

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

    def _register_definition_adders(self) -> None:
        super()._register_definition_adders()
        self._register_adder(GroupDefinition, self._add_group)

    def _add_unit(self, definition: UnitDefinition):
        super()._add_unit(definition)
        # TODO: delta units are missing
        self.get_group("root").add_units(definition.name)

    def _add_group(self, gd: GroupDefinition):
        if gd.name in self._groups:
            raise ValueError(f"Group {gd.name} already present in registry")
        try:
            # As a Group is a SharedRegistryObject
            # it adds itself to the registry.
            self.Group.from_definition(gd)
        except KeyError as e:
            raise errors.DefinitionSyntaxError(f"unknown dimension {e} in context")

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

    def _get_compatible_units(self, input_units, group) -> FrozenSet["Unit"]:
        ret = super()._get_compatible_units(input_units, group)

        if not group:
            return ret

        if group in self._groups:
            members = self._groups[group].members
        else:
            raise ValueError("Unknown Group with name '%s'" % group)
        return frozenset(ret & members)
