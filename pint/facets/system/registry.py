"""
    pint.facets.systems.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Dict, FrozenSet, Tuple, Union

from ... import errors

if TYPE_CHECKING:
    from ..._typing import Quantity, Unit

from ..._typing import UnitLike
from ...util import UnitsContainer as UnitsContainerT
from ...util import (
    build_dependent_class,
    create_class_with_registry,
    to_units_container,
)
from ..group import GroupRegistry
from .definitions import SystemDefinition
from .objects import Lister, System


class SystemRegistry(GroupRegistry):
    """Handle of Systems.

    Conversion between units with different dimensions according
    to previously established relations (contexts).
    (e.g. in the spectroscopy, conversion between frequency and energy is possible)

    Capabilities:

    - Register systems.
    - List systems
    - Get or get the default system.
    - Parse @group directive.
    """

    # TODO: Change this to System: System to specify class
    # and use introspection to get system class as a way
    # to enjoy typing goodies
    _system_class = System

    def __init__(self, system=None, **kwargs):
        super().__init__(**kwargs)

        #: Map system name to system.
        #: :type: dict[ str | System]
        self._systems: Dict[str, System] = {}

        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self._base_units_cache = dict()

        self._default_system = system

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.System = build_dependent_class(cls, "System", "_system_class")

    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""
        super()._init_dynamic_classes()
        self.System = create_class_with_registry(self, self.System)

    def _after_init(self) -> None:
        """Invoked at the end of ``__init__``.

        - Create default group and add all orphan units to it
        - Set default system
        """
        super()._after_init()

        #: System name to be used by default.
        self._default_system = self._default_system or self._defaults.get(
            "system", None
        )

    def _register_definition_adders(self) -> None:
        super()._register_definition_adders()
        self._register_adder(SystemDefinition, self._add_system)

    def _add_system(self, sd: SystemDefinition):
        if sd.name in self._systems:
            raise ValueError(f"System {sd.name} already present in registry")

        try:
            # As a System is a SharedRegistryObject
            # it adds itself to the registry.
            self.System.from_definition(sd)
        except KeyError as e:
            # TODO: fix this error message
            raise errors.DefinitionError(f"unknown dimension {e} in context")

    @property
    def sys(self):
        return Lister(self._systems)

    @property
    def default_system(self) -> System:
        return self._default_system

    @default_system.setter
    def default_system(self, name):
        if name:
            if name not in self._systems:
                raise ValueError("Unknown system %s" % name)

            self._base_units_cache = {}

        self._default_system = name

    def get_system(self, name: str, create_if_needed: bool = True) -> System:
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
        type
            System

        """
        if name in self._systems:
            return self._systems[name]

        if not create_if_needed:
            raise ValueError("Unknown system %s" % name)

        return self.System(name)

    def get_base_units(
        self,
        input_units: Union[UnitLike, Quantity],
        check_nonmult: bool = True,
        system: Union[str, System, None] = None,
    ) -> Tuple[Number, Unit]:
        """Convert unit or dict of units to the plain units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        Unlike PlainRegistry, in this registry root_units might be different
        from base_units

        Parameters
        ----------
        input_units : UnitsContainer or str
            units
        check_nonmult : bool
            if True, None will be returned as the
            multiplicative factor if a non-multiplicative
            units is found in the final Units. (Default value = True)
        system :
             (Default value = None)

        Returns
        -------
        type
            multiplicative factor, plain units

        """

        input_units = to_units_container(input_units)

        f, units = self._get_base_units(input_units, check_nonmult, system)

        return f, self.Unit(units)

    def _get_base_units(
        self,
        input_units: UnitsContainerT,
        check_nonmult: bool = True,
        system: Union[str, System, None] = None,
    ):
        if system is None:
            system = self._default_system

        # The cache is only done for check_nonmult=True and the current system.
        if (
            check_nonmult
            and system == self._default_system
            and input_units in self._base_units_cache
        ):
            return self._base_units_cache[input_units]

        factor, units = self.get_root_units(input_units, check_nonmult)

        if not system:
            return factor, units

        # This will not be necessary after integration with the registry
        # as it has a UnitsContainer intermediate
        units = to_units_container(units, self)

        destination_units = self.UnitsContainer()

        bu = self.get_system(system, False).base_units

        for unit, value in units.items():
            if unit in bu:
                new_unit = bu[unit]
                new_unit = to_units_container(new_unit, self)
                destination_units *= new_unit**value
            else:
                destination_units *= self.UnitsContainer({unit: value})

        base_factor = self.convert(factor, units, destination_units)

        if check_nonmult:
            self._base_units_cache[input_units] = base_factor, destination_units

        return base_factor, destination_units

    def _get_compatible_units(self, input_units, group_or_system) -> FrozenSet[Unit]:
        if group_or_system is None:
            group_or_system = self._default_system

        if group_or_system and group_or_system in self._systems:
            members = self._systems[group_or_system].members
            # group_or_system has been handled by System
            return frozenset(members & super()._get_compatible_units(input_units, None))

        try:
            return super()._get_compatible_units(input_units, group_or_system)
        except ValueError as ex:
            # It might be also a system
            if "Unknown Group" in str(ex):
                raise ValueError(
                    "Unknown Group o System with name '%s'" % group_or_system
                ) from ex
            raise ex
