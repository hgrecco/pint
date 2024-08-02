"""
    pint.facets.kind.objects
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2024 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import copy
import locale
import operator
from typing import TYPE_CHECKING

from ...compat import NUMERIC_TYPES, _to_magnitude
from ...util import PrettyIPython, SharedRegistryObject, UnitsContainer, logger
from ..plain.definitions import UnitDefinition

if TYPE_CHECKING:
    pass

from typing import Generic

from ..plain import MagnitudeT, PlainQuantity, PlainUnit


class KindQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    def to_kind(self, kind):
        return self._REGISTRY.QuantityKind(self, kind)

    def compatible_kinds(self):
        return self._REGISTRY.get_compatible_kinds(self.dimensionality)


class KindUnit(PlainUnit):
    def compatible_kinds(self):
        return self._REGISTRY.get_compatible_kinds(self.dimensionality)


class QuantityKind(KindQuantity, SharedRegistryObject):
    """Implements a class to describe a quantity and its kind.

    Parameters
    ----------
    value : pint.Quantity or any numeric type

    Returns
    -------

    """

    def __new__(cls, value, kinds, units=None):
        # if is_upcast_type(type(value)):
        #     raise TypeError(f"PlainQuantity cannot wrap upcast type {type(value)}")

        if units is None and isinstance(value, str) and value == "":
            raise ValueError(
                "Expression to parse as PlainQuantity cannot be an empty string."
            )

        if units is None and isinstance(value, str):
            ureg = SharedRegistryObject.__new__(cls)._REGISTRY
            inst = ureg.parse_expression(value)
            return cls.__new__(cls, inst)

        if units is None and isinstance(value, cls):
            return copy.copy(value)

        inst = SharedRegistryObject().__new__(cls)

        if isinstance(kinds, KindKind):
            kinds = kinds._kinds
        elif isinstance(kinds, str):
            kinds = inst._REGISTRY.parse_kinds(kinds)._kinds
        elif isinstance(kinds, UnitsContainer):
            kinds = kinds
        else:
            raise TypeError(
                "kinds must be of type str, KindKind or "
                "UnitsContainer; not {}.".format(type(kinds))
            )

        if units is None:
            kk = inst._REGISTRY.Kind(kinds)
            if kk.preferred_unit:
                units = kk.preferred_unit
            elif isinstance(value, PlainQuantity):
                units = value.units
            else:
                raise ValueError(
                    "units must be provided if value is not a Quantity and no preferred unit is defined for the kind."
                )
        else:
            if isinstance(units, (UnitsContainer, UnitDefinition)):
                units = units
            elif isinstance(units, str):
                units = inst._REGISTRY.parse_units(units)._units
            elif isinstance(units, SharedRegistryObject):
                if isinstance(units, PlainQuantity) and units.magnitude != 1:
                    units = copy.copy(units)._units
                    logger.warning(
                        "Creating new PlainQuantity using a non unity PlainQuantity as units."
                    )
                else:
                    units = units._units
            else:
                raise TypeError(
                    "units must be of type str, PlainQuantity or "
                    "UnitsContainer; not {}.".format(type(units))
                )
        if isinstance(value, PlainQuantity):
            magnitude = value.to(units)._magnitude
        else:
            magnitude = _to_magnitude(
                value, inst.force_ndarray, inst.force_ndarray_like
            )
        inst._magnitude = magnitude
        inst._units = units
        inst._kinds = kinds

        return inst

    def __repr__(self) -> str:
        return f"<QuantityKind({self._magnitude}, {self._units}, {self._kinds})>"

    def __str__(self):
        return f"{self}"

    def __format__(self, spec):
        spec = spec or self._REGISTRY.default_format
        return self._REGISTRY.formatter.format_quantitykind(self, spec)

    @property
    def kinds(self) -> KindKind:
        """PlainQuantity's kinds. Long form for `k`"""
        return self._REGISTRY.Kind(self._kinds)

    @property
    def k(self) -> KindKind:
        """PlainQuantity's kinds. Short form for `kinds`"""
        return self._REGISTRY.Kind(self._kinds)

    @property
    def quantity(self):
        return self._REGISTRY.Quantity(self.magnitude, self.units)

    @property
    def q(self):
        return self._REGISTRY.Quantity(self.magnitude, self.units)

    def _mul_div(self, other, op):
        if self._check(other):
            if isinstance(other, QuantityKind):
                result = op(self.quantity, other.quantity)
                result_units_container = op(self._kinds, other._kinds)

                for kind, relations in self._REGISTRY._kind_relations.items():
                    if result_units_container in relations:
                        return result.to_kind(kind)
                return result.to_kind(result_units_container)
        elif isinstance(other, NUMERIC_TYPES):
            return op(self.quantity, other).to_kind(self._kinds)
        else:
            return op(self.quantity, other)

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    def __div__(self, other):
        return self._mul_div(other, operator.truediv)

    def __truediv__(self, other):
        return self._mul_div(other, operator.truediv)

    def __eq__(self, other):
        if isinstance(other, QuantityKind):
            return self.q == other.q.to(self.units) and self.kinds == other.kinds
        else:
            return False

    def __pow__(self, other) -> KindKind:
        if isinstance(other, NUMERIC_TYPES):
            result_q = self.q**other
            return self.__class__(result_q.m, self._kinds**other, result_q.u)

        else:
            mess = f"Cannot power KindKind by {type(other)}"
            raise TypeError(mess)


class KindKind(PrettyIPython, SharedRegistryObject):
    """Implements a class to describe a kind supporting math operations"""

    def __reduce__(self):
        # See notes in Quantity.__reduce__
        from pint import _unpickle_unit

        return _unpickle_unit, (KindKind, self._kinds)

    def __init__(self, kinds) -> None:
        super().__init__()
        if isinstance(kinds, KindKind):
            self._kinds = kinds._kinds
        elif isinstance(kinds, UnitsContainer) and kinds.all_kinds():
            self._kinds = kinds
        elif isinstance(kinds, str):
            print(kinds, 1)
            self._kinds = self._REGISTRY.parse_kinds(kinds)._kinds
        else:
            raise TypeError(
                "kinds must be of type str, UnitsContainer; not {}.".format(type(kinds))
            )

    def __copy__(self) -> KindKind:
        ret = self.__class__(self._kinds)
        return ret

    def __deepcopy__(self, memo) -> KindKind:
        ret = self.__class__(copy.deepcopy(self._kinds, memo))
        return ret

    def __format__(self, spec: str) -> str:
        return self._REGISTRY.formatter.format_kind(self, spec)

    def __str__(self) -> str:
        return str(self._kinds)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __repr__(self) -> str:
        return f"<Kind('{self._kinds}')>"

    @property
    def dimensionless(self) -> bool:
        """Return True if the KindKind is dimensionless; False otherwise."""
        return not bool(self.dimensionality)

    @property
    def dimensionality(self) -> UnitsContainer:
        """
        Returns
        -------
        dict
            Dimensionality of the KindKind, e.g. ``{length: 1, time: -1}``
        """
        try:
            return self._dimensionality
        except AttributeError:
            dim = self._REGISTRY._get_dimensionality(self._kinds)
            self._dimensionality = dim

        return self._dimensionality

    @property
    def preferred_unit(self) -> UnitsContainer:
        units = UnitsContainer()
        for kind, exp in self._kinds.items():
            kind_definition = self._REGISTRY._dimensions[kind]
            if (
                hasattr(kind_definition, "preferred_unit")
                and kind_definition.preferred_unit
            ):
                units *= (
                    self._REGISTRY.Unit(kind_definition.preferred_unit)._units ** exp
                )
            else:
                # kind = "[torque]"
                base_dimensions = self.__class__(
                    UnitsContainer({kind: 1})
                ).dimensionality
                units *= (
                    self._REGISTRY._get_base_units_for_dimensionality(base_dimensions)
                    ** exp
                )
        return units

    def compatible_units(self, *contexts):
        return self._REGISTRY.Unit(self.preferred_unit).compatible_units(*contexts)

    def kind_relations(self):
        # TODO: Find a way to do for compound kinds
        if len(self._kinds) == 1:
            kind_name = list(self._kinds.keys())[0]
            return self._REGISTRY._kind_relations[kind_name]

    def _mul_div(self, other, op):
        if self._check(other) and isinstance(other, (KindKind, QuantityKind)):
            result_units_container = op(self._kinds, other._kinds)

            for kind, relations in self._REGISTRY._kind_relations.items():
                if result_units_container in relations:
                    return self.__class__(kind)
            return self.__class__(result_units_container)
        raise ValueError(
            f"Cannot {op} KindKind by {type(other)}. Use KindKind or QuantityKind instead."
        )

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._mul_div(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._mul_div(other, operator.truediv)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __pow__(self, other) -> KindKind:
        if isinstance(other, NUMERIC_TYPES):
            return self.__class__(self._kinds**other)

        else:
            mess = f"Cannot power KindKind by {type(other)}"
            raise TypeError(mess)

    def __hash__(self) -> int:
        return self._kinds.__hash__()

    def __eq__(self, other) -> bool:
        if self._check(other):
            if isinstance(other, self.__class__):
                return self._kinds == other._kinds
        return False

    def __ne__(self, other) -> bool:
        return not (self == other)
