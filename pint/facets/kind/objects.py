"""
    pint.facets.kind.objects
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2024 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import copy
import locale
from numbers import Number
from typing import TYPE_CHECKING

from ...compat import NUMERIC_TYPES
from ...util import (
    BaseSharedRegistryObject,
    PrettyIPython,
    UnitsContainer,
)

if TYPE_CHECKING:
    pass


import copy
from typing import Generic

from ...compat import ufloat
from ..plain import MagnitudeT, PlainQuantity, PlainUnit

MISSING = object()


class KindQuantity(Generic[MagnitudeT], PlainQuantity[MagnitudeT]):
    @property
    def kinds(self) -> Kind:
        print(self._REGISTRY.Kind(self._kinds))
        """PlainQuantity's kinds. Long form for `k`"""
        return self._REGISTRY.Kind(self._kinds)

    @property
    def k(self) -> Kind:
        """PlainQuantity's kinds. Short form for `kinds`"""
        return self._REGISTRY.Kind(self._kinds)

    def to_kind(self, kind):
        kind = self._REGISTRY.Kind(kind)
        result = self.to(kind.preferred_unit)
        result._kinds = kind._kinds
        return result

    def compatible_kinds(self):
        return self._REGISTRY.get_compatible_kinds(self.dimensionality)


class KindUnit(PlainUnit):
    pass


class QuantityWithKind(PlainQuantity):
    """Implements a class to describe a quantity with uncertainty.

    Parameters
    ----------
    value : pint.Quantity or any numeric type
        The expected value of the measurement
    error : pint.Quantity or any numeric type
        The error or uncertainty of the measurement

    Returns
    -------

    """

    def __new__(cls, value, error=MISSING, units=MISSING):
        if units is MISSING:
            try:
                value, units = value.magnitude, value.units
            except AttributeError:
                # if called with two arguments and the first looks like a ufloat
                # then assume the second argument is the units, keep value intact
                if hasattr(value, "nominal_value"):
                    units = error
                    error = MISSING  # used for check below
                else:
                    units = ""
        if error is MISSING:
            # We've already extracted the units from the Quantity above
            mag = value
        else:
            try:
                error = error.to(units).magnitude
            except AttributeError:
                pass
            if error < 0:
                raise ValueError("The magnitude of the error cannot be negative")
            else:
                mag = ufloat(value, error)

        inst = super().__new__(cls, mag, units)
        return inst

    def __repr__(self):
        return "<Measurement({}, {}, {})>".format(
            self.magnitude.nominal_value, self.magnitude.std_dev, self.units
        )

    def __str__(self):
        return f"{self}"

    def __format__(self, spec):
        spec = spec or self._REGISTRY.default_format
        return self._REGISTRY.formatter.format_measurement(self, spec)




class KindKind(PrettyIPython, BaseSharedRegistryObject):
    """Implements a class to describe a unit supporting math operations"""

    def __reduce__(self):
        # See notes in Quantity.__reduce__
        from pint import _unpickle_unit

        return _unpickle_unit, (KindKind, self._kinds)

    def __init__(self, kinds) -> None:
        super().__init__()
        if isinstance(kinds, UnitsContainer) and kinds.all_kinds():
            self._kinds = kinds
        elif isinstance(kinds, str):
            self._kinds = self._REGISTRY._parse_kinds_as_container(kinds)
        # super().__init__()
        # if isinstance(units, (UnitsContainer, UnitDefinition)):
        #     self._kinds = units
        # elif isinstance(units, str):
        #     self._kinds = self._REGISTRY.parse_units(units)._kinds
        # elif isinstance(units, KindKind):
        #     self._kinds = units._kinds
        # else:
        #     raise TypeError(
        #         "units must be of type str, Unit or " "UnitsContainer; not {}.".format(
        #             type(units)
        #         )
        #     )

    def __copy__(self) -> KindKind:
        ret = self.__class__(self._kinds)
        return ret

    def __deepcopy__(self, memo) -> KindKind:
        ret = self.__class__(copy.deepcopy(self._kinds, memo))
        return ret

    # def __format__(self, spec: str) -> str:
    #     return self._REGISTRY.formatter.format_unit(self, spec)

    def __str__(self) -> str:
        return str(self._kinds)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __repr__(self) -> str:
        return f"<Kind('{self._kinds}')>"

    def __format__(self, spec: str) -> str:
        return str(self)

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
        for kind in self._kinds:
            kind_definition = self._REGISTRY._dimensions[kind]
            if (
                hasattr(kind_definition, "preferred_unit")
                and kind_definition.preferred_unit
            ):
                units = (
                    units * self._REGISTRY.Unit(kind_definition.preferred_unit)._units
                )
            else:
                # kind = "[torque]"
                base_dimensions = KindKind(UnitsContainer({kind: 1})).dimensionality
                units = units * self._REGISTRY._get_base_units_for_dimensionality(
                    base_dimensions
                )
        return units

    # def compatible_units(self, *contexts):
    #     if contexts:
    #         with self._REGISTRY.context(*contexts):
    #             return self._REGISTRY.get_compatible_units(self)

    #     return self._REGISTRY.get_compatible_units(self)

    # def is_compatible_with(
    #     self, other: Any, *contexts: str | Context, **ctx_kwargs: Any
    # ) -> bool:
    #     """check if the other object is compatible

    #     Parameters
    #     ----------
    #     other
    #         The object to check. Treated as dimensionless if not a
    #         Quantity, KindKind or str.
    #     *contexts : str or pint.Context
    #         Contexts to use in the transformation.
    #     **ctx_kwargs :
    #         Values for the Context/s

    #     Returns
    #     -------
    #     bool
    #     """
    #     from .quantity import PlainQuantity

    #     if contexts or self._REGISTRY._active_ctx:
    #         try:
    #             (1 * self).to(other, *contexts, **ctx_kwargs)
    #             return True
    #         except DimensionalityError:
    #             return False

    #     if isinstance(other, (PlainQuantity, KindKind)):
    #         return self.dimensionality == other.dimensionality

    #     if isinstance(other, str):
    #         return (
    #             self.dimensionality == self._REGISTRY.parse_units(other).dimensionality
    #         )

    #     return self.dimensionless

    def __mul__(self, other):
        if self._check(other):
            if isinstance(other, self.__class__):
                return self.__class__(self._kinds * other._kinds)
            else:
                qself = self._REGISTRY.Quantity(1, self._kinds)
                return qself * other

        if isinstance(other, Number) and other == 1:
            return self._REGISTRY.Quantity(other, self._kinds)

        return self._REGISTRY.Quantity(1, self._kinds) * other

    __rmul__ = __mul__

    def __truediv__(self, other):
        if self._check(other):
            if isinstance(other, self.__class__):
                return self.__class__(self._kinds / other._kinds)
            else:
                qself = 1 * self
                return qself / other

        return self._REGISTRY.Quantity(1 / other, self._kinds)

    def __rtruediv__(self, other):
        # As KindKind and Quantity both handle truediv with each other rtruediv can
        # only be called for something different.
        if isinstance(other, NUMERIC_TYPES):
            return self._REGISTRY.Quantity(other, 1 / self._kinds)
        elif isinstance(other, UnitsContainer):
            return self.__class__(other / self._kinds)

        return NotImplemented

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
        # We compare to the plain class of KindKind because each KindKind class is
        # unique.
        if self._check(other):
            if isinstance(other, self.__class__):
                return self._kinds == other._kinds
            else:
                return other == self._REGISTRY.Quantity(1, self._kinds)

        elif isinstance(other, NUMERIC_TYPES):
            return other == self._REGISTRY.Quantity(1, self._kinds)

        else:
            return self._kinds == other

    def __ne__(self, other) -> bool:
        return not (self == other)

    # @property
    # def systems(self):
    #     out = set()
    #     for uname in self._kinds.keys():
    #         for sname, sys in self._REGISTRY._systems.items():
    #             if uname in sys.members:
    #                 out.add(sname)
    #     return frozenset(out)
