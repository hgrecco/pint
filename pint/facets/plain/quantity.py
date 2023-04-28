"""
    pint.facets.plain.quantity
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import bisect
import copy
import datetime
import locale
import math
import numbers
import operator
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from ..._typing import S, UnitLike, _MagnitudeType
from ...compat import (
    HAS_NUMPY,
    _to_magnitude,
    eq,
    is_duck_array_type,
    is_upcast_type,
    mip_INF,
    mip_INTEGER,
    mip_model,
    mip_Model,
    mip_OptimizationStatus,
    mip_xsum,
    np,
    zero_or_nan,
)
from ...errors import DimensionalityError, OffsetUnitCalculusError, PintTypeError
from ...util import (
    PrettyIPython,
    SharedRegistryObject,
    UnitsContainer,
    infer_base_unit,
    logger,
    to_units_container,
)
from .definitions import UnitDefinition

if TYPE_CHECKING:
    from ..context import Context
    from .unit import PlainUnit as Unit
    from .unit import UnitsContainer as UnitsContainerT

    if HAS_NUMPY:
        import numpy as np  # noqa


def reduce_dimensions(f):
    def wrapped(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        try:
            if result._REGISTRY.auto_reduce_dimensions:
                return result.to_reduced_units()
            else:
                return result
        except AttributeError:
            return result

    return wrapped


def ireduce_dimensions(f):
    def wrapped(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        try:
            if result._REGISTRY.auto_reduce_dimensions:
                result.ito_reduced_units()
        except AttributeError:
            pass
        return result

    return wrapped


def check_implemented(f):
    def wrapped(self, *args, **kwargs):
        other = args[0]
        if is_upcast_type(type(other)):
            return NotImplemented
        # pandas often gets to arrays of quantities [ Q_(1,"m"), Q_(2,"m")]
        # and expects PlainQuantity * array[PlainQuantity] should return NotImplemented
        elif isinstance(other, list) and other and isinstance(other[0], type(self)):
            return NotImplemented
        return f(self, *args, **kwargs)

    return wrapped


def method_wraps(numpy_func):
    if isinstance(numpy_func, str):
        numpy_func = getattr(np, numpy_func, None)

    def wrapper(func):
        func.__wrapped__ = numpy_func

        return func

    return wrapper


# Workaround to bypass dynamically generated PlainQuantity with overload method
Magnitude = TypeVar("Magnitude")


# TODO: remove all nonmultiplicative remnants


class PlainQuantity(PrettyIPython, SharedRegistryObject, Generic[_MagnitudeType]):
    """Implements a class to describe a physical quantity:
    the product of a numerical value and a unit of measurement.

    Parameters
    ----------
    value : str, pint.PlainQuantity or any numeric type
        Value of the physical quantity to be created.
    units : UnitsContainer, str or pint.PlainQuantity
        Units of the physical quantity to be created.

    Returns
    -------

    """

    #: Default formatting string.
    default_format: str = ""
    _magnitude: _MagnitudeType

    @property
    def ndim(self) -> int:
        if isinstance(self.magnitude, numbers.Number):
            return 0
        return self.magnitude.ndim

    @property
    def force_ndarray(self) -> bool:
        return self._REGISTRY.force_ndarray

    @property
    def force_ndarray_like(self) -> bool:
        return self._REGISTRY.force_ndarray_like

    @property
    def UnitsContainer(self) -> Callable[..., UnitsContainerT]:
        return self._REGISTRY.UnitsContainer

    def __reduce__(self) -> tuple:
        """Allow pickling quantities. Since UnitRegistries are not pickled, upon
        unpickling the new object is always attached to the application registry.
        """
        from pint import _unpickle_quantity

        # Note: type(self) would be a mistake as subclasses built by
        # dinamically can't be pickled
        return _unpickle_quantity, (PlainQuantity, self.magnitude, self._units)

    @overload
    def __new__(
        cls, value: str, units: Optional[UnitLike] = None
    ) -> PlainQuantity[Magnitude]:
        ...

    @overload
    def __new__(  # type: ignore[misc]
        cls, value: Sequence, units: Optional[UnitLike] = None
    ) -> PlainQuantity[np.ndarray]:
        ...

    @overload
    def __new__(
        cls, value: PlainQuantity[Magnitude], units: Optional[UnitLike] = None
    ) -> PlainQuantity[Magnitude]:
        ...

    @overload
    def __new__(
        cls, value: Magnitude, units: Optional[UnitLike] = None
    ) -> PlainQuantity[Magnitude]:
        ...

    def __new__(cls, value, units=None):
        if is_upcast_type(type(value)):
            raise TypeError(f"PlainQuantity cannot wrap upcast type {type(value)}")

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
        if units is None:
            units = inst.UnitsContainer()
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
        if isinstance(value, cls):
            magnitude = value.to(units)._magnitude
        else:
            magnitude = _to_magnitude(
                value, inst.force_ndarray, inst.force_ndarray_like
            )
        inst._magnitude = magnitude
        inst._units = units

        return inst

    def __iter__(self: PlainQuantity[Iterable[S]]) -> Iterator[S]:
        # Make sure that, if self.magnitude is not iterable, we raise TypeError as soon
        # as one calls iter(self) without waiting for the first element to be drawn from
        # the iterator
        it_magnitude = iter(self.magnitude)

        def it_outer():
            for element in it_magnitude:
                yield self.__class__(element, self._units)

        return it_outer()

    def __copy__(self) -> PlainQuantity[_MagnitudeType]:
        ret = self.__class__(copy.copy(self._magnitude), self._units)
        return ret

    def __deepcopy__(self, memo) -> PlainQuantity[_MagnitudeType]:
        ret = self.__class__(
            copy.deepcopy(self._magnitude, memo), copy.deepcopy(self._units, memo)
        )
        return ret

    def __str__(self) -> str:
        return str(self.magnitude) + " " + str(self.units)

    def __bytes__(self) -> bytes:
        return str(self).encode(locale.getpreferredencoding())

    def __repr__(self) -> str:
        if isinstance(self._magnitude, float):
            return f"<Quantity({self._magnitude:.9}, '{self._units}')>"
        else:
            return f"<Quantity({self._magnitude}, '{self._units}')>"

    def __hash__(self) -> int:
        self_base = self.to_base_units()
        if self_base.dimensionless:
            return hash(self_base.magnitude)
        else:
            return hash((self_base.__class__, self_base.magnitude, self_base.units))

    @property
    def magnitude(self) -> _MagnitudeType:
        """PlainQuantity's magnitude. Long form for `m`"""
        return self._magnitude

    @property
    def m(self) -> _MagnitudeType:
        """PlainQuantity's magnitude. Short form for `magnitude`"""
        return self._magnitude

    def m_as(self, units) -> _MagnitudeType:
        """PlainQuantity's magnitude expressed in particular units.

        Parameters
        ----------
        units : pint.PlainQuantity, str or dict
            destination units

        Returns
        -------

        """
        return self.to(units).magnitude

    @property
    def units(self) -> "Unit":
        """PlainQuantity's units. Long form for `u`"""
        return self._REGISTRY.Unit(self._units)

    @property
    def u(self) -> "Unit":
        """PlainQuantity's units. Short form for `units`"""
        return self._REGISTRY.Unit(self._units)

    @property
    def unitless(self) -> bool:
        """ """
        return not bool(self.to_root_units()._units)

    @property
    def dimensionless(self) -> bool:
        """ """
        tmp = self.to_root_units()

        return not bool(tmp.dimensionality)

    _dimensionality: Optional[UnitsContainerT] = None

    @property
    def dimensionality(self) -> UnitsContainerT:
        """
        Returns
        -------
        dict
            Dimensionality of the PlainQuantity, e.g. ``{length: 1, time: -1}``
        """
        if self._dimensionality is None:
            self._dimensionality = self._REGISTRY._get_dimensionality(self._units)

        return self._dimensionality

    def check(self, dimension: UnitLike) -> bool:
        """Return true if the quantity's dimension matches passed dimension."""
        return self.dimensionality == self._REGISTRY.get_dimensionality(dimension)

    @classmethod
    def from_list(
        cls, quant_list: List[PlainQuantity], units=None
    ) -> PlainQuantity[np.ndarray]:
        """Transforms a list of Quantities into an numpy.array quantity.
        If no units are specified, the unit of the first element will be used.
        Same as from_sequence.

        If units is not specified and list is empty, the unit cannot be determined
        and a ValueError is raised.

        Parameters
        ----------
        quant_list : list of pint.PlainQuantity
            list of pint.PlainQuantity
        units : UnitsContainer, str or pint.PlainQuantity
            units of the physical quantity to be created (Default value = None)

        Returns
        -------
        pint.PlainQuantity
        """
        return cls.from_sequence(quant_list, units=units)

    @classmethod
    def from_sequence(
        cls, seq: Sequence[PlainQuantity], units=None
    ) -> PlainQuantity[np.ndarray]:
        """Transforms a sequence of Quantities into an numpy.array quantity.
        If no units are specified, the unit of the first element will be used.

        If units is not specified and sequence is empty, the unit cannot be determined
        and a ValueError is raised.

        Parameters
        ----------
        seq : sequence of pint.PlainQuantity
            sequence of pint.PlainQuantity
        units : UnitsContainer, str or pint.PlainQuantity
            units of the physical quantity to be created (Default value = None)

        Returns
        -------
        pint.PlainQuantity
        """

        len_seq = len(seq)
        if units is None:
            if len_seq:
                units = seq[0].u
            else:
                raise ValueError("Cannot determine units from empty sequence!")

        a = np.empty(len_seq)

        for i, seq_i in enumerate(seq):
            a[i] = seq_i.m_as(units)
            # raises DimensionalityError if incompatible units are used in the sequence

        return cls(a, units)

    @classmethod
    def from_tuple(cls, tup):
        return cls(tup[0], cls._REGISTRY.UnitsContainer(tup[1]))

    def to_tuple(self) -> Tuple[_MagnitudeType, Tuple[Tuple[str]]]:
        return self.m, tuple(self._units.items())

    def compatible_units(self, *contexts):
        if contexts:
            with self._REGISTRY.context(*contexts):
                return self._REGISTRY.get_compatible_units(self._units)

        return self._REGISTRY.get_compatible_units(self._units)

    def is_compatible_with(
        self, other: Any, *contexts: Union[str, Context], **ctx_kwargs: Any
    ) -> bool:
        """check if the other object is compatible

        Parameters
        ----------
        other
            The object to check. Treated as dimensionless if not a
            PlainQuantity, Unit or str.
        *contexts : str or pint.Context
            Contexts to use in the transformation.
        **ctx_kwargs :
            Values for the Context/s

        Returns
        -------
        bool
        """
        from .unit import PlainUnit

        if contexts or self._REGISTRY._active_ctx:
            try:
                self.to(other, *contexts, **ctx_kwargs)
                return True
            except DimensionalityError:
                return False

        if isinstance(other, (PlainQuantity, PlainUnit)):
            return self.dimensionality == other.dimensionality

        if isinstance(other, str):
            return (
                self.dimensionality == self._REGISTRY.parse_units(other).dimensionality
            )

        return self.dimensionless

    def _convert_magnitude_not_inplace(self, other, *contexts, **ctx_kwargs):
        if contexts:
            with self._REGISTRY.context(*contexts, **ctx_kwargs):
                return self._REGISTRY.convert(self._magnitude, self._units, other)

        return self._REGISTRY.convert(self._magnitude, self._units, other)

    def _convert_magnitude(self, other, *contexts, **ctx_kwargs):
        if contexts:
            with self._REGISTRY.context(*contexts, **ctx_kwargs):
                return self._REGISTRY.convert(self._magnitude, self._units, other)

        return self._REGISTRY.convert(
            self._magnitude,
            self._units,
            other,
            inplace=is_duck_array_type(type(self._magnitude)),
        )

    def ito(self, other=None, *contexts, **ctx_kwargs) -> None:
        """Inplace rescale to different units.

        Parameters
        ----------
        other : pint.PlainQuantity, str or dict
            Destination units. (Default value = None)
        *contexts : str or pint.Context
            Contexts to use in the transformation.
        **ctx_kwargs :
            Values for the Context/s
        """
        other = to_units_container(other, self._REGISTRY)

        self._magnitude = self._convert_magnitude(other, *contexts, **ctx_kwargs)
        self._units = other

        return None

    def to(self, other=None, *contexts, **ctx_kwargs) -> PlainQuantity[_MagnitudeType]:
        """Return PlainQuantity rescaled to different units.

        Parameters
        ----------
        other : pint.PlainQuantity, str or dict
            destination units. (Default value = None)
        *contexts : str or pint.Context
            Contexts to use in the transformation.
        **ctx_kwargs :
            Values for the Context/s

        Returns
        -------
        pint.PlainQuantity
        """
        other = to_units_container(other, self._REGISTRY)

        magnitude = self._convert_magnitude_not_inplace(other, *contexts, **ctx_kwargs)

        return self.__class__(magnitude, other)

    def ito_root_units(self) -> None:
        """Return PlainQuantity rescaled to root units."""

        _, other = self._REGISTRY._get_root_units(self._units)

        self._magnitude = self._convert_magnitude(other)
        self._units = other

        return None

    def to_root_units(self) -> PlainQuantity[_MagnitudeType]:
        """Return PlainQuantity rescaled to root units."""

        _, other = self._REGISTRY._get_root_units(self._units)

        magnitude = self._convert_magnitude_not_inplace(other)

        return self.__class__(magnitude, other)

    def ito_base_units(self) -> None:
        """Return PlainQuantity rescaled to plain units."""

        _, other = self._REGISTRY._get_base_units(self._units)

        self._magnitude = self._convert_magnitude(other)
        self._units = other

        return None

    def to_base_units(self) -> PlainQuantity[_MagnitudeType]:
        """Return PlainQuantity rescaled to plain units."""

        _, other = self._REGISTRY._get_base_units(self._units)

        magnitude = self._convert_magnitude_not_inplace(other)

        return self.__class__(magnitude, other)

    def _get_reduced_units(self, units):
        # loop through individual units and compare to each other unit
        # can we do better than a nested loop here?
        for unit1, exp in units.items():
            # make sure it wasn't already reduced to zero exponent on prior pass
            if unit1 not in units:
                continue
            for unit2 in units:
                # get exponent after reduction
                exp = units[unit1]
                if unit1 != unit2:
                    power = self._REGISTRY._get_dimensionality_ratio(unit1, unit2)
                    if power:
                        units = units.add(unit2, exp / power).remove([unit1])
                        break
        return units

    def ito_reduced_units(self) -> None:
        """Return PlainQuantity scaled in place to reduced units, i.e. one unit per
        dimension. This will not reduce compound units (e.g., 'J/kg' will not
        be reduced to m**2/s**2), nor can it make use of contexts at this time.
        """

        # shortcuts in case we're dimensionless or only a single unit
        if self.dimensionless:
            return self.ito({})
        if len(self._units) == 1:
            return None

        units = self._units.copy()
        new_units = self._get_reduced_units(units)

        return self.ito(new_units)

    def to_reduced_units(self) -> PlainQuantity[_MagnitudeType]:
        """Return PlainQuantity scaled in place to reduced units, i.e. one unit per
        dimension. This will not reduce compound units (intentionally), nor
        can it make use of contexts at this time.
        """

        # shortcuts in case we're dimensionless or only a single unit
        if self.dimensionless:
            return self.to({})
        if len(self._units) == 1:
            return self

        units = self._units.copy()
        new_units = self._get_reduced_units(units)

        return self.to(new_units)

    def to_compact(self, unit=None) -> PlainQuantity[_MagnitudeType]:
        """ "Return PlainQuantity rescaled to compact, human-readable units.

        To get output in terms of a different unit, use the unit parameter.


        Examples
        --------

        >>> import pint
        >>> ureg = pint.UnitRegistry()
        >>> (200e-9*ureg.s).to_compact()
        <Quantity(200.0, 'nanosecond')>
        >>> (1e-2*ureg('kg m/s^2')).to_compact('N')
        <Quantity(10.0, 'millinewton')>
        """

        if not isinstance(self.magnitude, numbers.Number):
            msg = (
                "to_compact applied to non numerical types "
                "has an undefined behavior."
            )
            w = RuntimeWarning(msg)
            warnings.warn(w, stacklevel=2)
            return self

        if (
            self.unitless
            or self.magnitude == 0
            or math.isnan(self.magnitude)
            or math.isinf(self.magnitude)
        ):
            return self

        SI_prefixes: Dict[int, str] = {}
        for prefix in self._REGISTRY._prefixes.values():
            try:
                scale = prefix.converter.scale
                # Kludgy way to check if this is an SI prefix
                log10_scale = int(math.log10(scale))
                if log10_scale == math.log10(scale):
                    SI_prefixes[log10_scale] = prefix.name
            except Exception:
                SI_prefixes[0] = ""

        SI_prefixes_list = sorted(SI_prefixes.items())
        SI_powers = [item[0] for item in SI_prefixes_list]
        SI_bases = [item[1] for item in SI_prefixes_list]

        if unit is None:
            unit = infer_base_unit(self, registry=self._REGISTRY)
        else:
            unit = infer_base_unit(self.__class__(1, unit), registry=self._REGISTRY)

        q_base = self.to(unit)

        magnitude = q_base.magnitude

        units = list(q_base._units.items())
        units_numerator = [a for a in units if a[1] > 0]

        if len(units_numerator) > 0:
            unit_str, unit_power = units_numerator[0]
        else:
            unit_str, unit_power = units[0]

        if unit_power > 0:
            power = math.floor(math.log10(abs(magnitude)) / float(unit_power) / 3) * 3
        else:
            power = math.ceil(math.log10(abs(magnitude)) / float(unit_power) / 3) * 3

        index = bisect.bisect_left(SI_powers, power)

        if index >= len(SI_bases):
            index = -1

        prefix_str = SI_bases[index]

        new_unit_str = prefix_str + unit_str
        new_unit_container = q_base._units.rename(unit_str, new_unit_str)

        return self.to(new_unit_container)

    def to_preferred(
        self, preferred_units: List[UnitLike]
    ) -> PlainQuantity[_MagnitudeType]:
        """Return Quantity converted to a unit composed of the preferred units.

        Examples
        --------

        >>> import pint
        >>> ureg = pint.UnitRegistry()
        >>> (1*ureg.acre).to_preferred([ureg.meters])
        <Quantity(4046.87261, 'meter ** 2')>
        >>> (1*(ureg.force_pound*ureg.m)).to_preferred([ureg.W])
        <Quantity(4.44822162, 'second * watt')>
        """

        if not self.dimensionality:
            return self

        # The optimizer isn't perfect, and will sometimes miss obvious solutions.
        # This sub-algorithm is less powerful, but always finds the very simple solutions.
        def find_simple():
            best_ratio = None
            best_unit = None
            self_dims = sorted(self.dimensionality)
            self_exps = [self.dimensionality[d] for d in self_dims]
            s_exps_head, *s_exps_tail = self_exps
            n = len(s_exps_tail)
            for preferred_unit in preferred_units:
                dims = sorted(preferred_unit.dimensionality)
                if dims == self_dims:
                    p_exps_head, *p_exps_tail = [
                        preferred_unit.dimensionality[d] for d in dims
                    ]
                    if all(
                        s_exps_tail[i] * p_exps_head == p_exps_tail[i] ** s_exps_head
                        for i in range(n)
                    ):
                        ratio = p_exps_head / s_exps_head
                        ratio = max(ratio, 1 / ratio)
                        if best_ratio is None or ratio < best_ratio:
                            best_ratio = ratio
                            best_unit = preferred_unit ** (s_exps_head / p_exps_head)
            return best_unit

        simple = find_simple()
        if simple is not None:
            return self.to(simple)

        # For each dimension (e.g. T(ime), L(ength), M(ass)), assign a default base unit from
        # the collection of base units

        unit_selections = {
            base_unit.dimensionality: base_unit
            for base_unit in map(self._REGISTRY.Unit, self._REGISTRY._base_units)
        }

        # Override the default unit of each dimension with the 1D-units used in this Quantity
        unit_selections.update(
            {
                unit.dimensionality: unit
                for unit in map(self._REGISTRY.Unit, self._units.keys())
            }
        )

        # Determine the preferred unit for each dimensionality from the preferred_units
        # (A prefered unit doesn't have to be only one dimensional, e.g. Watts)
        preferred_dims = {
            preferred_unit.dimensionality: preferred_unit
            for preferred_unit in map(self._REGISTRY.Unit, preferred_units)
        }

        # Combine the defaults and preferred, favoring the preferred
        unit_selections.update(preferred_dims)

        # This algorithm has poor asymptotic time complexity, so first reduce the considered
        # dimensions and units to only those that are useful to the problem

        # The dimensions (without powers) of this Quantity
        dimension_set = set(self.dimensionality)

        # Getting zero exponents in dimensions not in dimension_set can be facilitated
        # by units that interact with that dimension and one or more dimension_set members.
        # For example MT^1 * LT^-1 lets you get MLT^0 when T is not in dimension_set.
        # For each candidate unit that interacts with a dimension_set member, add the
        # candidate unit's other dimensions to dimension_set, and repeat until no more
        # dimensions are selected.

        discovery_done = False
        while not discovery_done:
            discovery_done = True
            for d in unit_selections:
                unit_dimensions = set(d)
                intersection = unit_dimensions.intersection(dimension_set)
                if 0 < len(intersection) < len(unit_dimensions):
                    # there are dimensions in this unit that are in dimension set
                    # and others that are not in dimension set
                    dimension_set = dimension_set.union(unit_dimensions)
                    discovery_done = False
                    break

        # filter out dimensions and their unit selections that don't interact with any
        # dimension_set members
        unit_selections = {
            dimensionality: unit
            for dimensionality, unit in unit_selections.items()
            if set(dimensionality).intersection(dimension_set)
        }

        # update preferred_units with the selected units that were originally preferred
        preferred_units = list(
            set(u for d, u in unit_selections.items() if d in preferred_dims)
        )
        preferred_units.sort(key=lambda unit: str(unit))  # for determinism

        # and unpreferred_units are the selected units that weren't originally preferred
        unpreferred_units = list(
            set(u for d, u in unit_selections.items() if d not in preferred_dims)
        )
        unpreferred_units.sort(key=lambda unit: str(unit))  # for determinism

        # for indexability
        dimensions = list(dimension_set)
        dimensions.sort()  # for determinism

        # the powers for each elemet of dimensions (the list) for this Quantity
        dimensionality = [self.dimensionality[dimension] for dimension in dimensions]

        # Now that the input data is minimized, setup the optimization problem

        # use mip to select units from preferred units

        model = mip_Model()
        model.verbose = 0

        # Make one variable for each candidate unit

        vars = [
            model.add_var(str(unit), lb=-mip_INF, ub=mip_INF, var_type=mip_INTEGER)
            for unit in (preferred_units + unpreferred_units)
        ]

        # where [u1 ... uN] are powers of N candidate units (vars)
        # and [d1(uI) ... dK(uI)] are the K dimensional exponents of candidate unit I
        # and [t1 ... tK] are the dimensional exponents of the quantity (self)
        # create the following constraints
        #
        #                ⎡ d1(u1) ⋯ dK(u1) ⎤
        # [ u1 ⋯ uN ] * ⎢    ⋮    ⋱         ⎢ = [ t1 ⋯ tK ]
        #                ⎣ d1(uN)    dK(uN) ⎦
        #
        # in English, the units we choose, and their exponents, when combined, must have the
        # target dimensionality

        matrix = [
            [preferred_unit.dimensionality[dimension] for dimension in dimensions]
            for preferred_unit in (preferred_units + unpreferred_units)
        ]

        # Do the matrix multiplication with mip_model.xsum for performance and create constraints
        for i in range(len(dimensions)):
            dot = mip_model.xsum([var * vector[i] for var, vector in zip(vars, matrix)])
            # add constraint to the model
            model += dot == dimensionality[i]

        # where [c1 ... cN] are costs, 1 when a preferred variable, and a large value when not
        # minimize sum(abs(u1) * c1 ... abs(uN) * cN)

        # linearize the optimization variable via a proxy
        objective = model.add_var("objective", lb=0, ub=mip_INF, var_type=mip_INTEGER)

        # Constrain the objective to be equal to the sums of the absolute values of the preferred
        # unit powers. Do this by making a separate constraint for each permutation of signedness.
        # Also apply the cost coefficient, which causes the output to prefer the preferred units

        # prefer units that interact with fewer dimensions
        cost = [len(p.dimensionality) for p in preferred_units]

        # set the cost for non preferred units to a higher number
        bias = (
            max(map(abs, dimensionality)) * max((1, *cost)) * 10
        )  # arbitrary, just needs to be larger
        cost.extend([bias] * len(unpreferred_units))

        for i in range(1 << len(vars)):
            sum = mip_xsum(
                [
                    (-1 if i & 1 << (len(vars) - j - 1) else 1) * cost[j] * var
                    for j, var in enumerate(vars)
                ]
            )
            model += objective >= sum

        model.objective = objective

        # run the mips minimizer and extract the result if successful
        if model.optimize() == mip_OptimizationStatus.OPTIMAL:
            optimal_units = []
            min_objective = float("inf")
            for i in range(model.num_solutions):
                if model.objective_values[i] < min_objective:
                    min_objective = model.objective_values[i]
                    optimal_units.clear()
                elif model.objective_values[i] > min_objective:
                    continue

                temp_unit = self._REGISTRY.Unit("")
                for var in vars:
                    if var.xi(i):
                        temp_unit *= self._REGISTRY.Unit(var.name) ** var.xi(i)
                optimal_units.append(temp_unit)

            sorting_keys = {tuple(sorted(unit._units)): unit for unit in optimal_units}
            min_key = sorted(sorting_keys)[0]
            result_unit = sorting_keys[min_key]

            return self.to(result_unit)
        else:
            # for whatever reason, a solution wasn't found
            # return the original quantity
            return self

    # Mathematical operations
    def __int__(self) -> int:
        if self.dimensionless:
            return int(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self._units, "dimensionless")

    def __float__(self) -> float:
        if self.dimensionless:
            return float(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self._units, "dimensionless")

    def __complex__(self) -> complex:
        if self.dimensionless:
            return complex(self._convert_magnitude_not_inplace(UnitsContainer()))
        raise DimensionalityError(self._units, "dimensionless")

    @check_implemented
    def _iadd_sub(self, other, op):
        """Perform addition or subtraction operation in-place and return the result.

        Parameters
        ----------
        other : pint.PlainQuantity or any type accepted by :func:`_to_magnitude`
            object to be added to / subtracted from self
        op : function
            operator function (e.g. operator.add, operator.isub)

        """
        if not self._check(other):
            # other not from same Registry or not a PlainQuantity
            try:
                other_magnitude = _to_magnitude(
                    other, self.force_ndarray, self.force_ndarray_like
                )
            except PintTypeError:
                raise
            except TypeError:
                return NotImplemented
            if zero_or_nan(other, True):
                # If the other value is 0 (but not PlainQuantity 0)
                # do the operation without checking units.
                # We do the calculation instead of just returning the same
                # value to enforce any shape checking and type casting due to
                # the operation.
                self._magnitude = op(self._magnitude, other_magnitude)
            elif self.dimensionless:
                self.ito(self.UnitsContainer())
                self._magnitude = op(self._magnitude, other_magnitude)
            else:
                raise DimensionalityError(self._units, "dimensionless")
            return self

        if not self.dimensionality == other.dimensionality:
            raise DimensionalityError(
                self._units, other._units, self.dimensionality, other.dimensionality
            )

        # Next we define some variables to make if-clauses more readable.
        self_non_mul_units = self._get_non_multiplicative_units()
        is_self_multiplicative = len(self_non_mul_units) == 0
        if len(self_non_mul_units) == 1:
            self_non_mul_unit = self_non_mul_units[0]
        other_non_mul_units = other._get_non_multiplicative_units()
        is_other_multiplicative = len(other_non_mul_units) == 0
        if len(other_non_mul_units) == 1:
            other_non_mul_unit = other_non_mul_units[0]

        # Presence of non-multiplicative units gives rise to several cases.
        if is_self_multiplicative and is_other_multiplicative:
            if self._units == other._units:
                self._magnitude = op(self._magnitude, other._magnitude)
            # If only self has a delta unit, other determines unit of result.
            elif self._get_delta_units() and not other._get_delta_units():
                self._magnitude = op(
                    self._convert_magnitude(other._units), other._magnitude
                )
                self._units = other._units
            else:
                self._magnitude = op(self._magnitude, other.to(self._units)._magnitude)

        elif (
            op == operator.isub
            and len(self_non_mul_units) == 1
            and self._units[self_non_mul_unit] == 1
            and not other._has_compatible_delta(self_non_mul_unit)
        ):
            if self._units == other._units:
                self._magnitude = op(self._magnitude, other._magnitude)
            else:
                self._magnitude = op(self._magnitude, other.to(self._units)._magnitude)
            self._units = self._units.rename(
                self_non_mul_unit, "delta_" + self_non_mul_unit
            )

        elif (
            op == operator.isub
            and len(other_non_mul_units) == 1
            and other._units[other_non_mul_unit] == 1
            and not self._has_compatible_delta(other_non_mul_unit)
        ):
            # we convert to self directly since it is multiplicative
            self._magnitude = op(self._magnitude, other.to(self._units)._magnitude)

        elif (
            len(self_non_mul_units) == 1
            # order of the dimension of offset unit == 1 ?
            and self._units[self_non_mul_unit] == 1
            and other._has_compatible_delta(self_non_mul_unit)
        ):
            # Replace offset unit in self by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu = self._units.rename(self_non_mul_unit, "delta_" + self_non_mul_unit)
            self._magnitude = op(self._magnitude, other.to(tu)._magnitude)
        elif (
            len(other_non_mul_units) == 1
            # order of the dimension of offset unit == 1 ?
            and other._units[other_non_mul_unit] == 1
            and self._has_compatible_delta(other_non_mul_unit)
        ):
            # Replace offset unit in other by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu = other._units.rename(other_non_mul_unit, "delta_" + other_non_mul_unit)
            self._magnitude = op(self._convert_magnitude(tu), other._magnitude)
            self._units = other._units
        else:
            raise OffsetUnitCalculusError(self._units, other._units)

        return self

    @check_implemented
    def _add_sub(self, other, op):
        """Perform addition or subtraction operation and return the result.

        Parameters
        ----------
        other : pint.PlainQuantity or any type accepted by :func:`_to_magnitude`
            object to be added to / subtracted from self
        op : function
            operator function (e.g. operator.add, operator.isub)
        """
        if not self._check(other):
            # other not from same Registry or not a PlainQuantity
            if zero_or_nan(other, True):
                # If the other value is 0 or NaN (but not a PlainQuantity)
                # do the operation without checking units.
                # We do the calculation instead of just returning the same
                # value to enforce any shape checking and type casting due to
                # the operation.
                units = self._units
                magnitude = op(
                    self._magnitude,
                    _to_magnitude(other, self.force_ndarray, self.force_ndarray_like),
                )
            elif self.dimensionless:
                units = self.UnitsContainer()
                magnitude = op(
                    self.to(units)._magnitude,
                    _to_magnitude(other, self.force_ndarray, self.force_ndarray_like),
                )
            else:
                raise DimensionalityError(self._units, "dimensionless")
            return self.__class__(magnitude, units)

        if not self.dimensionality == other.dimensionality:
            raise DimensionalityError(
                self._units, other._units, self.dimensionality, other.dimensionality
            )

        # Next we define some variables to make if-clauses more readable.
        self_non_mul_units = self._get_non_multiplicative_units()
        is_self_multiplicative = len(self_non_mul_units) == 0
        if len(self_non_mul_units) == 1:
            self_non_mul_unit = self_non_mul_units[0]
        other_non_mul_units = other._get_non_multiplicative_units()
        is_other_multiplicative = len(other_non_mul_units) == 0
        if len(other_non_mul_units) == 1:
            other_non_mul_unit = other_non_mul_units[0]

        # Presence of non-multiplicative units gives rise to several cases.
        if is_self_multiplicative and is_other_multiplicative:
            if self._units == other._units:
                magnitude = op(self._magnitude, other._magnitude)
                units = self._units
            # If only self has a delta unit, other determines unit of result.
            elif self._get_delta_units() and not other._get_delta_units():
                magnitude = op(
                    self._convert_magnitude_not_inplace(other._units), other._magnitude
                )
                units = other._units
            else:
                units = self._units
                magnitude = op(self._magnitude, other.to(self._units).magnitude)

        elif (
            op == operator.sub
            and len(self_non_mul_units) == 1
            and self._units[self_non_mul_unit] == 1
            and not other._has_compatible_delta(self_non_mul_unit)
        ):
            if self._units == other._units:
                magnitude = op(self._magnitude, other._magnitude)
            else:
                magnitude = op(self._magnitude, other.to(self._units)._magnitude)
            units = self._units.rename(self_non_mul_unit, "delta_" + self_non_mul_unit)

        elif (
            op == operator.sub
            and len(other_non_mul_units) == 1
            and other._units[other_non_mul_unit] == 1
            and not self._has_compatible_delta(other_non_mul_unit)
        ):
            # we convert to self directly since it is multiplicative
            magnitude = op(self._magnitude, other.to(self._units)._magnitude)
            units = self._units

        elif (
            len(self_non_mul_units) == 1
            # order of the dimension of offset unit == 1 ?
            and self._units[self_non_mul_unit] == 1
            and other._has_compatible_delta(self_non_mul_unit)
        ):
            # Replace offset unit in self by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu = self._units.rename(self_non_mul_unit, "delta_" + self_non_mul_unit)
            magnitude = op(self._magnitude, other.to(tu).magnitude)
            units = self._units
        elif (
            len(other_non_mul_units) == 1
            # order of the dimension of offset unit == 1 ?
            and other._units[other_non_mul_unit] == 1
            and self._has_compatible_delta(other_non_mul_unit)
        ):
            # Replace offset unit in other by the corresponding delta unit.
            # This is done to prevent a shift by offset in the to()-call.
            tu = other._units.rename(other_non_mul_unit, "delta_" + other_non_mul_unit)
            magnitude = op(self._convert_magnitude_not_inplace(tu), other._magnitude)
            units = other._units
        else:
            raise OffsetUnitCalculusError(self._units, other._units)

        return self.__class__(magnitude, units)

    @overload
    def __iadd__(self, other: datetime.datetime) -> datetime.timedelta:  # type: ignore[misc]
        ...

    @overload
    def __iadd__(self, other) -> PlainQuantity[_MagnitudeType]:
        ...

    def __iadd__(self, other):
        if isinstance(other, datetime.datetime):
            return self.to_timedelta() + other
        elif is_duck_array_type(type(self._magnitude)):
            return self._iadd_sub(other, operator.iadd)
        else:
            return self._add_sub(other, operator.add)

    def __add__(self, other):
        if isinstance(other, datetime.datetime):
            return self.to_timedelta() + other
        else:
            return self._add_sub(other, operator.add)

    __radd__ = __add__

    def __isub__(self, other):
        if is_duck_array_type(type(self._magnitude)):
            return self._iadd_sub(other, operator.isub)
        else:
            return self._add_sub(other, operator.sub)

    def __sub__(self, other):
        return self._add_sub(other, operator.sub)

    def __rsub__(self, other):
        if isinstance(other, datetime.datetime):
            return other - self.to_timedelta()
        else:
            return -self._add_sub(other, operator.sub)

    @check_implemented
    @ireduce_dimensions
    def _imul_div(self, other, magnitude_op, units_op=None):
        """Perform multiplication or division operation in-place and return the
        result.

        Parameters
        ----------
        other : pint.PlainQuantity or any type accepted by :func:`_to_magnitude`
            object to be multiplied/divided with self
        magnitude_op : function
            operator function to perform on the magnitudes
            (e.g. operator.mul)
        units_op : function or None
            operator function to perform on the units; if None,
            *magnitude_op* is used (Default value = None)

        Returns
        -------

        """
        if units_op is None:
            units_op = magnitude_op

        offset_units_self = self._get_non_multiplicative_units()
        no_offset_units_self = len(offset_units_self)

        if not self._check(other):
            if not self._ok_for_muldiv(no_offset_units_self):
                raise OffsetUnitCalculusError(self._units, getattr(other, "units", ""))
            if len(offset_units_self) == 1:
                if self._units[offset_units_self[0]] != 1 or magnitude_op not in [
                    operator.mul,
                    operator.imul,
                ]:
                    raise OffsetUnitCalculusError(
                        self._units, getattr(other, "units", "")
                    )
            try:
                other_magnitude = _to_magnitude(
                    other, self.force_ndarray, self.force_ndarray_like
                )
            except PintTypeError:
                raise
            except TypeError:
                return NotImplemented
            self._magnitude = magnitude_op(self._magnitude, other_magnitude)
            self._units = units_op(self._units, self.UnitsContainer())
            return self

        if isinstance(other, self._REGISTRY.Unit):
            other = 1 * other

        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self._units, other._units)
        elif no_offset_units_self == 1 and len(self._units) == 1:
            self.ito_root_units()

        no_offset_units_other = len(other._get_non_multiplicative_units())

        if not other._ok_for_muldiv(no_offset_units_other):
            raise OffsetUnitCalculusError(self._units, other._units)
        elif no_offset_units_other == 1 and len(other._units) == 1:
            other.ito_root_units()

        self._magnitude = magnitude_op(self._magnitude, other._magnitude)
        self._units = units_op(self._units, other._units)

        return self

    @check_implemented
    @ireduce_dimensions
    def _mul_div(self, other, magnitude_op, units_op=None):
        """Perform multiplication or division operation and return the result.

        Parameters
        ----------
        other : pint.PlainQuantity or any type accepted by :func:`_to_magnitude`
            object to be multiplied/divided with self
        magnitude_op : function
            operator function to perform on the magnitudes
            (e.g. operator.mul)
        units_op : function or None
            operator function to perform on the units; if None,
            *magnitude_op* is used (Default value = None)

        Returns
        -------

        """
        if units_op is None:
            units_op = magnitude_op

        offset_units_self = self._get_non_multiplicative_units()
        no_offset_units_self = len(offset_units_self)

        if not self._check(other):
            if not self._ok_for_muldiv(no_offset_units_self):
                raise OffsetUnitCalculusError(self._units, getattr(other, "units", ""))
            if len(offset_units_self) == 1:
                if self._units[offset_units_self[0]] != 1 or magnitude_op not in [
                    operator.mul,
                    operator.imul,
                ]:
                    raise OffsetUnitCalculusError(
                        self._units, getattr(other, "units", "")
                    )
            try:
                other_magnitude = _to_magnitude(
                    other, self.force_ndarray, self.force_ndarray_like
                )
            except PintTypeError:
                raise
            except TypeError:
                return NotImplemented

            magnitude = magnitude_op(self._magnitude, other_magnitude)
            units = units_op(self._units, self.UnitsContainer())

            return self.__class__(magnitude, units)

        if isinstance(other, self._REGISTRY.Unit):
            other = 1 * other

        new_self = self

        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self._units, other._units)
        elif no_offset_units_self == 1 and len(self._units) == 1:
            new_self = self.to_root_units()

        no_offset_units_other = len(other._get_non_multiplicative_units())

        if not other._ok_for_muldiv(no_offset_units_other):
            raise OffsetUnitCalculusError(self._units, other._units)
        elif no_offset_units_other == 1 and len(other._units) == 1:
            other = other.to_root_units()

        magnitude = magnitude_op(new_self._magnitude, other._magnitude)
        units = units_op(new_self._units, other._units)

        return self.__class__(magnitude, units)

    def __imul__(self, other):
        if is_duck_array_type(type(self._magnitude)):
            return self._imul_div(other, operator.imul)
        else:
            return self._mul_div(other, operator.mul)

    def __mul__(self, other):
        return self._mul_div(other, operator.mul)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return np.matmul(self, other)

    __rmatmul__ = __matmul__

    def _truedivide_cast_int(self, a, b):
        t = self._REGISTRY.non_int_type
        if isinstance(a, int):
            a = t(a)
        if isinstance(b, int):
            b = t(b)
        return operator.truediv(a, b)

    def __itruediv__(self, other):
        if is_duck_array_type(type(self._magnitude)):
            return self._imul_div(other, operator.itruediv)
        else:
            return self._mul_div(other, operator.truediv)

    def __truediv__(self, other):
        if isinstance(self.m, int) or isinstance(getattr(other, "m", None), int):
            return self._mul_div(other, self._truedivide_cast_int, operator.truediv)
        return self._mul_div(other, operator.truediv)

    def __rtruediv__(self, other):
        try:
            other_magnitude = _to_magnitude(
                other, self.force_ndarray, self.force_ndarray_like
            )
        except PintTypeError:
            raise
        except TypeError:
            return NotImplemented

        no_offset_units_self = len(self._get_non_multiplicative_units())
        if not self._ok_for_muldiv(no_offset_units_self):
            raise OffsetUnitCalculusError(self._units, "")
        elif no_offset_units_self == 1 and len(self._units) == 1:
            self = self.to_root_units()

        return self.__class__(other_magnitude / self._magnitude, 1 / self._units)

    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __idiv__ = __itruediv__

    def __ifloordiv__(self, other):
        if self._check(other):
            self._magnitude //= other.to(self._units)._magnitude
        elif self.dimensionless:
            self._magnitude = self.to("")._magnitude // other
        else:
            raise DimensionalityError(self._units, "dimensionless")
        self._units = self.UnitsContainer({})
        return self

    @check_implemented
    def __floordiv__(self, other):
        if self._check(other):
            magnitude = self._magnitude // other.to(self._units)._magnitude
        elif self.dimensionless:
            magnitude = self.to("")._magnitude // other
        else:
            raise DimensionalityError(self._units, "dimensionless")
        return self.__class__(magnitude, self.UnitsContainer({}))

    @check_implemented
    def __rfloordiv__(self, other):
        if self._check(other):
            magnitude = other._magnitude // self.to(other._units)._magnitude
        elif self.dimensionless:
            magnitude = other // self.to("")._magnitude
        else:
            raise DimensionalityError(self._units, "dimensionless")
        return self.__class__(magnitude, self.UnitsContainer({}))

    @check_implemented
    def __imod__(self, other):
        if not self._check(other):
            other = self.__class__(other, self.UnitsContainer({}))
        self._magnitude %= other.to(self._units)._magnitude
        return self

    @check_implemented
    def __mod__(self, other):
        if not self._check(other):
            other = self.__class__(other, self.UnitsContainer({}))
        magnitude = self._magnitude % other.to(self._units)._magnitude
        return self.__class__(magnitude, self._units)

    @check_implemented
    def __rmod__(self, other):
        if self._check(other):
            magnitude = other._magnitude % self.to(other._units)._magnitude
            return self.__class__(magnitude, other._units)
        elif self.dimensionless:
            magnitude = other % self.to("")._magnitude
            return self.__class__(magnitude, self.UnitsContainer({}))
        else:
            raise DimensionalityError(self._units, "dimensionless")

    @check_implemented
    def __divmod__(self, other):
        if not self._check(other):
            other = self.__class__(other, self.UnitsContainer({}))
        q, r = divmod(self._magnitude, other.to(self._units)._magnitude)
        return (
            self.__class__(q, self.UnitsContainer({})),
            self.__class__(r, self._units),
        )

    @check_implemented
    def __rdivmod__(self, other):
        if self._check(other):
            q, r = divmod(other._magnitude, self.to(other._units)._magnitude)
            unit = other._units
        elif self.dimensionless:
            q, r = divmod(other, self.to("")._magnitude)
            unit = self.UnitsContainer({})
        else:
            raise DimensionalityError(self._units, "dimensionless")
        return (self.__class__(q, self.UnitsContainer({})), self.__class__(r, unit))

    @check_implemented
    def __ipow__(self, other):
        if not is_duck_array_type(type(self._magnitude)):
            return self.__pow__(other)

        try:
            _to_magnitude(other, self.force_ndarray, self.force_ndarray_like)
        except PintTypeError:
            raise
        except TypeError:
            return NotImplemented
        else:
            if not self._ok_for_muldiv:
                raise OffsetUnitCalculusError(self._units)

            if is_duck_array_type(type(getattr(other, "_magnitude", other))):
                # arrays are refused as exponent, because they would create
                # len(array) quantities of len(set(array)) different units
                # unless the plain is dimensionless. Ensure dimensionless
                # units are reduced to "dimensionless".
                # Note: this will strip Units of degrees or radians from PlainQuantity
                if self.dimensionless:
                    if getattr(other, "dimensionless", False):
                        self._magnitude = self.m_as("") ** other.m_as("")
                        self._units = self.UnitsContainer()
                        return self
                    elif not getattr(other, "dimensionless", True):
                        raise DimensionalityError(other._units, "dimensionless")
                    else:
                        self._magnitude = self.m_as("") ** other
                        self._units = self.UnitsContainer()
                        return self
                elif np.size(other) > 1:
                    raise DimensionalityError(
                        self._units,
                        "dimensionless",
                        extra_msg=". PlainQuantity array exponents are only allowed if the "
                        "plain is dimensionless",
                    )

            if other == 1:
                return self
            elif other == 0:
                self._units = self.UnitsContainer()
            else:
                if not self._is_multiplicative:
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        self.ito_base_units()
                    else:
                        raise OffsetUnitCalculusError(self._units)

                if getattr(other, "dimensionless", False):
                    other = other.to_base_units().magnitude
                    self._units **= other
                elif not getattr(other, "dimensionless", True):
                    raise DimensionalityError(self._units, "dimensionless")
                else:
                    self._units **= other

            self._magnitude **= _to_magnitude(
                other, self.force_ndarray, self.force_ndarray_like
            )
            return self

    @check_implemented
    def __pow__(self, other) -> PlainQuantity[_MagnitudeType]:
        try:
            _to_magnitude(other, self.force_ndarray, self.force_ndarray_like)
        except PintTypeError:
            raise
        except TypeError:
            return NotImplemented
        else:
            if not self._ok_for_muldiv:
                raise OffsetUnitCalculusError(self._units)

            if is_duck_array_type(type(getattr(other, "_magnitude", other))):
                # arrays are refused as exponent, because they would create
                # len(array) quantities of len(set(array)) different units
                # unless the plain is dimensionless.
                # Note: this will strip Units of degrees or radians from PlainQuantity
                if self.dimensionless:
                    if getattr(other, "dimensionless", False):
                        return self.__class__(
                            self._convert_magnitude_not_inplace(self.UnitsContainer())
                            ** other.m_as("")
                        )
                    elif not getattr(other, "dimensionless", True):
                        raise DimensionalityError(other._units, "dimensionless")
                    else:
                        return self.__class__(
                            self._convert_magnitude_not_inplace(self.UnitsContainer())
                            ** other
                        )
                elif np.size(other) > 1:
                    raise DimensionalityError(
                        self._units,
                        "dimensionless",
                        extra_msg=". PlainQuantity array exponents are only allowed if the "
                        "plain is dimensionless",
                    )

            new_self = self
            if other == 1:
                return self
            elif other == 0:
                exponent = 0
                units = self.UnitsContainer()
            else:
                if not self._is_multiplicative:
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        new_self = self.to_root_units()
                    else:
                        raise OffsetUnitCalculusError(self._units)

                if getattr(other, "dimensionless", False):
                    exponent = other.to_root_units().magnitude
                    units = new_self._units**exponent
                elif not getattr(other, "dimensionless", True):
                    raise DimensionalityError(other._units, "dimensionless")
                else:
                    exponent = _to_magnitude(
                        other, force_ndarray=False, force_ndarray_like=False
                    )
                    units = new_self._units**exponent

            magnitude = new_self._magnitude**exponent
            return self.__class__(magnitude, units)

    @check_implemented
    def __rpow__(self, other) -> PlainQuantity[_MagnitudeType]:
        try:
            _to_magnitude(other, self.force_ndarray, self.force_ndarray_like)
        except PintTypeError:
            raise
        except TypeError:
            return NotImplemented
        else:
            if not self.dimensionless:
                raise DimensionalityError(self._units, "dimensionless")
            new_self = self.to_root_units()
            return other**new_self._magnitude

    def __abs__(self) -> PlainQuantity[_MagnitudeType]:
        return self.__class__(abs(self._magnitude), self._units)

    def __round__(self, ndigits: Optional[int] = 0) -> PlainQuantity[int]:
        return self.__class__(round(self._magnitude, ndigits=ndigits), self._units)

    def __pos__(self) -> PlainQuantity[_MagnitudeType]:
        return self.__class__(operator.pos(self._magnitude), self._units)

    def __neg__(self) -> PlainQuantity[_MagnitudeType]:
        return self.__class__(operator.neg(self._magnitude), self._units)

    @check_implemented
    def __eq__(self, other):
        def bool_result(value):
            nonlocal other

            if not is_duck_array_type(type(self._magnitude)):
                return value

            if isinstance(other, PlainQuantity):
                other = other._magnitude

            template, _ = np.broadcast_arrays(self._magnitude, other)
            return np.full_like(template, fill_value=value, dtype=np.bool_)

        # We compare to the plain class of PlainQuantity because
        # each PlainQuantity class is unique.
        if not isinstance(other, PlainQuantity):
            if zero_or_nan(other, True):
                # Handle the special case in which we compare to zero or NaN
                # (or an array of zeros or NaNs)
                if self._is_multiplicative:
                    # compare magnitude
                    return eq(self._magnitude, other, False)
                else:
                    # compare the magnitude after converting the
                    # non-multiplicative quantity to plain units
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        return eq(self.to_base_units()._magnitude, other, False)
                    else:
                        raise OffsetUnitCalculusError(self._units)

            if self.dimensionless:
                return eq(
                    self._convert_magnitude_not_inplace(self.UnitsContainer()),
                    other,
                    False,
                )

            return bool_result(False)

        # TODO: this might be expensive. Do we even need it?
        if eq(self._magnitude, 0, True) and eq(other._magnitude, 0, True):
            return bool_result(self.dimensionality == other.dimensionality)

        if self._units == other._units:
            return eq(self._magnitude, other._magnitude, False)

        try:
            return eq(
                self._convert_magnitude_not_inplace(other._units),
                other._magnitude,
                False,
            )
        except DimensionalityError:
            return bool_result(False)

    @check_implemented
    def __ne__(self, other):
        out = self.__eq__(other)
        if is_duck_array_type(type(out)):
            return np.logical_not(out)
        return not out

    @check_implemented
    def compare(self, other, op):
        if not isinstance(other, PlainQuantity):
            if self.dimensionless:
                return op(
                    self._convert_magnitude_not_inplace(self.UnitsContainer()), other
                )
            elif zero_or_nan(other, True):
                # Handle the special case in which we compare to zero or NaN
                # (or an array of zeros or NaNs)
                if self._is_multiplicative:
                    # compare magnitude
                    return op(self._magnitude, other)
                else:
                    # compare the magnitude after converting the
                    # non-multiplicative quantity to plain units
                    if self._REGISTRY.autoconvert_offset_to_baseunit:
                        return op(self.to_base_units()._magnitude, other)
                    else:
                        raise OffsetUnitCalculusError(self._units)
            else:
                raise ValueError(
                    "Cannot compare PlainQuantity and {}".format(type(other))
                )

        # Registry equality check based on util.SharedRegistryObject
        if self._REGISTRY is not other._REGISTRY:
            mess = "Cannot operate with {} and {} of different registries."
            raise ValueError(
                mess.format(self.__class__.__name__, other.__class__.__name__)
            )

        if self._units == other._units:
            return op(self._magnitude, other._magnitude)
        if self.dimensionality != other.dimensionality:
            raise DimensionalityError(
                self._units, other._units, self.dimensionality, other.dimensionality
            )
        return op(self.to_root_units().magnitude, other.to_root_units().magnitude)

    __lt__ = lambda self, other: self.compare(other, op=operator.lt)
    __le__ = lambda self, other: self.compare(other, op=operator.le)
    __ge__ = lambda self, other: self.compare(other, op=operator.ge)
    __gt__ = lambda self, other: self.compare(other, op=operator.gt)

    def __bool__(self) -> bool:
        # Only cast when non-ambiguous (when multiplicative unit)
        if self._is_multiplicative:
            return bool(self._magnitude)
        else:
            raise ValueError(
                "Boolean value of PlainQuantity with offset unit is ambiguous."
            )

    __nonzero__ = __bool__

    def tolist(self):
        units = self._units

        try:
            values = self._magnitude.tolist()
            if not isinstance(values, list):
                return self.__class__(values, units)

            return [
                self.__class__(value, units).tolist()
                if isinstance(value, list)
                else self.__class__(value, units)
                for value in self._magnitude.tolist()
            ]
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support tolist."
            )

    def _get_unit_definition(self, unit: str) -> UnitDefinition:
        try:
            return self._REGISTRY._units[unit]
        except KeyError:
            # pint#1062: The __init__ method of this object added the unit to
            # UnitRegistry._units (e.g. units with prefix are added on the fly the
            # first time they're used) but the key was later removed, e.g. because
            # a Context with unit redefinitions was deactivated.
            self._REGISTRY.parse_units(unit)
            return self._REGISTRY._units[unit]

    # methods/properties that help for math operations with offset units
    @property
    def _is_multiplicative(self) -> bool:
        """Check if the PlainQuantity object has only multiplicative units."""
        return True

    def _get_non_multiplicative_units(self) -> List[str]:
        """Return a list of the of non-multiplicative units of the PlainQuantity object."""
        return []

    def _get_delta_units(self) -> List[str]:
        """Return list of delta units ot the PlainQuantity object."""
        return [u for u in self._units if u.startswith("delta_")]

    def _has_compatible_delta(self, unit: str) -> bool:
        """ "Check if PlainQuantity object has a delta_unit that is compatible with unit"""
        return False

    def _ok_for_muldiv(self, no_offset_units=None) -> bool:
        return True

    def to_timedelta(self: PlainQuantity[float]) -> datetime.timedelta:
        return datetime.timedelta(microseconds=self.to("microseconds").magnitude)
