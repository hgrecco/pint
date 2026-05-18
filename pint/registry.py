"""
pint.registry
~~~~~~~~~~~~~

Defines the UnitRegistry, a class to contain units and their relations.

This registry contains all pint capabilities, but you can build your
customized registry by picking only the features that you actually
need.

:copyright: 2022 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""
# pyright: reportNoOverloadImplementation=none
# pyright: reportInvalidTypeArguments=warning

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    overload,
    override,
)

from . import facets, registry_helpers
from .util import logger, pi_theorem

if TYPE_CHECKING:
    import datetime

    import numpy as np
    import optype as opt

    from ._typing import Magnitude, UnitLike
    from ._typing import Quantity as _Quantity
    from ._typing import Unit as _Unit

# To build the Quantity and Unit classes
# we follow the UnitRegistry bases
# but


MagnitudeT_co = TypeVar("MagnitudeT_co", covariant=True, bound="Magnitude")


class Quantity(
    facets.SystemRegistry.Quantity[MagnitudeT_co],
    facets.ContextRegistry.Quantity[MagnitudeT_co],
    facets.DaskRegistry.Quantity[MagnitudeT_co],
    facets.NumpyRegistry.Quantity[MagnitudeT_co],
    facets.MeasurementRegistry.Quantity[MagnitudeT_co],
    facets.NonMultiplicativeRegistry.Quantity[MagnitudeT_co],
    facets.PlainRegistry.Quantity[MagnitudeT_co],
    Generic[MagnitudeT_co],
):
    if TYPE_CHECKING:

        @override
        def __iter__[T: Magnitude](
            self: Quantity[opt.CanIter[T]],
        ) -> Iterator[Quantity[T]]: ...

        @overload
        def __round__[T: Magnitude](
            self: Quantity[opt.CanRound1[T]], ndigits: None = None
        ) -> Quantity[T]: ...
        @overload
        def __round__[T: Magnitude](
            self: Quantity[opt.CanRound2[int, T]], ndigits: int
        ) -> Quantity[T]: ...

        @override
        def __abs__[T: Magnitude](self: Quantity[opt.CanAbs[T]]) -> Quantity[T]: ...

        @classmethod
        @override
        def from_list[T: np.floating | np.integer](
            cls: type[Quantity[opt.numpy.Array1D[np.float64]]],
            quant_list: list[
                Quantity[np.floating]
                | Quantity[np.integer]
                | Quantity[float]
                | Quantity[int]
            ],
            units: UnitLike | None = None,
        ) -> Quantity[opt.numpy.Array1D[np.float64]]: ...

        @classmethod
        @override
        def from_sequence[T: np.number](
            cls: type[Quantity[opt.numpy.Array1D[np.float64]]],
            seq: Sequence[
                Quantity[np.floating]
                | Quantity[np.integer]
                | Quantity[float]
                | Quantity[int]
            ],
            units: UnitLike | None = None,
        ) -> Quantity[opt.numpy.Array1D[np.float64]]: ...

        @overload
        def tolist[T: opt.numpy.Array0D | np.number](
            self: Quantity[T],
        ) -> Quantity[T]: ...
        @overload
        def tolist[X: np.number](
            self: Quantity[opt.numpy.Array1D[X]],
        ) -> list[Quantity[X]]: ...
        @overload
        def tolist[X: np.number](
            self: Quantity[opt.numpy.Array2D[X]],
        ) -> list[list[Quantity[X]]]: ...
        @overload
        def tolist[X: np.number](
            self: Quantity[opt.numpy.Array3D[X]],
        ) -> list[list[list[Quantity[X]]]]: ...
        @overload
        def tolist[X: np.number](
            self: Quantity[
                opt.numpy.Array[tuple[int, int, int, int, *tuple[int, ...]], X]
            ],
        ) -> list[list[list[list[Any]]]]: ...

        @overload
        def __iadd__(
            self: Quantity[int | float], other: datetime.datetime
        ) -> datetime.timedelta: ...
        @overload
        def __iadd__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanIAdd[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __isub__(
            self: Quantity[int | float], other: datetime.datetime
        ) -> datetime.timedelta: ...
        @overload
        def __isub__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanISub[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __add__(
            self: Quantity[int | float], other: datetime.datetime
        ) -> datetime.timedelta: ...
        @overload
        def __add__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanAdd[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __add__[U: Magnitude](
            self,
            other: Quantity[opt.CanRAdd[MagnitudeT_co, U]]
            | opt.CanRAdd[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __sub__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanSub[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __sub__[U: Magnitude](
            self,
            other: Quantity[opt.CanRSub[MagnitudeT_co, U]]
            | opt.CanRSub[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __rsub__(
            self: Quantity[int | float],
            other: datetime.datetime,
        ) -> datetime.datetime: ...
        @overload
        def __rsub__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanRSub[T, U]], other: T
        ) -> Quantity[U]: ...
        @overload
        def __rsub__[U: Magnitude](
            self,
            other: opt.CanSub[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @override
        def __imul__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanIMul[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __mul__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanMul[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __mul__[U: Magnitude](
            self,
            other: Quantity[opt.CanRMul[MagnitudeT_co, U]]
            | opt.CanRMul[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        __rmul__ = __mul__

        @overload
        def __matmul__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanMatmul[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __matmul__[U: Magnitude](
            self,
            other: Quantity[opt.CanRMatmul[MagnitudeT_co, U]]
            | opt.CanRMatmul[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        __rmatmul__ = __matmul__

        @override
        def __itruediv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanITruediv[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __truediv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanTruediv[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __truediv__[U: Magnitude](
            self,
            other: Quantity[opt.CanRTruediv[MagnitudeT_co, U]]
            | opt.CanRTruediv[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __rtruediv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanRTruediv[T, U]], other: T
        ) -> Quantity[U]: ...
        @overload
        def __rtruediv__[U: Magnitude](
            self,
            other: opt.CanTruediv[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @override
        def __ifloordiv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanIFloordiv[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __floordiv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanFloordiv[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __floordiv__[U: Magnitude](
            self,
            other: Quantity[opt.CanRFloordiv[MagnitudeT_co, U]]
            | opt.CanRTruediv[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __rfloordiv__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanRFloordiv[T, U]], other: T
        ) -> Quantity[U]: ...
        @overload
        def __rfloordiv__[U: Magnitude](
            self,
            other: opt.CanFloordiv[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @override
        def __imod__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanIMod[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __mod__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanMod[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __mod__[U: Magnitude](
            self,
            other: Quantity[opt.CanRMod[MagnitudeT_co, U]]
            | opt.CanRMod[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __rmod__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanRMod[T, U]], other: T
        ) -> Quantity[U]: ...
        @overload
        def __rmod__[U: Magnitude](
            self,
            other: opt.CanMod[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @override
        def __ipow__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanIPow[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...

        @overload
        def __pow__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanPow[T, U]], other: Quantity[T] | T
        ) -> Quantity[U]: ...
        @overload
        def __pow__[U: Magnitude](
            self,
            other: Quantity[opt.CanRPow[MagnitudeT_co, U]]
            | opt.CanRPow[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __rpow__[T: Magnitude, U: Magnitude](
            self: Quantity[opt.CanRPow[T, U]], other: T
        ) -> Quantity[U]: ...
        @overload
        def __rpow__[U: Magnitude](
            self,
            other: opt.CanPow[MagnitudeT_co, U],
        ) -> Quantity[U]: ...

        @overload
        def __divmod__[T: Magnitude, U1: Magnitude, U2: Magnitude](
            self: Quantity[opt.CanDivmod[T, tuple[U1, U2]]],
            other: Quantity[T] | T,
        ) -> tuple[Quantity[U1], Quantity[U2]]: ...
        @overload
        def __divmod__[U1: Magnitude, U2: Magnitude](
            self,
            other: Quantity[opt.CanRDivmod[MagnitudeT_co, tuple[U1, U2]]]
            | opt.CanRDivmod[MagnitudeT_co, tuple[U1, U2]],
        ) -> tuple[Quantity[U1], Quantity[U2]]: ...

        @overload
        def __rdivmod__[T: Magnitude, U1: Magnitude, U2: Magnitude](
            self: Quantity[opt.CanRDivmod[T, tuple[U1, U2]]], other: T
        ) -> tuple[Quantity[U1], Quantity[U2]]: ...
        @overload
        def __rdivmod__[U1: Magnitude, U2: Magnitude](
            self, other: opt.CanDivmod[MagnitudeT_co, tuple[U1, U2]]
        ) -> tuple[Quantity[U1], Quantity[U2]]: ...


class Unit(
    facets.SystemRegistry.Unit,
    facets.ContextRegistry.Unit,
    facets.DaskRegistry.Unit,
    facets.NumpyRegistry.Unit,
    facets.MeasurementRegistry.Unit,
    facets.NonMultiplicativeRegistry.Unit,
    facets.PlainRegistry.Unit,
):
    if TYPE_CHECKING:

        @overload
        def __mul__(self, other: Self) -> Self: ...
        @overload
        def __mul__[T: Magnitude](self, other: T) -> Quantity[T]: ...
        @overload
        def __mul__(self, other: str) -> Quantity[Any]: ...

        __rmul__ = __mul__


class GenericUnitRegistry[QuantityT: _Quantity, UnitT: _Unit](
    facets.GenericSystemRegistry[QuantityT, UnitT],
    facets.GenericContextRegistry[QuantityT, UnitT],
    facets.GenericDaskRegistry[QuantityT, UnitT],
    facets.GenericNumpyRegistry[QuantityT, UnitT],
    facets.GenericMeasurementRegistry[QuantityT, UnitT],
    facets.GenericNonMultiplicativeRegistry[QuantityT, UnitT],
    facets.GenericPlainRegistry[QuantityT, UnitT],
):
    pass


class UnitRegistry[MagnitudeT: Magnitude](
    GenericUnitRegistry[Quantity[MagnitudeT], Unit]
):
    """The unit registry stores the definitions and relationships between units.

    Parameters
    ----------
    filename :
        path of the units definition file to load or line-iterable object.
        Empty string to load the default definition file. (default)
        None to leave the UnitRegistry empty.
    force_ndarray : bool
        convert any input, scalar or not to a numpy.ndarray.
        (Default: False)
    force_ndarray_like : bool
        convert all inputs other than duck arrays to a numpy.ndarray.
        (Default: False)
    default_as_delta :
        In the context of a multiplication of units, interpret
        non-multiplicative units as their *delta* counterparts.
        (Default: False)
    autoconvert_offset_to_baseunit :
        If True converts offset units in quantities are
        converted to their plain units in multiplicative
        context. If False no conversion happens. (Default: False)
    on_redefinition : str
        action to take in case a unit is redefined.
        'warn', 'raise', 'ignore' (Default: 'raise')
    auto_reduce_dimensions :
        If True, reduce dimensionality on appropriate operations.
        (Default: False)
    autoconvert_to_preferred :
        If True, converts preferred units on appropriate operations.
        (Default: False)
    preprocessors :
        list of callables which are iteratively ran on any input expression
        or unit string or None for no preprocessor.
        (Default=None)
    fmt_locale :
        locale identifier string, used in `format_babel` or None.
        (Default=None)
    case_sensitive : bool, optional
        Control default case sensitivity of unit parsing. (Default: True)
    cache_folder : str or pathlib.Path or None, optional
        Specify the folder in which cache files are saved and loaded from.
        If None, the cache is disabled. (default)
    """

    Quantity: TypeAlias = Quantity
    Unit: TypeAlias = Unit

    def __init__(
        self,
        filename="",
        force_ndarray: bool = False,
        force_ndarray_like: bool = False,
        default_as_delta: bool = True,
        autoconvert_offset_to_baseunit: bool = False,
        on_redefinition: str = "warn",
        system=None,
        auto_reduce_dimensions=False,
        autoconvert_to_preferred=False,
        preprocessors=None,
        fmt_locale=None,
        non_int_type=float,
        case_sensitive: bool = True,
        cache_folder=None,
    ):
        super().__init__(
            filename=filename,
            force_ndarray=force_ndarray,
            force_ndarray_like=force_ndarray_like,
            on_redefinition=on_redefinition,
            default_as_delta=default_as_delta,
            autoconvert_offset_to_baseunit=autoconvert_offset_to_baseunit,
            system=system,
            auto_reduce_dimensions=auto_reduce_dimensions,
            autoconvert_to_preferred=autoconvert_to_preferred,
            preprocessors=preprocessors,
            fmt_locale=fmt_locale,
            non_int_type=non_int_type,
            case_sensitive=case_sensitive,
            cache_folder=cache_folder,
        )

    def pi_theorem(self, quantities):
        """Builds dimensionless quantities using the Buckingham π theorem

        Parameters
        ----------
        quantities : dict
            mapping between variable name and units

        Returns
        -------
        list
            a list of dimensionless quantities expressed as dicts

        """
        return pi_theorem(quantities, self)

    def setup_matplotlib(self, enable: bool = True) -> None:
        """Set up handlers for matplotlib's unit support.

        Parameters
        ----------
        enable : bool
            whether support should be enabled or disabled (Default value = True)

        """
        # Delays importing matplotlib until it's actually requested
        from .matplotlib import setup_matplotlib_handlers

        setup_matplotlib_handlers(self, enable)

    wraps = registry_helpers.wraps

    check = registry_helpers.check


class LazyRegistry[QuantityT: Quantity, UnitT: Unit]:
    def __init__(self, args=None, kwargs=None):
        self.__dict__["params"] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)
        self._after_init()

    def __getattr__(self, item):
        if item == "_on_redefinition":
            return "raise"
        self.__init()
        return getattr(self, item)

    def __setattr__(self, key, value):
        if key == "__class__":
            super().__setattr__(key, value)
        else:
            self.__init()
            setattr(self, key, value)

    def __getitem__(self, item):
        self.__init()
        return self[item]

    def __call__(self, *args, **kwargs):
        self.__init()
        return self(*args, **kwargs)


class ApplicationRegistry:
    """A wrapper class used to distribute changes to the application registry."""

    __slots__ = ["_registry"]

    def __init__(self, registry):
        self._registry = registry

    def get(self):
        """Get the wrapped registry"""
        return self._registry

    def set(self, new_registry):
        """Set the new registry

        Parameters
        ----------
        new_registry : ApplicationRegistry or LazyRegistry or UnitRegistry
            The new registry.

        See Also
        --------
        set_application_registry
        """
        if isinstance(new_registry, type(self)):
            new_registry = new_registry.get()

        if not isinstance(new_registry, (LazyRegistry, UnitRegistry)):
            raise TypeError("Expected UnitRegistry; got %s" % type(new_registry))
        logger.debug(
            "Changing app registry from %r to %r.", self._registry, new_registry
        )
        self._registry = new_registry

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            setattr(self._registry, name, value)

    def __dir__(self):
        return dir(self._registry)

    def __getitem__(self, item):
        return self._registry[item]

    def __call__(self, *args, **kwargs):
        return self._registry(*args, **kwargs)

    def __contains__(self, item):
        return self._registry.__contains__(item)

    def __iter__(self):
        return iter(self._registry)
