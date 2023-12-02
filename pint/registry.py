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

from __future__ import annotations

import copy
import pathlib
from typing import (
    Generic,
    Union,
    Optional,
    Callable,
    Generator,
    Any,
    Iterable,
    Iterator,
)
from typing_extensions import Self
from .util import UnitsContainer
from . import registry_helpers
from . import facets
from ._typing import QuantityOrUnitLike, UnitLike, QuantityArgument, Scalar, F, T

from .facets.plain.registry import QuantityT, UnitT
from .facets.context import ContextDefinition
from .facets.system import System

from .util import logger, pi_theorem, getattr_maybe_raise
from .compat import TypeAlias


# To build the Quantity and Unit classes
# we follow the UnitRegistry bases
# but


class Quantity(
    facets.SystemRegistry.Quantity,
    facets.ContextRegistry.Quantity,
    facets.DaskRegistry.Quantity,
    facets.NumpyRegistry.Quantity,
    facets.MeasurementRegistry.Quantity,
    facets.FormattingRegistry.Quantity,
    facets.NonMultiplicativeRegistry.Quantity,
    facets.PlainRegistry.Quantity,
):
    pass


class Unit(
    facets.SystemRegistry.Unit,
    facets.ContextRegistry.Unit,
    facets.DaskRegistry.Unit,
    facets.NumpyRegistry.Unit,
    facets.MeasurementRegistry.Unit,
    facets.FormattingRegistry.Unit,
    facets.NonMultiplicativeRegistry.Unit,
    facets.PlainRegistry.Unit,
):
    pass


class GenericUnitRegistry(
    Generic[facets.QuantityT, facets.UnitT],
    facets.GenericSystemRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericContextRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericDaskRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericNumpyRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericMeasurementRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericFormattingRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericNonMultiplicativeRegistry[facets.QuantityT, facets.UnitT],
    facets.GenericPlainRegistry[facets.QuantityT, facets.UnitT],
):
    pass


class SubRegistry(GenericUnitRegistry[Quantity, Unit]):
    Quantity: TypeAlias = Quantity
    Unit: TypeAlias = Unit


class UnitRegistry:
    """The unit registry stores the definitions and relationships between units.

    Parameters
    ----------
    filename :
        path of the units definition file to load or line-iterable object.
        Empty to load the default definition file.
        None to leave the UnitRegistry empty.
    force_ndarray : bool
        convert any input, scalar or not to a numpy.ndarray.
    force_ndarray_like : bool
        convert all inputs other than duck arrays to a numpy.ndarray.
    default_as_delta :
        In the context of a multiplication of units, interpret
        non-multiplicative units as their *delta* counterparts.
    autoconvert_offset_to_baseunit :
        If True converts offset units in quantities are
        converted to their plain units in multiplicative
        context. If False no conversion happens.
    on_redefinition : str
        action to take in case a unit is redefined.
        'warn', 'raise', 'ignore'
    auto_reduce_dimensions :
        If True, reduce dimensionality on appropriate operations.
    autoconvert_to_preferred :
        If True, converts preferred units on appropriate operations.
    preprocessors :
        list of callables which are iteratively ran on any input expression
        or unit string
    fmt_locale :
        locale identifier string, used in `format_babel`. Default to None
    case_sensitive : bool, optional
        Control default case sensitivity of unit parsing. (Default: True)
    cache_folder : str or pathlib.Path or None, optional
        Specify the folder in which cache files are saved and loaded from.
        If None, the cache is disabled. (default)
    """

    Quantity: TypeAlias = SubRegistry.Quantity
    Unit: TypeAlias = SubRegistry.Unit
    Measurement: TypeAlias = SubRegistry.Measurement
    Context: TypeAlias = SubRegistry.Context
    Group: TypeAlias = SubRegistry.Group

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
        self._subregistry = SubRegistry(
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
        self.Quantity = self._subregistry.Quantity
        self.Unit = self._subregistry.Unit
        self.Measurement = self._subregistry.Measurement
        self.Context = self._subregistry.Context
        self.Group = self._subregistry.Group
        self.System = self._subregistry.System
        self.UnitsContainer = self._subregistry.UnitsContainer

    @property
    def default_as_delta(self):
        return self._subregistry.default_as_delta

    @default_as_delta.setter
    def default_as_delta(self, value):
        self._subregistry.default_as_delta = value

    @property
    def non_int_type(self):
        return self._subregistry.non_int_type

    @non_int_type.setter
    def non_int_type(self, value):
        self._subregistry.non_int_type = value

    @property
    def separate_format_defaults(self):
        return self._subregistry.separate_format_defaults

    @separate_format_defaults.setter
    def separate_format_defaults(self, value):
        self._subregistry.separate_format_defaults = value

    @property
    def force_ndarray(self):
        return self._subregistry.force_ndarray

    @force_ndarray.setter
    def force_ndarray(self, value):
        self._subregistry.force_ndarray = value

    @property
    def force_ndarray_like(self):
        return self._subregistry.force_ndarray_like

    @force_ndarray_like.setter
    def force_ndarray_like(self, value):
        self._subregistry.force_ndarray_like = value

    @property
    def fmt_locale(self):
        return self._subregistry.fmt_locale

    @fmt_locale.setter
    def fmt_locale(self, value):
        self._subregistry.fmt_locale = value

    def set_fmt_locale(self, loc: Optional[str]) -> None:
        """Change the locale used by default by `format_babel`.

        Parameters
        ----------
        loc : str or None
            None` (do not translate), 'sys' (detect the system locale) or a locale id string.
        """
        return self._subregistry.set_fmt_locale(loc)

    @property
    def default_format(self):
        return self._subregistry.default_format

    @default_format.setter
    def default_format(self, value):
        self._subregistry.default_format = value

    @property
    def cache_folder(self) -> Optional[pathlib.Path]:
        return self._subregistry.cache_folder

    @property
    def autoconvert_offset_to_baseunit(self):
        return self._subregistry.autoconvert_offset_to_baseunit

    @autoconvert_offset_to_baseunit.setter
    def autoconvert_offset_to_baseunit(self, value):
        self._subregistry.autoconvert_offset_to_baseunit = value

    @property
    def preprocessors(self):
        return self._subregistry.preprocessors

    @preprocessors.setter
    def preprocessors(self, value):
        self._subregistry.preprocessors = value

    @property
    def sys(self):
        return self._subregistry.sys

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

    def wraps(
        self,
        ret: Optional[Union[str, Unit, Iterable[Optional[Union[str, Unit]]]]],
        args: Optional[Union[str, Unit, Iterable[Optional[Union[str, Unit]]]]],
        strict: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Quantity]]:
        """Wraps a function to become pint-aware.

        Use it when a function requires a numerical value but in some specific
        units. The wrapper function will take a pint quantity, convert to the units
        specified in `args` and then call the wrapped function with the resulting
        magnitude.

        The value returned by the wrapped function will be converted to the units
        specified in `ret`.

        Parameters
        ----------
        ureg : pint.UnitRegistry
            a UnitRegistry instance.
        ret : str, pint.Unit, or iterable of str or pint.Unit
            Units of each of the return values. Use `None` to skip argument conversion.
        args : str, pint.Unit, or iterable of str or pint.Unit
            Units of each of the input arguments. Use `None` to skip argument conversion.
        strict : bool
            Indicates that only quantities are accepted. (Default value = True)

        Returns
        -------
        callable
            the wrapper function.

        Raises
        ------
        TypeError
            if the number of given arguments does not match the number of function parameters.
            if any of the provided arguments is not a unit a string or Quantity

        """
        return registry_helpers.wraps(self._subregistry, ret, args, strict)

    def check(
        self, *args: Optional[Union[str, UnitsContainer, Unit]]
    ) -> Callable[[F], F]:
        """Decorator to for quantity type checking for function inputs.

        Use it to ensure that the decorated function input parameters match
        the expected dimension of pint quantity.

        The wrapper function raises:
        - `pint.DimensionalityError` if an argument doesn't match the required dimensions.

        ureg : UnitRegistry
            a UnitRegistry instance.
        args : str or UnitContainer or None
            Dimensions of each of the input arguments.
            Use `None` to skip argument conversion.

        Returns
        -------
        callable
            the wrapped function.

        Raises
        ------
        TypeError
            If the number of given dimensions does not match the number of function
            parameters.
        ValueError
            If the any of the provided dimensions cannot be parsed as a dimension.
        """
        return registry_helpers.check(self._subregistry, *args)

    ##########
    # Context
    ##########

    def add_context(self, context: Union[Context, ContextDefinition]) -> None:
        """Add a context object to the registry.

        The context will be accessible by its name and aliases.

        Notice that this method will NOT enable the context;
        see :meth:`enable_contexts`.
        """
        return self._subregistry.add_context(context)

    def remove_context(self, name_or_alias: str) -> Context:
        """Remove a context from the registry and return it.

        Notice that this methods will not disable the context;
        see :meth:`disable_contexts`.
        """
        return self._subregistry.remove_context(name_or_alias)

    def enable_contexts(
        self, *names_or_contexts: Union[str, Context], **kwargs: Any
    ) -> None:
        """Enable contexts provided by name or by object.

        Parameters
        ----------
        *names_or_contexts :
            one or more contexts or context names/aliases
        **kwargs :
            keyword arguments for the context(s)

        Examples
        --------
        See :meth:`context`
        """
        return self._subregistry.enable_contexts(*names_or_contexts, **kwargs)

    def disable_contexts(self, n: Optional[int] = None) -> None:
        """Disable the last n enabled contexts.

        Parameters
        ----------
        n : int
            Number of contexts to disable. Default: disable all contexts.
        """
        return self._subregistry.disable_contexts(n)

    def context(
        self, *names: str, **kwargs: Any
    ) -> Generator[UnitRegistry, None, None]:
        """Used as a context manager, this function enables to activate a context
        which is removed after usage.

        Parameters
        ----------
        *names : name(s) of the context(s).
        **kwargs : keyword arguments for the contexts.

        Examples
        --------
        Context can be called by their name:

        >>> import pint.facets.context.objects
        >>> import pint
        >>> ureg = pint.UnitRegistry()
        >>> ureg.add_context(pint.facets.context.objects.Context('one'))
        >>> ureg.add_context(pint.facets.context.objects.Context('two'))
        >>> with ureg.context('one'):
        ...     pass

        If a context has an argument, you can specify its value as a keyword argument:

        >>> with ureg.context('one', n=1):
        ...     pass

        Multiple contexts can be entered in single call:

        >>> with ureg.context('one', 'two', n=1):
        ...     pass

        Or nested allowing you to give different values to the same keyword argument:

        >>> with ureg.context('one', n=1):
        ...     with ureg.context('two', n=2):
        ...         pass

        A nested context inherits the defaults from the containing context:

        >>> with ureg.context('one', n=1):
        ...     # Here n takes the value of the outer context
        ...     with ureg.context('two'):
        ...         pass
        """
        return self._subregistry.context(*names, **kwargs)

    def with_context(self, name: str, **kwargs: Any) -> Callable[[F], F]:
        """Decorator to wrap a function call in a Pint context.

        Use it to ensure that a certain context is active when
        calling a function.

        Parameters
        ----------
        name :
            name of the context.
        **kwargs :
            keyword arguments for the context


        Returns
        -------
        callable: the wrapped function.

        Examples
        --------
        >>> @ureg.with_context('sp')
        ... def my_cool_fun(wavelength):
        ...     print('This wavelength is equivalent to: %s', wavelength.to('terahertz'))
        """
        return self._subregistry.with_context(name, **kwargs)

    ##########
    # General
    ##########

    def __deepcopy__(self: Self, memo) -> type[Self]:
        new = object.__new__(type(self))
        new.__dict__ = copy.deepcopy(self.__dict__, memo)
        new._subregistry._init_dynamic_classes()
        return new

    def __getattr__(self, item: str) -> QuantityT:
        getattr_maybe_raise(self._subregistry, item)

        # self.Unit will call parse_units
        return self._subregistry.__getattr__(item)

    def __getitem__(self, item: str) -> UnitT:
        logger.warning(
            "Calling the getitem method from a UnitRegistry is deprecated. "
            "use `parse_expression` method or use the registry as a callable."
        )
        return self._subregistry.parse_expression(item)

    def __contains__(self, item: str) -> bool:
        """Support checking prefixed units with the `in` operator"""
        return self._subregistry.__contains__(item)

    def __dir__(self) -> list[str]:
        #: Calling dir(registry) gives all units, methods, and attributes.
        #: Also used for autocompletion in IPython.
        return self._subregistry.__dir__()

    def __iter__(self) -> Iterator[str]:
        """Allows for listing all units in registry with `list(ureg)`.

        Returns
        -------
        Iterator over names of all units in registry, ordered alphabetically.
        """
        return self._subregistry.__iter__()

    def define(self, definition: Union[str, type]) -> None:
        """Add unit to the registry.

        Parameters
        ----------
        definition : str or Definition
            a dimension, unit or prefix definition.
        """
        return self._subregistry.define(definition)

    def parse_unit_name(
        self, unit_name: str, case_sensitive: Optional[bool] = None
    ) -> tuple[tuple[str, str, str], ...]:
        """Parse a unit to identify prefix, unit name and suffix
        by walking the list of prefix and suffix.
        In case of equivalent combinations (e.g. ('kilo', 'gram', '') and
        ('', 'kilogram', ''), prefer those with prefix.

        Parameters
        ----------
        unit_name :

        case_sensitive : bool or None
            Control if unit lookup is case sensitive. Defaults to None, which uses the
            registry's case_sensitive setting

        Returns
        -------
        tuple of tuples (str, str, str)
            all non-equivalent combinations of (prefix, unit name, suffix)
        """
        return self._subregistry.parse_unit_name(unit_name, case_sensitive)

    def parse_units_as_container(
        self,
        input_string: str,
        as_delta: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
    ) -> UnitT:
        return self._subregistry.parse_units_as_container(
            input_string, as_delta, case_sensitive
        )

    def parse_units(
        self,
        input_string: str,
        as_delta: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
    ) -> UnitT:
        """Parse a units expression and returns a UnitContainer with
        the canonical names.

        The expression can only contain products, ratios and powers of units.

        Parameters
        ----------
        input_string : str
        as_delta : bool or None
            if the expression has multiple units, the parser will
            interpret non multiplicative units as their `delta_` counterparts. (Default value = None)
        case_sensitive : bool or None
            Control if unit parsing is case sensitive. Defaults to None, which uses the
            registry's setting.

        Returns
        -------
            pint.Unit

        """
        return self._subregistry.parse_units(input_string, as_delta, case_sensitive)

    def parse_expression(
        self,
        input_string: str,
        case_sensitive: Optional[bool] = None,
        **values: QuantityArgument,
    ) -> QuantityT:
        """Parse a mathematical expression including units and return a quantity object.

        Numerical constants can be specified as keyword arguments and will take precedence
        over the names defined in the registry.

        Parameters
        ----------
        input_string

        case_sensitive, optional
            If true, a case sensitive matching of the unit name will be done in the registry.
            If false, a case INsensitive matching of the unit name will be done in the registry.
            (Default value = None, which uses registry setting)
        **values
            Other string that will be parsed using the Quantity constructor on their corresponding value.
        """
        return self._subregistry.parse_expression(
            input_string, case_sensitive, **values
        )

    def parse_pattern(
        self,
        input_string: str,
        pattern: str,
        case_sensitive: Optional[bool] = None,
        many: bool = False,
    ) -> Optional[Union[list[str], str]]:
        """Parse a string with a given regex pattern and returns result.

        Parameters
        ----------
        input_string

        pattern_string:
            The regex parse string
        case_sensitive, optional
            If true, a case sensitive matching of the unit name will be done in the registry.
            If false, a case INsensitive matching of the unit name will be done in the registry.
            (Default value = None, which uses registry setting)
        many, optional
             Match many results
             (Default value = False)
        """
        return self._subregistry.parse_pattern(
            input_string, pattern, case_sensitive, many
        )

    def convert(
        self,
        value: T,
        src: QuantityOrUnitLike,
        dst: QuantityOrUnitLike,
        inplace: bool = False,
    ) -> T:
        """Convert value from some source to destination units.

        Parameters
        ----------
        value :
            value
        src : pint.Quantity or str
            source units.
        dst : pint.Quantity or str
            destination units.
        inplace :
             (Default value = False)

        Returns
        -------
        type
            converted value

        """
        return self._subregistry.convert(value, src, dst, inplace)

    def get_name(
        self, name_or_alias: str, case_sensitive: Optional[bool] = None
    ) -> str:
        """Return the canonical name of a unit."""
        return self._subregistry.get_name(name_or_alias, case_sensitive)

    def get_symbol(
        self, name_or_alias: str, case_sensitive: Optional[bool] = None
    ) -> str:
        """Return the preferred alias for a unit."""
        return self._subregistry.get_symbol(name_or_alias, case_sensitive)

    def get_root_units(
        self, input_units: UnitLike, check_nonmult: bool = True
    ) -> tuple[Scalar, UnitT]:
        """Convert unit or dict of units to the root units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        Parameters
        ----------
        input_units : UnitsContainer or str
            units
        check_nonmult : bool
            if True, None will be returned as the
            multiplicative factor if a non-multiplicative
            units is found in the final Units. (Default value = True)

        Returns
        -------
        Number, pint.Unit
            multiplicative factor, plain units

        """
        return self._subregistry.get_root_units(input_units, check_nonmult)

    def get_dimensionality(self, input_units: UnitLike) -> UnitsContainer:
        """Convert unit or dict of units or dimensions to a dict of plain dimensions
        dimensions
        """
        return self._subregistry.get_dimensionality(input_units)

    def get_base_units(
        self,
        input_units: Union[UnitsContainer, str],
        check_nonmult: bool = True,
        system: Optional[Union[str, System]] = None,
    ) -> tuple[Scalar, UnitT]:
        """Convert unit or dict of units to the plain units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        Parameters
        ----------
        input_units : UnitsContainer or str
            units
        check_nonmult : bool
            If True, None will be returned as the multiplicative factor if
            non-multiplicative units are found in the final Units.
            (Default value = True)
        system :
             (Default value = None)

        Returns
        -------
        Number, pint.Unit
            multiplicative factor, plain units

        """
        return self._subregistry.get_base_units(input_units, check_nonmult, system)

    def get_compatible_units(
        self, input_units: QuantityOrUnitLike, group_or_system: Optional[str] = None
    ) -> frozenset[UnitT]:
        """ """
        return self._subregistry.get_compatible_units(input_units, group_or_system)

    def get_system(self, name: str, create_if_needed: bool = True) -> System:
        """Return a Group.

        Parameters
        ----------
        name : str
            Name of the group to be.
        create_if_needed : bool
            If True, create a group if not found. If False, raise an Exception.
            (Default value = True)

        Returns
        -------
        type
            System

        """
        return self._subregistry.get_system(name, create_if_needed)

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
        return self._subregistry.get_group(name, create_if_needed)

    def load_definitions(
        self, file: Union[Iterable[str], str, pathlib.Path], is_resource: bool = False
    ):
        """Add units and prefixes defined in a definition text file.

        Parameters
        ----------
        file :
            can be a filename or a line iterable.
        is_resource :
            used to indicate that the file is a resource file
            and therefore should be loaded from the package. (Default value = False)
        """
        return self._subregistry.load_definitions(file, is_resource)

    __call__ = parse_expression


class LazyRegistry:
    def __init__(self, args=None, kwargs=None):
        self.__dict__["params"] = args or (), kwargs or {}

    def __init(self):
        args, kwargs = self.__dict__["params"]
        kwargs["on_redefinition"] = "raise"
        self.__class__ = UnitRegistry
        self.__init__(*args, **kwargs)
        # self._subregistry._after_init()

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


# LazyRegistry = UnitRegistry


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
