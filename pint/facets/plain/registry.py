"""
    pint.facets.plain.registry
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import copy
import functools
import inspect
import itertools
import locale
import pathlib
import re
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from numbers import Number
from token import NAME, NUMBER
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from ..context import Context
    from pint import Quantity, Unit

from ..._typing import QuantityOrUnitLike, UnitLike
from ..._vendor import appdirs
from ...compat import HAS_BABEL, babel_parse, tokenizer
from ...errors import DimensionalityError, RedefinitionError, UndefinedUnitError
from ...pint_eval import build_eval_tree
from ...util import ParserHelper
from ...util import UnitsContainer
from ...util import UnitsContainer as UnitsContainerT
from ...util import (
    _is_dim,
    build_dependent_class,
    create_class_with_registry,
    getattr_maybe_raise,
    logger,
    solve_dependencies,
    string_preprocessor,
    to_units_container,
)
from .definitions import (
    AliasDefinition,
    CommentDefinition,
    DefaultsDefinition,
    DerivedDimensionDefinition,
    DimensionDefinition,
    PrefixDefinition,
    UnitDefinition,
)
from .objects import PlainQuantity, PlainUnit

if TYPE_CHECKING:

    if HAS_BABEL:
        import babel

        Locale = babel.Locale
    else:
        Locale = None

T = TypeVar("T")

_BLOCK_RE = re.compile(r"[ (]")


@functools.lru_cache()
def pattern_to_regex(pattern):
    if hasattr(pattern, "finditer"):
        pattern = pattern.pattern

    # Replace "{unit_name}" match string with float regex with unit_name as group
    pattern = re.sub(
        r"{(\w+)}", r"(?P<\1>[+-]?[0-9]+(?:.[0-9]+)?(?:[Ee][+-]?[0-9]+)?)", pattern
    )

    return re.compile(pattern)


NON_INT_TYPE = Type[Union[float, Decimal, Fraction]]
PreprocessorType = Callable[[str], str]


class RegistryCache:
    """Cache to speed up unit registries"""

    def __init__(self) -> None:
        #: Maps dimensionality (UnitsContainer) to Units (str)
        self.dimensional_equivalents: Dict[UnitsContainer, Set[str]] = {}
        #: Maps dimensionality (UnitsContainer) to Dimensionality (UnitsContainer)
        self.root_units = {}
        #: Maps dimensionality (UnitsContainer) to Units (UnitsContainer)
        self.dimensionality: Dict[UnitsContainer, UnitsContainer] = {}
        #: Cache the unit name associated to user input. ('mV' -> 'millivolt')
        self.parse_unit: Dict[str, UnitsContainer] = {}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        attrs = (
            "dimensional_equivalents",
            "root_units",
            "dimensionality",
            "parse_unit",
        )
        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs)


class RegistryMeta(type):
    """This is just to call after_init at the right time
    instead of asking the developer to do it when subclassing.
    """

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj._after_init()
        return obj


class PlainRegistry(metaclass=RegistryMeta):
    """Base class for all registries.

    Capabilities:

    - Register units, prefixes, and dimensions, and their relations.
    - Convert between units.
    - Find dimensionality of a unit.
    - Parse units with prefix and/or suffix.
    - Parse expressions.
    - Parse a definition file.
    - Allow extending the definition file parser by registering @ directives.

    Parameters
    ----------
    filename : str or None
        path of the units definition file to load or line iterable object. Empty to load
        the default definition file. None to leave the UnitRegistry empty.
    force_ndarray : bool
        convert any input, scalar or not to a numpy.ndarray.
    force_ndarray_like : bool
        convert all inputs other than duck arrays to a numpy.ndarray.
    on_redefinition : str
        action to take in case a unit is redefined: 'warn', 'raise', 'ignore'
    auto_reduce_dimensions :
        If True, reduce dimensionality on appropriate operations.
    preprocessors :
        list of callables which are iteratively ran on any input expression or unit
        string
    fmt_locale :
        locale identifier string, used in `format_babel`
    non_int_type : type
        numerical type used for non integer values. (Default: float)
    case_sensitive : bool, optional
        Control default case sensitivity of unit parsing. (Default: True)
    cache_folder : str or pathlib.Path or None, optional
        Specify the folder in which cache files are saved and loaded from.
        If None, the cache is disabled. (default)
    separate_format_defaults : bool, optional
        Separate the default format into magnitude and unit formats as soon as
        possible. The deprecated default is not to separate. This will change in a
        future release.
    """

    #: Babel.Locale instance or None
    fmt_locale: Optional[Locale] = None

    _diskcache = None

    _quantity_class = PlainQuantity
    _unit_class = PlainUnit

    _def_parser = None

    def __init__(
        self,
        filename="",
        force_ndarray: bool = False,
        force_ndarray_like: bool = False,
        on_redefinition: str = "warn",
        auto_reduce_dimensions: bool = False,
        preprocessors: Optional[List[PreprocessorType]] = None,
        fmt_locale: Optional[str] = None,
        non_int_type: NON_INT_TYPE = float,
        case_sensitive: bool = True,
        cache_folder: Union[str, pathlib.Path, None] = None,
        separate_format_defaults: Optional[bool] = None,
    ):
        #: Map a definition class to a adder methods.
        self._adders = dict()
        self._register_definition_adders()
        self._init_dynamic_classes()

        if cache_folder == ":auto:":
            cache_folder = appdirs.user_cache_dir(appname="pint", appauthor=False)
            cache_folder = pathlib.Path(cache_folder)

        from ... import delegates  # TODO: change thiss

        if cache_folder is not None:
            self._diskcache = delegates.build_disk_cache_class(non_int_type)(
                cache_folder
            )

        self._def_parser = delegates.txt_defparser.DefParser(
            delegates.ParserConfig(non_int_type), diskcache=self._diskcache
        )

        self._filename = filename
        self.force_ndarray = force_ndarray
        self.force_ndarray_like = force_ndarray_like
        self.preprocessors = preprocessors or []

        #: mode used to fill in the format defaults
        self.separate_format_defaults = separate_format_defaults

        #: Action to take in case a unit is redefined. 'warn', 'raise', 'ignore'
        self._on_redefinition = on_redefinition

        #: Determines if dimensionality should be reduced on appropriate operations.
        self.auto_reduce_dimensions = auto_reduce_dimensions

        #: Default locale identifier string, used when calling format_babel without explicit locale.
        self.set_fmt_locale(fmt_locale)

        #: Numerical type used for non integer values.
        self._non_int_type = non_int_type

        #: Default unit case sensitivity
        self.case_sensitive = case_sensitive

        #: Map between name (string) and value (string) of defaults stored in the
        #: definitions file.
        self._defaults: Dict[str, str] = {}

        #: Map dimension name (string) to its definition (DimensionDefinition).
        self._dimensions: Dict[
            str, Union[DimensionDefinition, DerivedDimensionDefinition]
        ] = {}

        #: Map unit name (string) to its definition (UnitDefinition).
        #: Might contain prefixed units.
        self._units: Dict[str, UnitDefinition] = {}

        #: Map unit name in lower case (string) to a set of unit names with the right
        #: case.
        #: Does not contain prefixed units.
        #: e.g: 'hz' - > set('Hz', )
        self._units_casei: Dict[str, Set[str]] = defaultdict(set)

        #: Map prefix name (string) to its definition (PrefixDefinition).
        self._prefixes: Dict[str, PrefixDefinition] = {"": PrefixDefinition("", 1)}

        #: Map suffix name (string) to canonical , and unit alias to canonical unit name
        self._suffixes: Dict[str, str] = {"": "", "s": ""}

        #: Map contexts to RegistryCache
        self._cache = RegistryCache()

        self._initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
        cls.Unit = build_dependent_class(cls, "Unit", "_unit_class")
        cls.Quantity = build_dependent_class(cls, "Quantity", "_quantity_class")

    def _init_dynamic_classes(self) -> None:
        """Generate subclasses on the fly and attach them to self"""

        self.Unit = create_class_with_registry(self, self.Unit)
        self.Quantity = create_class_with_registry(self, self.Quantity)

    def _after_init(self) -> None:
        """This should be called after all __init__"""

        if self._filename == "":
            path = pathlib.Path(__file__).parent.parent.parent / "default_en.txt"
            loaded_files = self.load_definitions(path, True)
        elif self._filename is not None:
            loaded_files = self.load_definitions(self._filename)
        else:
            loaded_files = None

        self._build_cache(loaded_files)
        self._initialized = True

    def _register_adder(self, definition_class, adder_func):
        """Register a block definition."""
        self._adders[definition_class] = adder_func

    def _register_definition_adders(self) -> None:
        self._register_adder(AliasDefinition, self._add_alias)
        self._register_adder(DefaultsDefinition, self._add_defaults)
        self._register_adder(CommentDefinition, lambda o: o)
        self._register_adder(PrefixDefinition, self._add_prefix)
        self._register_adder(UnitDefinition, self._add_unit)
        self._register_adder(DimensionDefinition, self._add_dimension)
        self._register_adder(DerivedDimensionDefinition, self._add_derived_dimension)

    def __deepcopy__(self, memo) -> "PlainRegistry":
        new = object.__new__(type(self))
        new.__dict__ = copy.deepcopy(self.__dict__, memo)
        new._init_dynamic_classes()
        return new

    def __getattr__(self, item):
        getattr_maybe_raise(self, item)
        return self.Unit(item)

    def __getitem__(self, item):
        logger.warning(
            "Calling the getitem method from a UnitRegistry is deprecated. "
            "use `parse_expression` method or use the registry as a callable."
        )
        return self.parse_expression(item)

    def __contains__(self, item) -> bool:
        """Support checking prefixed units with the `in` operator"""
        try:
            self.__getattr__(item)
            return True
        except UndefinedUnitError:
            return False

    def __dir__(self) -> List[str]:
        #: Calling dir(registry) gives all units, methods, and attributes.
        #: Also used for autocompletion in IPython.
        return list(self._units.keys()) + list(object.__dir__(self))

    def __iter__(self) -> Iterator[str]:
        """Allows for listing all units in registry with `list(ureg)`.

        Returns
        -------
        Iterator over names of all units in registry, ordered alphabetically.
        """
        return iter(sorted(self._units.keys()))

    def set_fmt_locale(self, loc: Optional[str]) -> None:
        """Change the locale used by default by `format_babel`.

        Parameters
        ----------
        loc : str or None
            None` (do not translate), 'sys' (detect the system locale) or a locale id string.
        """
        if isinstance(loc, str):
            if loc == "sys":
                loc = locale.getdefaultlocale()[0]

            # We call babel parse to fail here and not in the formatting operation
            babel_parse(loc)

        self.fmt_locale = loc

    def UnitsContainer(self, *args, **kwargs) -> UnitsContainerT:
        return UnitsContainer(*args, non_int_type=self.non_int_type, **kwargs)

    @property
    def default_format(self) -> str:
        """Default formatting string for quantities."""
        return self.Quantity.default_format

    @default_format.setter
    def default_format(self, value: str):
        self.Unit.default_format = value
        self.Quantity.default_format = value
        self.Measurement.default_format = value

    @property
    def cache_folder(self) -> Optional[pathlib.Path]:
        if self._diskcache:
            return self._diskcache.cache_folder
        return None

    @property
    def non_int_type(self):
        return self._non_int_type

    def define(self, definition):
        """Add unit to the registry.

        Parameters
        ----------
        definition : str or Definition
            a dimension, unit or prefix definition.
        """

        if isinstance(definition, str):
            parsed_project = self._def_parser.parse_string(definition)

            for definition in self._def_parser.iter_parsed_project(parsed_project):
                self._helper_dispatch_adder(definition)
        else:
            self._helper_dispatch_adder(definition)

    ############
    # Adders
    # - we first provide some helpers that deal with repetitive task.
    # - then we define specific adder for each definition class. :-D
    ############

    def _helper_dispatch_adder(self, definition):
        """Helper function to add a single definition,
        choosing the appropiate method by class.
        """
        for cls in inspect.getmro(definition.__class__):
            if cls in self._adders:
                adder_func = self._adders[cls]
                break
        else:
            raise TypeError(
                f"No loader function defined " f"for {definition.__class__.__name__}"
            )

        adder_func(definition)

    def _helper_adder(self, definition, target_dict, casei_target_dict):
        """Helper function to store a definition in the internal dictionaries.
        It stores the definition under its name, symbol and aliases.
        """
        self._helper_single_adder(
            definition.name, definition, target_dict, casei_target_dict
        )

        if getattr(definition, "has_symbol", ""):
            self._helper_single_adder(
                definition.symbol, definition, target_dict, casei_target_dict
            )

        for alias in getattr(definition, "aliases", ()):
            if " " in alias:
                logger.warn("Alias cannot contain a space: " + alias)

            self._helper_single_adder(alias, definition, target_dict, casei_target_dict)

    def _helper_single_adder(self, key, value, target_dict, casei_target_dict):
        """Helper function to store a definition in the internal dictionaries.

        It warns or raise error on redefinition.
        """
        if key in target_dict:
            if self._on_redefinition == "raise":
                raise RedefinitionError(key, type(value))
            elif self._on_redefinition == "warn":
                logger.warning("Redefining '%s' (%s)" % (key, type(value)))

        target_dict[key] = value
        if casei_target_dict is not None:
            casei_target_dict[key.lower()].add(key)

    def _add_defaults(self, defaults_definition: DefaultsDefinition):
        for k, v in defaults_definition.items():
            self._defaults[k] = v

    def _add_alias(self, definition: AliasDefinition):
        unit_dict = self._units
        unit = unit_dict[definition.name]
        while not isinstance(unit, UnitDefinition):
            unit = unit_dict[unit.name]
        for alias in definition.aliases:
            self._helper_single_adder(alias, unit, self._units, self._units_casei)

    def _add_dimension(self, definition: DimensionDefinition):
        self._helper_adder(definition, self._dimensions, None)

    def _add_derived_dimension(self, definition: DerivedDimensionDefinition):
        for dim_name in definition.reference.keys():
            if dim_name not in self._dimensions:
                self._add_dimension(DimensionDefinition(dim_name))
        self._helper_adder(definition, self._dimensions, None)

    def _add_prefix(self, definition: PrefixDefinition):
        self._helper_adder(definition, self._prefixes, None)

    def _add_unit(self, definition: UnitDefinition):
        if definition.is_base:
            for dim_name in definition.reference.keys():
                if dim_name not in self._dimensions:
                    self._add_dimension(DimensionDefinition(dim_name))

        self._helper_adder(definition, self._units, self._units_casei)

    def load_definitions(self, file, is_resource: bool = False):
        """Add units and prefixes defined in a definition text file.

        Parameters
        ----------
        file :
            can be a filename or a line iterable.
        is_resource :
            used to indicate that the file is a resource file
            and therefore should be loaded from the package. (Default value = False)
        """

        if isinstance(file, (list, tuple)):
            # TODO: this hack was to keep it backwards compatible.
            parsed_project = self._def_parser.parse_string("\n".join(file))
        else:
            parsed_project = self._def_parser.parse_file(file)

        for definition in self._def_parser.iter_parsed_project(parsed_project):
            self._helper_dispatch_adder(definition)

        return parsed_project

    def _build_cache(self, loaded_files=None) -> None:
        """Build a cache of dimensionality and plain units."""

        diskcache = self._diskcache
        if loaded_files and diskcache:
            cache, cache_basename = diskcache.load(loaded_files, "build_cache")
            if cache is None:
                self._build_cache()
                diskcache.save(self._cache, loaded_files, "build_cache")
            return

        self._cache = RegistryCache()

        deps = {
            name: definition.reference.keys() if definition.reference else set()
            for name, definition in self._units.items()
        }

        for unit_names in solve_dependencies(deps):
            for unit_name in unit_names:
                if "[" in unit_name:
                    continue
                parsed_names = self.parse_unit_name(unit_name)
                if parsed_names:
                    prefix, base_name, _ = parsed_names[0]
                else:
                    prefix, base_name = "", unit_name

                try:
                    uc = ParserHelper.from_word(base_name, self.non_int_type)

                    bu = self._get_root_units(uc)
                    di = self._get_dimensionality(uc)

                    self._cache.root_units[uc] = bu
                    self._cache.dimensionality[uc] = di

                    if not prefix:
                        dimeq_set = self._cache.dimensional_equivalents.setdefault(
                            di, set()
                        )
                        dimeq_set.add(self._units[base_name].name)

                except Exception as exc:
                    logger.warning(f"Could not resolve {unit_name}: {exc!r}")
        return self._cache

    def get_name(
        self, name_or_alias: str, case_sensitive: Optional[bool] = None
    ) -> str:
        """Return the canonical name of a unit."""

        if name_or_alias == "dimensionless":
            return ""

        try:
            return self._units[name_or_alias].name
        except KeyError:
            pass

        candidates = self.parse_unit_name(name_or_alias, case_sensitive)
        if not candidates:
            raise UndefinedUnitError(name_or_alias)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning(
                "Parsing {} yield multiple results. "
                "Options are: {}".format(name_or_alias, candidates)
            )
            prefix, unit_name, _ = candidates[0]

        if prefix:
            name = prefix + unit_name
            symbol = self.get_symbol(name, case_sensitive)
            prefix_def = self._prefixes[prefix]
            self._units[name] = UnitDefinition(
                name,
                symbol,
                (),
                prefix_def.converter,
                self.UnitsContainer({unit_name: 1}),
            )
            return prefix + unit_name

        return unit_name

    def get_symbol(
        self, name_or_alias: str, case_sensitive: Optional[bool] = None
    ) -> str:
        """Return the preferred alias for a unit."""
        candidates = self.parse_unit_name(name_or_alias, case_sensitive)
        if not candidates:
            raise UndefinedUnitError(name_or_alias)
        elif len(candidates) == 1:
            prefix, unit_name, _ = candidates[0]
        else:
            logger.warning(
                "Parsing {0} yield multiple results. "
                "Options are: {1!r}".format(name_or_alias, candidates)
            )
            prefix, unit_name, _ = candidates[0]

        return self._prefixes[prefix].symbol + self._units[unit_name].symbol

    def _get_symbol(self, name: str) -> str:
        return self._units[name].symbol

    def get_dimensionality(self, input_units) -> UnitsContainerT:
        """Convert unit or dict of units or dimensions to a dict of plain dimensions
        dimensions
        """

        # TODO: This should be to_units_container(input_units, self)
        # but this tries to reparse and fail for dimensions.
        input_units = to_units_container(input_units)

        return self._get_dimensionality(input_units)

    def _get_dimensionality(
        self, input_units: Optional[UnitsContainerT]
    ) -> UnitsContainerT:
        """Convert a UnitsContainer to plain dimensions."""
        if not input_units:
            return self.UnitsContainer()

        cache = self._cache.dimensionality

        try:
            return cache[input_units]
        except KeyError:
            pass

        accumulator = defaultdict(int)
        self._get_dimensionality_recurse(input_units, 1, accumulator)

        if "[]" in accumulator:
            del accumulator["[]"]

        dims = self.UnitsContainer({k: v for k, v in accumulator.items() if v != 0})

        cache[input_units] = dims

        return dims

    def _get_dimensionality_recurse(self, ref, exp, accumulator):
        for key in ref:
            exp2 = exp * ref[key]
            if _is_dim(key):
                reg = self._dimensions[key]
                if reg.is_base:
                    accumulator[key] += exp2
                elif reg.reference is not None:
                    self._get_dimensionality_recurse(reg.reference, exp2, accumulator)
            else:
                reg = self._units[self.get_name(key)]
                if reg.reference is not None:
                    self._get_dimensionality_recurse(reg.reference, exp2, accumulator)

    def _get_dimensionality_ratio(self, unit1, unit2):
        """Get the exponential ratio between two units, i.e. solve unit2 = unit1**x for x.

        Parameters
        ----------
        unit1 : UnitsContainer compatible (str, Unit, UnitsContainer, dict)
            first unit
        unit2 : UnitsContainer compatible (str, Unit, UnitsContainer, dict)
            second unit

        Returns
        -------
        number or None
            exponential proportionality or None if the units cannot be converted

        """
        # shortcut in case of equal units
        if unit1 == unit2:
            return 1

        dim1, dim2 = (self.get_dimensionality(unit) for unit in (unit1, unit2))
        if dim1 == dim2:
            return 1
        elif not dim1 or not dim2 or dim1.keys() != dim2.keys():  # not comparable
            return None

        ratios = (dim2[key] / val for key, val in dim1.items())
        first = next(ratios)
        if all(r == first for r in ratios):  # all are same, we're good
            return first
        return None

    def get_root_units(
        self, input_units: UnitLike, check_nonmult: bool = True
    ) -> Tuple[Number, PlainUnit]:
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
        input_units = to_units_container(input_units, self)

        f, units = self._get_root_units(input_units, check_nonmult)

        return f, self.Unit(units)

    def _get_root_units(self, input_units, check_nonmult=True):
        """Convert unit or dict of units to the root units.

        If any unit is non multiplicative and check_converter is True,
        then None is returned as the multiplicative factor.

        Parameters
        ----------
        input_units : UnitsContainer or dict
            units
        check_nonmult : bool
            if True, None will be returned as the
            multiplicative factor if a non-multiplicative
            units is found in the final Units. (Default value = True)

        Returns
        -------
        number, Unit
            multiplicative factor, plain units

        """
        if not input_units:
            return 1, self.UnitsContainer()

        cache = self._cache.root_units
        try:
            return cache[input_units]
        except KeyError:
            pass

        accumulators = [1, defaultdict(int)]
        self._get_root_units_recurse(input_units, 1, accumulators)

        factor = accumulators[0]
        units = self.UnitsContainer(
            {k: v for k, v in accumulators[1].items() if v != 0}
        )

        # Check if any of the final units is non multiplicative and return None instead.
        if check_nonmult:
            if any(not self._units[unit].converter.is_multiplicative for unit in units):
                factor = None

        cache[input_units] = factor, units
        return factor, units

    def get_base_units(self, input_units, check_nonmult=True, system=None):
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

        return self.get_root_units(input_units, check_nonmult)

    def _get_root_units_recurse(self, ref, exp, accumulators):
        for key in ref:
            exp2 = exp * ref[key]
            key = self.get_name(key)
            reg = self._units[key]
            if reg.is_base:
                accumulators[1][key] += exp2
            else:
                accumulators[0] *= reg.converter.scale**exp2
                if reg.reference is not None:
                    self._get_root_units_recurse(reg.reference, exp2, accumulators)

    def get_compatible_units(
        self, input_units, group_or_system=None
    ) -> FrozenSet[Unit]:
        """ """
        input_units = to_units_container(input_units)

        equiv = self._get_compatible_units(input_units, group_or_system)

        return frozenset(self.Unit(eq) for eq in equiv)

    def _get_compatible_units(self, input_units, group_or_system):
        """ """
        if not input_units:
            return frozenset()

        src_dim = self._get_dimensionality(input_units)
        return self._cache.dimensional_equivalents[src_dim]

    # TODO: remove context from here
    def is_compatible_with(
        self, obj1: Any, obj2: Any, *contexts: Union[str, Context], **ctx_kwargs
    ) -> bool:
        """check if the other object is compatible

        Parameters
        ----------
        obj1, obj2
            The objects to check against each other. Treated as
            dimensionless if not a Quantity, Unit or str.
        *contexts : str or pint.Context
            Contexts to use in the transformation.
        **ctx_kwargs :
            Values for the Context/s

        Returns
        -------
        bool
        """
        if isinstance(obj1, (self.Quantity, self.Unit)):
            return obj1.is_compatible_with(obj2, *contexts, **ctx_kwargs)

        if isinstance(obj1, str):
            return self.parse_expression(obj1).is_compatible_with(
                obj2, *contexts, **ctx_kwargs
            )

        return not isinstance(obj2, (self.Quantity, self.Unit))

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
        src = to_units_container(src, self)

        dst = to_units_container(dst, self)

        if src == dst:
            return value

        return self._convert(value, src, dst, inplace)

    def _convert(self, value, src, dst, inplace=False, check_dimensionality=True):
        """Convert value from some source to destination units.

        Parameters
        ----------
        value :
            value
        src : UnitsContainer
            source units.
        dst : UnitsContainer
            destination units.
        inplace :
             (Default value = False)
        check_dimensionality :
             (Default value = True)

        Returns
        -------
        type
            converted value

        """

        if check_dimensionality:

            src_dim = self._get_dimensionality(src)
            dst_dim = self._get_dimensionality(dst)

            # If the source and destination dimensionality are different,
            # then the conversion cannot be performed.
            if src_dim != dst_dim:
                raise DimensionalityError(src, dst, src_dim, dst_dim)

        # Here src and dst have only multiplicative units left. Thus we can
        # convert with a factor.
        factor, _ = self._get_root_units(src / dst)

        # factor is type float and if our magnitude is type Decimal then
        # must first convert to Decimal before we can '*' the values
        if isinstance(value, Decimal):
            factor = Decimal(str(factor))
        elif isinstance(value, Fraction):
            factor = Fraction(str(factor))

        if inplace:
            value *= factor
        else:
            value = value * factor

        return value

    def parse_unit_name(
        self, unit_name: str, case_sensitive: Optional[bool] = None
    ) -> Tuple[Tuple[str, str, str], ...]:
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
        return self._dedup_candidates(
            self._parse_unit_name(unit_name, case_sensitive=case_sensitive)
        )

    def _parse_unit_name(
        self, unit_name: str, case_sensitive: Optional[bool] = None
    ) -> Iterator[Tuple[str, str, str]]:
        """Helper of parse_unit_name."""
        case_sensitive = (
            self.case_sensitive if case_sensitive is None else case_sensitive
        )
        stw = unit_name.startswith
        edw = unit_name.endswith
        for suffix, prefix in itertools.product(self._suffixes, self._prefixes):
            if stw(prefix) and edw(suffix):
                name = unit_name[len(prefix) :]
                if suffix:
                    name = name[: -len(suffix)]
                    if len(name) == 1:
                        continue
                if case_sensitive:
                    if name in self._units:
                        yield (
                            self._prefixes[prefix].name,
                            self._units[name].name,
                            self._suffixes[suffix],
                        )
                else:
                    for real_name in self._units_casei.get(name.lower(), ()):
                        yield (
                            self._prefixes[prefix].name,
                            self._units[real_name].name,
                            self._suffixes[suffix],
                        )

    @staticmethod
    def _dedup_candidates(
        candidates: Iterable[Tuple[str, str, str]]
    ) -> Tuple[Tuple[str, str, str], ...]:
        """Helper of parse_unit_name.

        Given an iterable of unit triplets (prefix, name, suffix), remove those with
        different names but equal value, preferring those with a prefix.

        e.g. ('kilo', 'gram', '') and ('', 'kilogram', '')
        """
        candidates = dict.fromkeys(candidates)  # ordered set
        for cp, cu, cs in list(candidates):
            assert isinstance(cp, str)
            assert isinstance(cu, str)
            if cs != "":
                raise NotImplementedError("non-empty suffix")
            if cp:
                candidates.pop(("", cp + cu, ""), None)
        return tuple(candidates)

    def parse_units(
        self,
        input_string: str,
        as_delta: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
    ) -> Unit:
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
        for p in self.preprocessors:
            input_string = p(input_string)
        units = self._parse_units(input_string, as_delta, case_sensitive)
        return self.Unit(units)

    def _parse_units(
        self,
        input_string: str,
        as_delta: bool = True,
        case_sensitive: Optional[bool] = None,
    ) -> UnitsContainerT:
        """Parse a units expression and returns a UnitContainer with
        the canonical names.
        """

        cache = self._cache.parse_unit
        # Issue #1097: it is possible, when a unit was defined while a different context
        # was active, that the unit is in self._cache.parse_unit but not in self._units.
        # If this is the case, force self._units to be repopulated.
        if as_delta and input_string in cache and input_string in self._units:
            return cache[input_string]

        if not input_string:
            return self.UnitsContainer()

        # Sanitize input_string with whitespaces.
        input_string = input_string.strip()

        units = ParserHelper.from_string(input_string, self.non_int_type)
        if units.scale != 1:
            raise ValueError("Unit expression cannot have a scaling factor.")

        ret = self.UnitsContainer({})
        many = len(units) > 1
        for name in units:
            cname = self.get_name(name, case_sensitive=case_sensitive)
            value = units[name]
            if not cname:
                continue
            if as_delta and (many or (not many and value != 1)):
                definition = self._units[cname]
                if not definition.is_multiplicative:
                    cname = "delta_" + cname
            ret = ret.add(cname, value)

        if as_delta:
            cache[input_string] = ret

        return ret

    def _eval_token(self, token, case_sensitive=None, use_decimal=False, **values):

        # TODO: remove this code when use_decimal is deprecated
        if use_decimal:
            raise DeprecationWarning(
                "`use_decimal` is deprecated, use `non_int_type` keyword argument when instantiating the registry.\n"
                ">>> from decimal import Decimal\n"
                ">>> ureg = UnitRegistry(non_int_type=Decimal)"
            )

        token_type = token[0]
        token_text = token[1]
        if token_type == NAME:
            if token_text == "dimensionless":
                return 1 * self.dimensionless
            elif token_text.lower() in ("inf", "infinity"):
                return self.non_int_type("inf")
            elif token_text.lower() == "nan":
                return self.non_int_type("nan")
            elif token_text in values:
                return self.Quantity(values[token_text])
            else:
                return self.Quantity(
                    1,
                    self.UnitsContainer(
                        {self.get_name(token_text, case_sensitive=case_sensitive): 1}
                    ),
                )
        elif token_type == NUMBER:
            return ParserHelper.eval_token(token, non_int_type=self.non_int_type)
        else:
            raise Exception("unknown token type")

    def parse_pattern(
        self,
        input_string: str,
        pattern: str,
        case_sensitive: Optional[bool] = None,
        use_decimal: bool = False,
        many: bool = False,
    ) -> Union[List[str], str, None]:
        """Parse a string with a given regex pattern and returns result.

        Parameters
        ----------
        input_string :

        pattern_string:
             The regex parse string
        case_sensitive :
             (Default value = None, which uses registry setting)
        use_decimal :
             (Default value = False)
        many :
             Match many results
             (Default value = False)


        Returns
        -------

        """

        if not input_string:
            return [] if many else None

        # Parse string
        pattern = pattern_to_regex(pattern)
        matched = re.finditer(pattern, input_string)

        # Extract result(s)
        results = []
        for match in matched:
            # Extract units from result
            match = match.groupdict()

            # Parse units
            units = []
            for unit, value in match.items():
                # Construct measure by multiplying value by unit
                units.append(
                    float(value)
                    * self.parse_expression(unit, case_sensitive, use_decimal)
                )

            # Add to results
            results.append(units)

            # Return first match only
            if not many:
                return results[0]

        return results

    def parse_expression(
        self,
        input_string: str,
        case_sensitive: Optional[bool] = None,
        use_decimal: bool = False,
        **values,
    ) -> Quantity:
        """Parse a mathematical expression including units and return a quantity object.

        Numerical constants can be specified as keyword arguments and will take precedence
        over the names defined in the registry.

        Parameters
        ----------
        input_string :

        case_sensitive :
             (Default value = None, which uses registry setting)
        use_decimal :
             (Default value = False)
        **values :


        Returns
        -------

        """

        # TODO: remove this code when use_decimal is deprecated
        if use_decimal:
            raise DeprecationWarning(
                "`use_decimal` is deprecated, use `non_int_type` keyword argument when instantiating the registry.\n"
                ">>> from decimal import Decimal\n"
                ">>> ureg = UnitRegistry(non_int_type=Decimal)"
            )

        if not input_string:
            return self.Quantity(1)

        for p in self.preprocessors:
            input_string = p(input_string)
        input_string = string_preprocessor(input_string)
        gen = tokenizer(input_string)

        return build_eval_tree(gen).evaluate(
            lambda x: self._eval_token(x, case_sensitive=case_sensitive, **values)
        )

    __call__ = parse_expression
