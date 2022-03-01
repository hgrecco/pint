"""
    pint.context
    ~~~~~~~~~~~~

    Functions and classes related to context definitions and application.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details..
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numbers
import re
import weakref
from collections import ChainMap, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from .definitions import Definition, UnitDefinition
from .errors import DefinitionSyntaxError, RedefinitionError
from .util import ParserHelper, SourceIterator, to_units_container

if TYPE_CHECKING:
    from .quantity import Quantity
    from .registry import UnitRegistry
    from .util import UnitsContainer

#: Regex to match the header parts of a context.
_header_re = re.compile(
    r"@context\s*(?P<defaults>\(.*\))?\s+(?P<name>\w+)\s*(=(?P<aliases>.*))*"
)

#: Regex to match variable names in an equation.
_varname_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class Expression:
    def __init__(self, eq):
        self._eq = eq

    def __call__(self, ureg: UnitRegistry, value: Any, **kwargs: Any):
        return ureg.parse_expression(self._eq, value=value, **kwargs)


@dataclass(frozen=True)
class Relation:

    bidirectional: True
    src: ParserHelper
    dst: ParserHelper
    tranformation: Callable[..., Quantity[Any]]


@dataclass(frozen=True)
class ContextDefinition:
    """Definition of a Context

        @context[(defaults)] <canonical name> [= <alias>] [= <alias>]
            # units can be redefined within the context
            <redefined unit> = <relation to another unit>

            # can establish unidirectional relationships between dimensions
            <dimension 1> -> <dimension 2>: <transformation function>

            # can establish bidirectionl relationships between dimensions
            <dimension 3> <-> <dimension 4>: <transformation function>
        @end

    Example::

        @context(n=1) spectroscopy = sp
            # n index of refraction of the medium.
            [length] <-> [frequency]: speed_of_light / n / value
            [frequency] -> [energy]: planck_constant * value
            [energy] -> [frequency]: value / planck_constant
            # allow wavenumber / kayser
            [wavenumber] <-> [length]: 1 / value
        @end
    """

    name: str
    aliases: Tuple[str, ...]
    variables: Tuple[str, ...]
    defaults: Dict[str, numbers.Number]

    # Each element indicates: line number, is_bidirectional, src, dst, transformation func
    relations: Tuple[Tuple[int, Relation], ...]
    redefinitions: Tuple[Tuple[int, UnitDefinition], ...]

    @staticmethod
    def parse_definition(line, non_int_type) -> UnitDefinition:
        definition = Definition.from_string(line, non_int_type)
        if not isinstance(definition, UnitDefinition):
            raise DefinitionSyntaxError(
                "Expected <unit> = <converter>; got %s" % line.strip()
            )
        if definition.symbol != definition.name or definition.aliases:
            raise DefinitionSyntaxError(
                "Can't change a unit's symbol or aliases within a context"
            )
        if definition.is_base:
            raise DefinitionSyntaxError("Can't define base units within a context")
        return definition

    @classmethod
    def from_lines(cls, lines, non_int_type=float) -> ContextDefinition:
        lines = SourceIterator(lines)

        lineno, header = next(lines)
        try:
            r = _header_re.search(header)
            name = r.groupdict()["name"].strip()
            aliases = r.groupdict()["aliases"]
            if aliases:
                aliases = tuple(a.strip() for a in r.groupdict()["aliases"].split("="))
            else:
                aliases = ()
            defaults = r.groupdict()["defaults"]
        except Exception as exc:
            raise DefinitionSyntaxError(
                "Could not parse the Context header '%s'" % header, lineno=lineno
            ) from exc

        if defaults:

            def to_num(val):
                val = complex(val)
                if not val.imag:
                    return val.real
                return val

            txt = defaults
            try:
                defaults = (part.split("=") for part in defaults.strip("()").split(","))
                defaults = {str(k).strip(): to_num(v) for k, v in defaults}
            except (ValueError, TypeError) as exc:
                raise DefinitionSyntaxError(
                    f"Could not parse Context definition defaults: '{txt}'",
                    lineno=lineno,
                ) from exc
        else:
            defaults = {}

        variables = set()
        redefitions = []
        relations = []
        for lineno, line in lines:
            try:
                if "=" in line:
                    definition = cls.parse_definition(line, non_int_type)
                    redefitions.append((lineno, definition))
                elif ":" in line:
                    rel, eq = line.split(":")
                    variables.update(_varname_re.findall(eq))

                    func = Expression(eq)

                    bidir = True
                    parts = rel.split("<->")
                    if len(parts) != 2:
                        bidir = False
                        parts = rel.split("->")
                        if len(parts) != 2:
                            raise Exception

                    src, dst = (
                        ParserHelper.from_string(s, non_int_type) for s in parts
                    )
                    relation = Relation(bidir, src, dst, func)
                    relations.append((lineno, relation))
                else:
                    raise Exception
            except Exception as exc:
                raise DefinitionSyntaxError(
                    "Could not parse Context %s relation '%s': %s" % (name, line, exc),
                    lineno=lineno,
                ) from exc

        if defaults:
            missing_pars = defaults.keys() - set(variables)
            if missing_pars:
                raise DefinitionSyntaxError(
                    f"Context parameters {missing_pars} not found in any equation"
                )

        return cls(
            name,
            aliases,
            tuple(variables),
            defaults,
            tuple(relations),
            tuple(redefitions),
        )


class Context:
    """A specialized container that defines transformation functions from one
    dimension to another. Each Dimension are specified using a UnitsContainer.
    Simple transformation are given with a function taking a single parameter.

    Conversion functions may take optional keyword arguments and the context
    can have default values for these arguments.

    Additionally, a context may host redefinitions.

    A redefinition must be performed among units that already exist in the registry. It
    cannot change the dimensionality of a unit. The symbol and aliases are automatically
    inherited from the registry.

    See ContextDefinition for the definition file syntax.

    Parameters
    ----------
    name : str or None, optional
        Name of the context (must be unique within the registry).
        Use None for anonymous Context. (Default value = None).
    aliases : iterable of str
        Other names for the context.
    defaults : None or dict
        Maps variable names to values.

    Example
    -------

    >>> from pint.util import UnitsContainer
    >>> from pint import Context, UnitRegistry
    >>> ureg = UnitRegistry()
    >>> timedim = UnitsContainer({'[time]': 1})
    >>> spacedim = UnitsContainer({'[length]': 1})
    >>> def time_to_len(ureg, time):
    ...     'Time to length converter'
    ...     return 3. * time
    >>> c = Context()
    >>> c.add_transformation(timedim, spacedim, time_to_len)
    >>> c.transform(timedim, spacedim, ureg, 2)
    6.0
    >>> def time_to_len_indexed(ureg, time, n=1):
    ...     'Time to length converter, n is the index of refraction of the material'
    ...     return 3. * time / n
    >>> c = Context(defaults={'n':3})
    >>> c.add_transformation(timedim, spacedim, time_to_len_indexed)
    >>> c.transform(timedim, spacedim, ureg, 2)
    2.0
    >>> c.redefine("pound = 0.5 kg")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        aliases: Tuple[str, ...] = (),
        defaults: Optional[dict] = None,
    ) -> None:

        self.name = name
        self.aliases = aliases

        #: Maps (src, dst) -> transformation function
        self.funcs = {}

        #: Maps defaults variable names to values
        self.defaults = defaults or {}

        # Store Definition objects that are context-specific
        self.redefinitions = []

        # Flag set to True by the Registry the first time the context is enabled
        self.checked = False

        #: Maps (src, dst) -> self
        #: Used as a convenience dictionary to be composed by ContextChain
        self.relation_to_context = weakref.WeakValueDictionary()

    @classmethod
    def from_context(cls, context: Context, **defaults) -> Context:
        """Creates a new context that shares the funcs dictionary with the
        original context. The default values are copied from the original
        context and updated with the new defaults.

        If defaults is empty, return the same context.

        Parameters
        ----------
        context : pint.Context
            Original context.
        **defaults


        Returns
        -------
        pint.Context
        """
        if defaults:
            newdef = dict(context.defaults, **defaults)
            c = cls(context.name, context.aliases, newdef)
            c.funcs = context.funcs
            c.redefinitions = context.redefinitions
            for edge in context.funcs:
                c.relation_to_context[edge] = c
            return c
        return context

    @classmethod
    def from_lines(cls, lines, to_base_func=None, non_int_type=float) -> Context:
        cd = ContextDefinition.from_lines(lines, non_int_type)
        return cls.from_definition(cd, to_base_func)

    @classmethod
    def from_definition(cls, cd: ContextDefinition, to_base_func=None) -> Context:
        ctx = cls(cd.name, cd.aliases, cd.defaults)

        for lineno, definition in cd.redefinitions:
            try:
                ctx._redefine(definition)
            except (RedefinitionError, DefinitionSyntaxError) as ex:
                if ex.lineno is None:
                    ex.lineno = lineno
                raise ex

        for lineno, relation in cd.relations:
            try:
                if to_base_func:
                    src = to_base_func(relation.src)
                    dst = to_base_func(relation.dst)
                else:
                    src, dst = relation.src, relation.dst
                ctx.add_transformation(src, dst, relation.tranformation)
                if relation.bidirectional:
                    ctx.add_transformation(dst, src, relation.tranformation)
            except Exception as exc:
                raise DefinitionSyntaxError(
                    "Could not add Context %s relation on line '%s'"
                    % (cd.name, lineno),
                    lineno=lineno,
                ) from exc

        return ctx

    def add_transformation(self, src, dst, func) -> None:
        """Add a transformation function to the context."""

        _key = self.__keytransform__(src, dst)
        self.funcs[_key] = func
        self.relation_to_context[_key] = self

    def remove_transformation(self, src, dst) -> None:
        """Add a transformation function to the context."""

        _key = self.__keytransform__(src, dst)
        del self.funcs[_key]
        del self.relation_to_context[_key]

    @staticmethod
    def __keytransform__(src, dst) -> Tuple[UnitsContainer, UnitsContainer]:
        return to_units_container(src), to_units_container(dst)

    def transform(self, src, dst, registry, value):
        """Transform a value."""

        _key = self.__keytransform__(src, dst)
        return self.funcs[_key](registry, value, **self.defaults)

    def redefine(self, definition: str) -> None:
        """Override the definition of a unit in the registry.

        Parameters
        ----------
        definition : str
            <unit> = <new definition>``, e.g. ``pound = 0.5 kg``
        """

        for line in definition.splitlines():
            # TODO: What is the right non_int_type value.
            definition = ContextDefinition.parse_definition(line, float)
            self._redefine(definition)

    def _redefine(self, definition: UnitDefinition):
        self.redefinitions.append(definition)

    def hashable(
        self,
    ) -> Tuple[Optional[str], Tuple[str, ...], frozenset, frozenset, tuple]:
        """Generate a unique hashable and comparable representation of self, which can
        be used as a key in a dict. This class cannot define ``__hash__`` because it is
        mutable, and the Python interpreter does cache the output of ``__hash__``.

        Returns
        -------
        tuple
        """
        return (
            self.name,
            tuple(self.aliases),
            frozenset((k, id(v)) for k, v in self.funcs.items()),
            frozenset(self.defaults.items()),
            tuple(self.redefinitions),
        )


class ContextChain(ChainMap):
    """A specialized ChainMap for contexts that simplifies finding rules
    to transform from one dimension to another.
    """

    def __init__(self):
        super().__init__()
        self.contexts = []
        self.maps.clear()  # Remove default empty map
        self._graph = None

    def insert_contexts(self, *contexts):
        """Insert one or more contexts in reversed order the chained map.
        (A rule in last context will take precedence)

        To facilitate the identification of the context with the matching rule,
        the *relation_to_context* dictionary of the context is used.
        """

        self.contexts = list(reversed(contexts)) + self.contexts
        self.maps = [ctx.relation_to_context for ctx in reversed(contexts)] + self.maps
        self._graph = None

    def remove_contexts(self, n: int = None):
        """Remove the last n inserted contexts from the chain.

        Parameters
        ----------
        n: int
            (Default value = None)
        """

        del self.contexts[:n]
        del self.maps[:n]
        self._graph = None

    @property
    def defaults(self):
        for ctx in self.values():
            return ctx.defaults
        return {}

    @property
    def graph(self):
        """The graph relating"""
        if self._graph is None:
            self._graph = defaultdict(set)
            for fr_, to_ in self:
                self._graph[fr_].add(to_)
        return self._graph

    def transform(self, src, dst, registry, value):
        """Transform the value, finding the rule in the chained context.
        (A rule in last context will take precedence)
        """
        return self[(src, dst)].transform(src, dst, registry, value)

    def hashable(self):
        """Generate a unique hashable and comparable representation of self, which can
        be used as a key in a dict. This class cannot define ``__hash__`` because it is
        mutable, and the Python interpreter does cache the output of ``__hash__``.
        """
        return tuple(ctx.hashable() for ctx in self.contexts)
