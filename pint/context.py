# -*- coding: utf-8 -*-
"""
    pint.context
    ~~~~~~~~~~~~

    Functions and classes related to context definitions and application.

    :copyright: 2013 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import division, unicode_literals, print_function, absolute_import


import re
from collections import defaultdict
import weakref
from pint.compat import ChainMap
from pint.util import ParserHelper, string_types

#: Regex to match the header parts of a context.
_header_re = re.compile('@context\s*(?P<defaults>\(.*\))?\s+(?P<name>\w+)\s*(=(?P<aliases>.*))*')

#: Reqex to match the different parts of a relation definition.
_def_re = re.compile('\s*(\w+)\s*=\s*([\w\d+-/*()]+)\s*')

#: Regex to match variable names in an equation.
_varname_re = re.compile('[A-Za-z_][A-Za-z0-9_]*')


def _freeze(d):
    """Return a hashable view of dict.
    """
    if isinstance(d, string_types):
        d = ParserHelper.from_string(d)
    if isinstance(d, frozenset):
        return d
    return frozenset(d.items())


def _expression_to_function(eq):
    def func(ureg, value, **kwargs):
        return ureg.parse_expression(eq, value=value, **kwargs)
    return func


class Context(object):
    """A specialized container that defines transformation functions from
    one dimension to another. Each Dimension are specified using a UnitsContainer.
    Simple transformation are given with a function taking a single parameter.

        >>> timedim = UnitsContainer({'[time]': 1})
        >>> spacedim = UnitsContainer({'[length]': 1})
        >>> def f(time):
        ...     'Time to length converter'
        ...     return 3. * time
        >>> c = Context()
        >>> c.add_transformation(timedim, spacedim, f)
        >>> c.transform(timedim, spacedim, 2)
        6

    Conversion functions may take optional keyword arguments and the context can
    have default values for these arguments.

        >>> def f(time, n):
        ...     'Time to length converter, n is the index of refraction of the material'
        ...     return 3. * time / n
        >>> c = Context(n=3)
        >>> c.add_transformation(timedim, spacedim, f)
        >>> c.transform(timedim, spacedim, 2)
        2

    """

    def __init__(self, name, aliases=(), defaults=None):

        self.name = name
        self.aliases = aliases

        #: Maps (src, dst) -> transformation function
        self.funcs = {}

        #: Maps defaults variable names to values
        self.defaults = defaults or {}

        #: Maps (src, dst) -> self
        #: Used as a convenience dictionary to be composed by ContextChain
        self.relation_to_context = weakref.WeakValueDictionary()

    @classmethod
    def from_context(cls, context, **defaults):
        """Creates a new context that shares the funcs dictionary with the original
        context. The default values are copied from the original context and updated
        with the new defaults.

        If defaults is empty, return the same context.
        """
        if defaults:
            newdef = dict(context.defaults, **defaults)
            c = cls(context.name, context.aliases, newdef)
            c.funcs = context.funcs
            for edge in context.funcs.keys():
                c.relation_to_context[edge] = c
            return c
        return context

    @classmethod
    def from_lines(cls, lines, to_base_func=None):
        header, lines = lines[0], lines[1:]

        r = _header_re.search(header)
        name = r.groupdict()['name'].strip()
        aliases = r.groupdict()['aliases']
        if aliases:
            aliases = tuple(a.strip() for a in r.groupdict()['aliases'].split('='))
        else:
            aliases = ()
        defaults = r.groupdict()['defaults']

        if defaults:
            def to_num(val):
                val = complex(val)
                if not val.imag:
                    return val.real
                return val
            defaults = {k: to_num(v) for k, v in _def_re.findall(defaults.strip('()'))}
            ctx = cls(name, aliases, defaults)
        else:
            ctx = cls(name, aliases)

        names = set()
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            rel, eq = line.split(':')
            names.update(_varname_re.findall(eq))

            func = _expression_to_function(eq)

            if '<->' in rel:
                src, dst = (ParserHelper.from_string(s) for s in rel.split('<->'))
                if to_base_func:
                    src = to_base_func(src)
                    dst = to_base_func(dst)
                ctx.add_transformation(src, dst, func)
                ctx.add_transformation(dst, src, func)
            elif '->' in rel:
                src, dst = (ParserHelper.from_string(s) for s in rel.split('->'))
                if to_base_func:
                    src = to_base_func(src)
                    dst = to_base_func(dst)
                ctx.add_transformation(src, dst, func)
            else:
                raise ValueError('Relationships must be specified with <-> or ->.')

        if defaults:
            missing_pars = set(defaults.keys()).difference(set(names))
            if missing_pars:
                raise ValueError('Context parameters {} not found in any equation.'.format(missing_pars))

        return ctx

    def add_transformation(self, src, dst, func):
        """Add a transformation function to the context.
        """
        _key = self.__keytransform__(src, dst)
        self.funcs[_key] = func
        self.relation_to_context[_key] = self

    def remove_transformation(self, src, dst):
        """Add a transformation function to the context.
        """
        _key = self.__keytransform__(src, dst)
        del self.funcs[_key]
        del self.relation_to_context[_key]

    @staticmethod
    def __keytransform__(src, dst):
        return _freeze(src), _freeze(dst)

    def transform(self, src, dst, registry, value):
        """Transform a value.
        """
        _key = self.__keytransform__(src, dst)
        return self.funcs[_key](registry, value, **self.defaults)


class ContextChain(ChainMap):
    """A specialized ChainMap for contexts that simplifies finding rules
    to transform from one dimension to another.
    """

    def __init__(self, *args, **kwargs):
        super(ContextChain, self).__init__(*args, **kwargs)
        self._graph = None
        self._contexts = []

    def insert_contexts(self, *contexts):
        """Insert one or more contexts in reversed order the chained map.
        (A rule in last context will take precedence)

        To facilitate the identification of the context with the matching rule,
        the *relation_to_context* dictionary of the context is used.
        """
        self._contexts.insert(0, contexts)
        self.maps = [ctx.relation_to_context for ctx in reversed(contexts)] + self.maps
        self._graph = None

    def remove_contexts(self, n):
        """Remove the last n inserted contexts from the chain.
        """
        self._contexts = self._contexts[n:]
        self.maps = self.maps[n:]
        self._graph = None

    @property
    def defaults(self):
        if self:
            return list(self.maps[0].values())[0].defaults
        return {}

    @property
    def graph(self):
        """The graph relating
        """
        if self._graph is None:
            self._graph = defaultdict(set)
            for fr_, to_ in self:
                self._graph[fr_].add(to_)
        return self._graph

    def transform(self, src, dst, registry, value):
        """Transform the value, finding the rule in the chained context.
        (A rule in last context will take precedence)

        :raises: KeyError if the rule is not found.
        """
        return self[(src, dst)].transform(src, dst, registry, value)
