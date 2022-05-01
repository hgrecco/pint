"""
    pint.facets.context.definitions
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    :copyright: 2022 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from ...definitions import Definition
from ...errors import DefinitionSyntaxError
from ...util import ParserHelper, SourceIterator
from ..plain.definitions import UnitDefinition

if TYPE_CHECKING:
    from ..plain.quantity import Quantity

_header_re = re.compile(
    r"@context\s*(?P<defaults>\(.*\))?\s+(?P<name>\w+)\s*(=(?P<aliases>.*))*"
)
_varname_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# TODO: Put back annotation when possible
# registry_cache: "UnitRegistry"


class Expression:
    def __init__(self, eq):
        self._eq = eq

    def __call__(self, ureg, value: Any, **kwargs: Any):
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
            raise DefinitionSyntaxError("Can't define plain units within a context")
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
