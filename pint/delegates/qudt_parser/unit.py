"""
parser for unit file
"""


from __future__ import annotations

from dataclasses import dataclass, field

import flexparser as fp
import typing as ty

from ...converters import Converter
from ...facets.plain import definitions
from ...util import UnitsContainer
from ..base_defparser import ParserConfig, PintParsedStatement
from . import common, block
import re


class BeginUnit(block.BeginDirectiveBlock):
    _object_type = "unit"


def make_units_container(dimensionvector: str) -> UnitsContainer:
    # use regex to split the dimension vector 'A0E1L0I0M0H0T0D0' into a dictionary of units
    if ":" in dimensionvector:
        dimensionvector = dimensionvector.split(":")[1]
    dimensionvector = re.sub(r'[a-zA-Z]', r' ', dimensionvector).split()
    units = {
        unit: int(exponent) for unit, exponent in zip(common.BASE_UNITS, dimensionvector)
    }
    return UnitsContainer(units)


@dataclass(frozen=True)
class UnitDefinitionBlock(fp.Block[BeginUnit, common.Attribute, block.EndDirectiveBlock, ParserConfig]):
    
    qudt: dict[str, str] = field(default_factory=dict)

    def parse_attribute(self) -> None:
        self.qudt['symbol'] = None
        self.qudt['conversionMultiplier'] = 1.
        for attr in self.body:
            if isinstance(attr, common.Attribute) and attr.parent == "qudt":
                self.qudt[attr.attr] = attr.value
                if attr.attr == "symbol":
                    self.qudt['symbol'] = attr.value.strip('\"')

    def derive_definition(self) -> definitions.UnitDefinition:
        self.parse_attribute()
        unit_name = self.opening.name.replace("-", "_")#.lower()
        if self.qudt['symbol'] in common.BASE_UNITS:
            dim = common.DIMENSIONS[common.BASE_UNITS.index(self.qudt['symbol'])]
            reference = UnitsContainer({"["+dim+"]": 1})
            self.qudt['conversionMultiplier'] = 1
        else:
            reference = make_units_container(self.qudt['hasDimensionVector'])

        converter = Converter.from_arguments(scale=self.qudt['conversionMultiplier'])
        definition = definitions.UnitDefinition(
                unit_name,
                self.qudt['symbol'],
                tuple(),
                converter,
                reference,
            )
        return definition


@dataclass(frozen=True)
class EntryBlock(fp.RootBlock[ty.Union[
                 common.HeaderBlock,
                 UnitDefinitionBlock
                 ], ParserConfig]):
    pass
