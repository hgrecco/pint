"""
parser for unit file
"""


from __future__ import annotations

from dataclasses import dataclass, field

import flexparser as fp

from ...converters import Converter
from ...facets.plain import definitions
from ...util import UnitsContainer
from ..base_defparser import ParserConfig
from . import block, common


class BeginUnit(block.BeginDirectiveBlock):
    _object_type = "unit"


@dataclass(frozen=True)
class UnitDefinitionBlock(
    fp.Block[BeginUnit, common.Attribute, block.EndDirectiveBlock, ParserConfig]
):
    qudt: dict[str, str] = field(default_factory=dict)

    def parse_attribute(self) -> None:
        self.qudt["symbol"] = None
        self.qudt["conversionMultiplier"] = 1.0
        for attr in self.body:
            if isinstance(attr, common.Attribute) and attr.parent == "qudt":
                self.qudt[attr.attr] = attr.value
                if attr.attr == "symbol":
                    self.qudt["symbol"] = attr.value.strip('"')

    def derive_definition(self) -> definitions.UnitDefinition:
        self.parse_attribute()
        unit_name = self.opening.name.replace("-", "_")  # .lower()
        if self.qudt["symbol"] in common.BASE_UNITS:
            dim = common.DIMENSIONS[common.BASE_UNITS.index(self.qudt["symbol"])]
            reference = UnitsContainer({"[" + dim + "]": 1})
            self.qudt["conversionMultiplier"] = 1
        else:
            reference = common.make_units_container(self.qudt["hasDimensionVector"])

        converter = Converter.from_arguments(scale=self.qudt["conversionMultiplier"])
        definition = definitions.UnitDefinition(
            unit_name,
            self.qudt["symbol"],
            tuple(),
            converter,
            reference,
        )
        return definition
