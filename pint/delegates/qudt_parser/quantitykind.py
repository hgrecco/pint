"""
parser for unit file
"""


from __future__ import annotations

from dataclasses import dataclass, field

import flexparser as fp

from ...facets.plain import definitions
from ..base_defparser import ParserConfig
from . import block, common


class BeginQuantitykind(block.BeginDirectiveBlock):
    _object_type = "quantitykind"


@dataclass(frozen=True)
class QuantitykindDefinitionBlock(
    fp.Block[BeginQuantitykind, common.Attribute, block.EndDirectiveBlock, ParserConfig]
):
    qudt: dict[str, str] = field(default_factory=dict)

    def parse_attribute(self) -> None:
        self.qudt["symbol"] = None
        for attr in self.body:
            if isinstance(attr, common.Attribute) and attr.parent == "qudt":
                self.qudt[attr.attr] = attr.value
                if attr.attr == "symbol":
                    self.qudt["symbol"] = attr.value.strip('"')

    def derive_definition(self) -> definitions.QuantitykindDefinition:
        self.parse_attribute()
        qk_name = self.opening.name.replace("-", "_")
        reference = common.make_units_container(self.qudt["hasDimensionVector"])
        definition = definitions.QuantitykindDefinition(
            qk_name,
            self.qudt["symbol"],
            reference,
            metadata=self.qudt,
        )
        return definition
