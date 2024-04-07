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


class BeginQuantitykind(block.BeginDirectiveBlock):
    _object_type = "quantitykind"


@dataclass(frozen=True)
class QuantitykindDefinitionBlock(fp.Block[BeginQuantitykind, common.Attribute, block.EndDirectiveBlock, ParserConfig]):
    
    qudt: dict[str, str] = field(default_factory=dict)

    def parse_attribute(self) -> None:
        for attr in self.body:
            if isinstance(attr, common.Attribute) and attr.parent == "qudt":
                self.qudt[attr.attr] = attr.value

    def derive_definition(self) -> definitions.UnitDefinition:
        self.parse_attribute()
        qk_name = self.opening.name.replace("-", "_")
        

        definition = definitions.QuantitykindDefinition(
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
