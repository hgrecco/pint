import pint
import flexparser as fp
from dataclasses import dataclass
import typing as ty
from pint.facets.plain import definitions

ureg = pint.UnitRegistry(r"E://unit_short.txt", parser = "qudt")
