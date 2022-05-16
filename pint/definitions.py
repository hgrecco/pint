"""
    pint.definitions
    ~~~~~~~~~~~~~~~~

    Functions and classes related to unit definitions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

from .converters import Converter


@dataclass(frozen=True)
class PreprocessedDefinition:
    """Splits a definition into the constitutive parts.

    A definition is given as a string with equalities in a single line::

        ---------------> rhs
        a = b = c = d = e
        |   |   |   -------> aliases (optional)
        |   |   |
        |   |   -----------> symbol (use "_" for no symbol)
        |   |
        |   ---------------> value
        |
        -------------------> name
    """

    name: str
    symbol: Optional[str]
    aliases: Tuple[str, ...]
    value: str
    rhs_parts: Tuple[str, ...]

    @classmethod
    def from_string(cls, definition: str) -> PreprocessedDefinition:
        name, definition = definition.split("=", 1)
        name = name.strip()

        rhs_parts = tuple(res.strip() for res in definition.split("="))

        value, aliases = rhs_parts[0], tuple([x for x in rhs_parts[1:] if x != ""])
        symbol, aliases = (aliases[0], aliases[1:]) if aliases else (None, aliases)
        if symbol == "_":
            symbol = None
        aliases = tuple([x for x in aliases if x != "_"])

        return cls(name, symbol, aliases, value, rhs_parts)


@dataclass(frozen=True)
class Definition:
    """Base class for definitions.

    Parameters
    ----------
    name : str
        Canonical name of the unit/prefix/etc.
    defined_symbol : str or None
        A short name or symbol for the definition.
    aliases : iterable of str
        Other names for the unit/prefix/etc.
    converter : callable or Converter or None
    """

    name: str
    defined_symbol: Optional[str]
    aliases: Tuple[str, ...]
    converter: Optional[Union[Callable, Converter]]

    _subclasses = []
    _default_subclass = None

    def __init_subclass__(cls, **kwargs):
        if kwargs.pop("default", False):
            if cls._default_subclass is not None:
                raise ValueError("There is already a registered default definition.")
            Definition._default_subclass = cls
        super().__init_subclass__(**kwargs)
        cls._subclasses.append(cls)

    def __post_init__(self):
        if isinstance(self.converter, str):
            raise TypeError(
                "The converter parameter cannot be an instance of `str`. Use `from_string` method"
            )

    @property
    def is_multiplicative(self) -> bool:
        return self.converter.is_multiplicative

    @property
    def is_logarithmic(self) -> bool:
        return self.converter.is_logarithmic

    @classmethod
    def accept_to_parse(cls, preprocessed: PreprocessedDefinition):
        return False

    @classmethod
    def from_string(
        cls, definition: Union[str, PreprocessedDefinition], non_int_type: type = float
    ) -> Definition:
        """Parse a definition.

        Parameters
        ----------
        definition : str or PreprocessedDefinition
        non_int_type : type

        Returns
        -------
        Definition or subclass of Definition
        """

        if isinstance(definition, str):
            definition = PreprocessedDefinition.from_string(definition)

        for subclass in cls._subclasses:
            if subclass.accept_to_parse(definition):
                return subclass.from_string(definition, non_int_type)

        if cls._default_subclass is None:
            raise ValueError("No matching definition (and no default parser).")

        return cls._default_subclass.from_string(definition, non_int_type)

    @property
    def symbol(self) -> str:
        return self.defined_symbol or self.name

    @property
    def has_symbol(self) -> bool:
        return bool(self.defined_symbol)

    def add_aliases(self, *alias: str) -> None:
        raise Exception("Cannot add aliases, definitions are inmutable.")

    def __str__(self) -> str:
        return self.name
