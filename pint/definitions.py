"""
    pint.definitions
    ~~~~~~~~~~~~~~~~

    Functions and classes related to unit definitions.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from .converters import OffsetConverter, ScaleConverter
from .errors import DefinitionSyntaxError
from .util import ParserHelper, UnitsContainer, _is_dim


class _NotNumeric(Exception):
    def __init__(self, value):
        self.value = value


def numeric_parse(s):
    ph = ParserHelper.from_string(s)

    if len(ph):
        raise _NotNumeric(s)

    return ph.scale


class Definition:
    """Base class for definitions.

    Parameters
    ----------
    name : str
        Canonical name of the unit/prefix/etc.
    symbol : str or None
        A short name or symbol for the definition.
    aliases : iterable of str
        Other names for the unit/prefix/etc.
    converter : callable
        an instance of Converter.
    """

    def __init__(self, name, symbol, aliases, converter):
        self._name = name
        self._symbol = symbol
        self._aliases = aliases
        self._converter = converter

    @property
    def is_multiplicative(self):
        return self._converter.is_multiplicative

    @classmethod
    def from_string(cls, definition):
        """Parse a definition

        Parameters
        ----------
        definition :


        Returns
        -------

        """
        name, definition = definition.split("=", 1)
        name = name.strip()

        result = [res.strip() for res in definition.split("=")]

        # @alias name = alias1 = alias2 = ...
        if name.startswith("@alias "):
            name = name[len("@alias ") :].lstrip()
            return AliasDefinition(name, tuple(result))

        value, aliases = result[0], tuple([x for x in result[1:] if x != ""])
        symbol, aliases = (aliases[0], aliases[1:]) if aliases else (None, aliases)
        if symbol == "_":
            symbol = None
        aliases = tuple([x for x in aliases if x != "_"])

        if name.startswith("["):
            return DimensionDefinition(name, symbol, aliases, value)
        elif name.endswith("-"):
            name = name.rstrip("-")
            return PrefixDefinition(name, symbol, aliases, value)
        else:
            return UnitDefinition(name, symbol, aliases, value)

    @property
    def name(self):
        return self._name

    @property
    def symbol(self):
        return self._symbol or self._name

    @property
    def has_symbol(self):
        return bool(self._symbol)

    @property
    def aliases(self):
        return self._aliases

    def add_aliases(self, *alias):
        alias = tuple(a for a in alias if a not in self._aliases)
        self._aliases = self._aliases + alias

    @property
    def converter(self):
        return self._converter

    def __str__(self):
        return self.name


class PrefixDefinition(Definition):
    """Definition of a prefix."""

    def __init__(self, name, symbol, aliases, converter):
        if isinstance(converter, str):
            try:
                converter = ScaleConverter(numeric_parse(converter))
            except _NotNumeric as ex:
                raise ValueError(
                    f"Prefix definition ('{name}') must contain only numbers, not {ex.value}"
                )

        aliases = tuple(alias.strip("-") for alias in aliases)
        if symbol:
            symbol = symbol.strip("-")
        super().__init__(name, symbol, aliases, converter)


class UnitDefinition(Definition):
    """Definition of a unit.

    Parameters
    ----------
    reference : UnitsContainer
        Reference units.
    is_base : bool
        Indicates if it is a base unit.

    """

    def __init__(self, name, symbol, aliases, converter, reference=None, is_base=False):
        self.reference = reference
        self.is_base = is_base
        if isinstance(converter, str):
            if ";" in converter:
                [converter, modifiers] = converter.split(";", 2)

                try:
                    modifiers = dict(
                        (key.strip(), numeric_parse(value))
                        for key, value in (
                            part.split(":") for part in modifiers.split(";")
                        )
                    )
                except _NotNumeric as ex:
                    raise ValueError(
                        f"Unit definition ('{name}') must contain only numbers in modifier, not {ex.value}"
                    )

            else:
                modifiers = {}

            converter = ParserHelper.from_string(converter)
            if not any(_is_dim(key) for key in converter.keys()):
                self.is_base = False
            elif all(_is_dim(key) for key in converter.keys()):
                self.is_base = True
            else:
                raise DefinitionSyntaxError(
                    "Cannot mix dimensions and units in the same definition. "
                    "Base units must be referenced only to dimensions. "
                    "Derived units must be referenced only to units."
                )
            self.reference = UnitsContainer(converter)
            if modifiers.get("offset", 0.0) != 0.0:
                converter = OffsetConverter(converter.scale, modifiers["offset"])
            else:
                converter = ScaleConverter(converter.scale)

        super().__init__(name, symbol, aliases, converter)


class DimensionDefinition(Definition):
    """Definition of a dimension."""

    def __init__(self, name, symbol, aliases, converter, reference=None, is_base=False):
        self.reference = reference
        self.is_base = is_base
        if isinstance(converter, str):
            converter = ParserHelper.from_string(converter)
            if not converter:
                self.is_base = True
            elif all(_is_dim(key) for key in converter.keys()):
                self.is_base = False
            else:
                raise DefinitionSyntaxError(
                    "Base dimensions must be referenced to None. "
                    "Derived dimensions must only be referenced "
                    "to dimensions."
                )
            self.reference = UnitsContainer(converter)

        super().__init__(name, symbol, aliases, converter=None)


class AliasDefinition(Definition):
    """Additional alias(es) for an already existing unit"""

    def __init__(self, name, aliases):
        super().__init__(name=name, symbol=None, aliases=aliases, converter=None)
