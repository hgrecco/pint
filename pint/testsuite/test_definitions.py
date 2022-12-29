import pytest

from pint.definitions import Definition
from pint.errors import DefinitionSyntaxError
from pint.facets.nonmultiplicative.definitions import (
    LogarithmicConverter,
    OffsetConverter,
)
from pint.facets.plain import (
    AliasDefinition,
    DimensionDefinition,
    PrefixDefinition,
    ScaleConverter,
    UnitDefinition,
)
from pint.util import UnitsContainer


class TestDefinition:
    def test_invalid(self):
        with pytest.raises(DefinitionSyntaxError):
            Definition.from_string("x = [time] * meter")
        with pytest.raises(DefinitionSyntaxError):
            Definition.from_string("[x] = [time] * meter")

    def test_prefix_definition(self):

        with pytest.raises(ValueError):
            Definition.from_string("m- = 1e-3 k")

        for definition in ("m- = 1e-3", "m- = 10**-3", "m- = 0.001"):
            x = Definition.from_string(definition)
            assert isinstance(x, PrefixDefinition)
            assert x.name == "m"
            assert x.aliases == ()
            assert x.converter.to_reference(1000) == 1
            assert x.converter.from_reference(0.001) == 1

        x = Definition.from_string("kilo- = 1e-3 = k-")
        assert isinstance(x, PrefixDefinition)
        assert x.name == "kilo"
        assert x.aliases == ()
        assert x.symbol == "k"
        assert x.converter.to_reference(1000) == 1
        assert x.converter.from_reference(0.001) == 1

        x = Definition.from_string("kilo- = 1e-3 = k- = anotherk-")
        assert isinstance(x, PrefixDefinition)
        assert x.name == "kilo"
        assert x.aliases == ("anotherk",)
        assert x.symbol == "k"
        assert x.converter.to_reference(1000) == 1
        assert x.converter.from_reference(0.001) == 1

    def test_baseunit_definition(self):
        x = Definition.from_string("meter = [length]")
        assert isinstance(x, UnitDefinition)
        assert x.is_base
        assert x.reference == UnitsContainer({"[length]": 1})

    def test_unit_definition(self):
        x = Definition.from_string("coulomb = ampere * second")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, ScaleConverter)
        assert x.converter.scale == 1
        assert x.reference == UnitsContainer(ampere=1, second=1)

        x = Definition.from_string("faraday =  96485.3399 * coulomb")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, ScaleConverter)
        assert x.converter.scale == 96485.3399
        assert x.reference == UnitsContainer(coulomb=1)

        x = Definition.from_string("degF = 9 / 5 * kelvin; offset: 255.372222")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, OffsetConverter)
        assert x.converter.scale == 9 / 5
        assert x.converter.offset == 255.372222
        assert x.reference == UnitsContainer(kelvin=1)

        x = Definition.from_string(
            "turn = 6.28 * radian = _ = revolution = = cycle = _"
        )
        assert isinstance(x, UnitDefinition)
        assert x.name == "turn"
        assert x.aliases == ("revolution", "cycle")
        assert x.symbol == "turn"
        assert not x.is_base
        assert isinstance(x.converter, ScaleConverter)
        assert x.converter.scale == 6.28
        assert x.reference == UnitsContainer(radian=1)

        with pytest.raises(ValueError):
            Definition.from_string(
                "degF = 9 / 5 * kelvin; offset: 255.372222 bla",
            )

    def test_log_unit_definition(self):

        x = Definition.from_string(
            "decibelmilliwatt = 1e-3 watt; logbase: 10; logfactor: 10 = dBm"
        )
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1e-3
        assert x.converter.logbase == 10
        assert x.converter.logfactor == 10
        assert x.reference == UnitsContainer(watt=1)

        x = Definition.from_string("decibel = 1 ; logbase: 10; logfactor: 10 = dB")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1
        assert x.converter.logbase == 10
        assert x.converter.logfactor == 10
        assert x.reference == UnitsContainer()

        x = Definition.from_string("bell = 1 ; logbase: 10; logfactor: 1 = B")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1
        assert x.converter.logbase == 10
        assert x.converter.logfactor == 1
        assert x.reference == UnitsContainer()

        x = Definition.from_string("decade = 1 ; logbase: 10; logfactor: 1")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1
        assert x.converter.logbase == 10
        assert x.converter.logfactor == 1
        assert x.reference == UnitsContainer()

        eulersnumber = 2.71828182845904523536028747135266249775724709369995
        x = Definition.from_string(
            "neper = 1 ; logbase: %1.50f; logfactor: 0.5 = Np" % eulersnumber
        )
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1
        assert x.converter.logbase == eulersnumber
        assert x.converter.logfactor == 0.5
        assert x.reference == UnitsContainer()

        x = Definition.from_string("octave = 1 ; logbase: 2; logfactor: 1 = oct")
        assert isinstance(x, UnitDefinition)
        assert not x.is_base
        assert isinstance(x.converter, LogarithmicConverter)
        assert x.converter.scale == 1
        assert x.converter.logbase == 2
        assert x.converter.logfactor == 1
        assert x.reference == UnitsContainer()

    def test_dimension_definition(self):
        x = DimensionDefinition("[time]")
        assert x.is_base
        assert x.name == "[time]"

        x = Definition.from_string("[speed] = [length]/[time]")
        assert isinstance(x, DimensionDefinition)
        assert x.reference == UnitsContainer({"[length]": 1, "[time]": -1})

    def test_alias_definition(self):
        x = Definition.from_string("@alias meter = metro = metr")
        assert isinstance(x, AliasDefinition)
        assert x.name == "meter"
        assert x.aliases == ("metro", "metr")
