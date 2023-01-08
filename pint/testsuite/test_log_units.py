import logging
import math
import operator as op

import pytest

from pint import (
    DimensionalityError,
    LogarithmicUnitCalculusError,
    OffsetUnitCalculusError,
    Unit,
    UnitRegistry,
)
from pint.facets.plain.unit import UnitsContainer
from pint.testsuite import QuantityTestCase, helpers


@pytest.fixture(scope="module")
def module_registry_auto_offset():
    return UnitRegistry(autoconvert_offset_to_baseunit=True)


# TODO: do not subclass from QuantityTestCase
class TestLogarithmicQuantity(QuantityTestCase):
    def test_log_quantity_creation(self, caplog):

        # Following Quantity Creation Pattern
        for args in (
            (4.2, "dBm"),
            (4.2, UnitsContainer(decibelmilliwatt=1)),
            (4.2, self.ureg.dBm),
        ):
            x = self.Q_(*args)
            assert x.magnitude == 4.2
            assert x.units == UnitsContainer(decibelmilliwatt=1)

        x = self.Q_(self.Q_(4.2, "dBm"))
        assert x.magnitude == 4.2
        assert x.units == UnitsContainer(decibelmilliwatt=1)

        x = self.Q_(4.2, UnitsContainer(decibelmilliwatt=1))
        y = self.Q_(x)
        assert x.magnitude == y.magnitude
        assert x.units == y.units
        assert x is not y

        # Using multiplications for dB units requires autoconversion to baseunits
        new_reg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        x = new_reg.Quantity("4.2 * dBm")
        assert x.magnitude == 4.2
        assert x.units == UnitsContainer(decibelmilliwatt=1)

        with caplog.at_level(logging.DEBUG):
            assert "wally" not in caplog.text
            assert 4.2 * new_reg.dBm == new_reg.Quantity(4.2, 2 * new_reg.dBm)

        assert len(caplog.records) == 1

    def test_delta_log_quantity_creation(self, log_module_registry):
        #  Following Quantity Creation Pattern for "delta_" units:
        # tests the quantity creation of an absolute decibel unit: decibelmilliwatt.
        for args in (
            (4.2, "delta_dBm"),
            (4.2, UnitsContainer(delta_decibelmilliwatt=1)),
            (4.2, log_module_registry.delta_dBm),
        ):
            x = log_module_registry.Quantity(*args)
            assert x.magnitude == 4.2
            assert x.units == UnitsContainer(delta_decibelmilliwatt=1)
        # tests the quantity creation of an absolute decibel unit: decibelmilliwatt.
        for args in (
            (4.2, "delta_dB"),
            (4.2, UnitsContainer(delta_decibel=1)),
            (4.2, log_module_registry.delta_dB),
        ):
            x = log_module_registry.Quantity(*args)
            assert x.magnitude == 4.2
            assert x.units == UnitsContainer(delta_decibel=1)

    def test_log_convert(self):
        # # 1 dB = 1/10 * bel
        # helpers.assert_quantity_almost_equal(self.Q_(1.0, "dB").to("dimensionless"), self.Q_(1, "bell") / 10)
        # # Uncomment Bell unit in default_en.txt

        # ## Test dB to dB units octave - decade
        # 1 decade = log2(10) octave
        helpers.assert_quantity_almost_equal(
            self.Q_(1.0, "decade"), self.Q_(math.log(10, 2), "octave")
        )
        # ## Test dB to dB units dBm - dBu
        # 0 dBm = 1mW = 1e3 uW = 30 dBu
        helpers.assert_quantity_almost_equal(
            self.Q_(0.0, "dBm"), self.Q_(29.999999999999996, "dBu"), atol=1e-7
        )
        # ## Test dB to dB units dBm - dBW
        # 0 dBW = 1W = 1e3 mW = 30 dBm
        helpers.assert_quantity_almost_equal(
            self.Q_(0.0, "dBW"), self.Q_(29.999999999999996, "dBm"), atol=1e-7
        )

        # ## Test dB to dB units dBm - dBW
        # 0 dBW = 1W = 1e3 mW = 30 dBm
        helpers.assert_quantity_almost_equal(
            self.Q_(0.0, "dBW"), self.Q_(29.999999999999996, "dBm"), atol=1e-7
        )

    def test_mix_regular_log_units(self):
        # Test regular-logarithmic mixed definition, such as dB/km or dB/cm

        # Multiplications and divisions with a mix of Logarithmic Units and regular Units is normally not possible.
        # The reason is that dB are considered by pint like offset units.
        # Multiplications and divisions that involve offset units are badly defined, so pint raises an error
        with pytest.raises(OffsetUnitCalculusError):
            (-10.0 * self.ureg.dB) / (1 * self.module_registry.cm)

        # However, if the flag autoconvert_offset_to_baseunit=True is given to UnitRegistry, then pint converts the unit to plain.
        # With this flag on multiplications and divisions are now possible:
        new_reg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        helpers.assert_quantity_almost_equal(
            -10 * new_reg.dB / new_reg.cm, 0.1 / new_reg.cm
        )


log_unit_names = [
    "decibelwatt",
    "dBW",
    "decibelmilliwatt",
    "dBm",
    "decibelmicrowatt",
    "dBu",
    "decibel",
    "dB",
    "decade",
    "octave",
    "oct",
]


@pytest.mark.parametrize("unit_name", log_unit_names)
def test_unit_by_attribute(module_registry, unit_name):
    """Can the logarithmic units be accessed by attribute lookups?"""
    unit = getattr(module_registry, unit_name)
    assert isinstance(unit, Unit)


@pytest.mark.parametrize("unit_name", log_unit_names)
def test_unit_parsing(module_registry, unit_name):
    """Can the logarithmic units be understood by the parser?"""
    unit = module_registry.parse_units(unit_name)
    assert isinstance(unit, Unit)


@pytest.mark.parametrize("mag", [1.0, 4.2])
@pytest.mark.parametrize("unit_name", log_unit_names)
def test_quantity_by_constructor(module_registry, unit_name, mag):
    """Can Quantity() objects be constructed using logarithmic units?"""
    q = module_registry.Quantity(mag, unit_name)
    assert q.magnitude == pytest.approx(mag)
    assert q.units == getattr(module_registry, unit_name)


@pytest.mark.parametrize("mag", [1.0, 4.2])
@pytest.mark.parametrize("unit_name", log_unit_names)
def test_quantity_by_multiplication(module_registry_auto_offset, unit_name, mag):
    """Test that logarithmic units can be defined with multiplication

    Requires setting `autoconvert_offset_to_baseunit` to True
    """
    unit = getattr(module_registry_auto_offset, unit_name)
    q = mag * unit
    assert q.magnitude == pytest.approx(mag)
    assert q.units == unit


log_delta_unit_names = ["delta_" + name for name in log_unit_names if name != "decade"]


@pytest.mark.parametrize("unit_name", log_delta_unit_names)
def test_deltaunit_by_attribute(log_module_registry, unit_name):
    """Can the delta logarithmic units be accessed by attribute lookups?"""
    unit = getattr(log_module_registry, unit_name)
    assert isinstance(unit, Unit)


@pytest.mark.parametrize("unit_name", log_delta_unit_names)
def test_deltaunit_parsing(log_module_registry, unit_name):
    """Can the delta logarithmic units be understood by the parser?"""
    unit = getattr(log_module_registry, unit_name)
    assert isinstance(unit, Unit)


@pytest.mark.parametrize("mag", [1.0, 4.2])
@pytest.mark.parametrize("unit_name", log_delta_unit_names)
def test_delta_quantity_by_constructor(log_module_registry, unit_name, mag):
    """Can Quantity() objects be constructed using delta logarithmic units?"""
    q = log_module_registry.Quantity(mag, unit_name)
    assert q.magnitude == pytest.approx(mag)
    assert q.units == getattr(log_module_registry, unit_name)


@pytest.mark.parametrize("mag", [1.0, 4.2])
@pytest.mark.parametrize("unit_name", log_delta_unit_names)
def test_delta_quantity_by_multiplication(log_module_registry, unit_name, mag):
    """Test that delta logarithmic units can be defined with multiplication

    Requires setting `autoconvert_offset_to_baseunit` to True
    """
    unit = getattr(log_module_registry, unit_name)
    q = mag * unit
    assert q.magnitude == pytest.approx(mag)
    assert q.units == unit


@pytest.mark.parametrize(
    "unit1,unit2",
    [
        ("decibelwatt", "dBW"),
        ("decibelmilliwatt", "dBm"),
        ("decibelmicrowatt", "dBu"),
        ("decibel", "dB"),
        ("octave", "oct"),
    ],
)
def test_unit_equivalence(module_registry, unit1, unit2):
    """Are certain pairs of units equivalent?"""
    assert getattr(module_registry, unit1) == getattr(module_registry, unit2)


@pytest.mark.parametrize(
    "db_value,scalar",
    [
        (0.0, 1.0),  # 0 dB == 1x
        (-10.0, 0.1),  # -10 dB == 0.1x
        (10.0, 10.0),
        (30.0, 1e3),
        (60.0, 1e6),
    ],
)
def test_db_conversion(module_registry, db_value, scalar):
    """Test that a dB value can be converted to a scalar and back."""
    Q_ = module_registry.Quantity
    assert Q_(db_value, "dB").to("dimensionless").magnitude == pytest.approx(scalar)
    assert Q_(scalar, "dimensionless").to("dB").magnitude == pytest.approx(db_value)


@pytest.mark.parametrize(
    "octave,scalar",
    [
        (2.0, 4.0),  # 2 octave == 4x
        (1.0, 2.0),  # 1 octave == 2x
        (0.0, 1.0),
        (-1.0, 0.5),
        (-2.0, 0.25),
    ],
)
def test_octave_conversion(module_registry, octave, scalar):
    """Test that an octave can be converted to a scalar and back."""
    Q_ = module_registry.Quantity
    assert Q_(octave, "octave").to("dimensionless").magnitude == pytest.approx(scalar)
    assert Q_(scalar, "dimensionless").to("octave").magnitude == pytest.approx(octave)


@pytest.mark.parametrize(
    "decade,scalar",
    [
        (2.0, 100.0),  # 2 decades == 100x
        (1.0, 10.0),  # 1 octave == 2x
        (0.0, 1.0),
        (-1.0, 0.1),
        (-2.0, 0.01),
    ],
)
def test_decade_conversion(module_registry, decade, scalar):
    """Test that a decade can be converted to a scalar and back."""
    Q_ = module_registry.Quantity
    assert Q_(decade, "decade").to("dimensionless").magnitude == pytest.approx(scalar)
    assert Q_(scalar, "dimensionless").to("decade").magnitude == pytest.approx(decade)


@pytest.mark.parametrize(
    "dbm_value,mw_value",
    [
        (0.0, 1.0),  # 0.0 dBm == 1.0 mW
        (10.0, 10.0),
        (20.0, 100.0),
        (-10.0, 0.1),
        (-20.0, 0.01),
    ],
)
def test_dbm_mw_conversion(module_registry, dbm_value, mw_value):
    """Test dBm values can convert to mW and back."""
    Q_ = module_registry.Quantity
    assert Q_(dbm_value, "dBm").to("mW").magnitude == pytest.approx(mw_value)
    assert Q_(mw_value, "mW").to("dBm").magnitude == pytest.approx(dbm_value)


@pytest.mark.xfail
def test_compound_log_unit_multiply_definition(module_registry_auto_offset):
    """Check that compound log units can be defined using multiply."""
    Q_ = module_registry_auto_offset.Quantity
    canonical_def = Q_(-161, "dBm") / module_registry_auto_offset.Hz
    mult_def = -161 * module_registry_auto_offset("dBm/Hz")
    assert mult_def == canonical_def


@pytest.mark.xfail
def test_compound_log_unit_quantity_definition(module_registry_auto_offset):
    """Check that compound log units can be defined using ``Quantity()``."""
    Q_ = module_registry_auto_offset.Quantity
    canonical_def = Q_(-161, "dBm") / module_registry_auto_offset.Hz
    quantity_def = Q_(-161, "dBm/Hz")
    assert quantity_def == canonical_def


def test_compound_log_unit_parse_definition(module_registry_auto_offset):
    Q_ = module_registry_auto_offset.Quantity
    canonical_def = Q_(-161, "dBm") / module_registry_auto_offset.Hz
    parse_def = module_registry_auto_offset("-161 dBm/Hz")
    assert parse_def == canonical_def


def test_compound_log_unit_parse_expr(module_registry_auto_offset):
    """Check that compound log units can be defined using ``parse_expression()``."""
    Q_ = module_registry_auto_offset.Quantity
    canonical_def = Q_(-161, "dBm") / module_registry_auto_offset.Hz
    parse_def = module_registry_auto_offset.parse_expression("-161 dBm/Hz")
    assert canonical_def == parse_def


@pytest.mark.xfail
def test_dbm_db_addition(module_registry_auto_offset):
    """Test a dB value can be added to a dBm and the answer is correct."""
    power = (5 * module_registry_auto_offset.dBm) + (
        10 * module_registry_auto_offset.dB
    )
    assert power.to("dBm").magnitude == pytest.approx(15)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "freq1,octaves,freq2",
    [
        (100, 2.0, 400),
        (50, 1.0, 100),
        (200, 0.0, 200),
    ],  # noqa: E231
)
def test_frequency_octave_addition(module_registry_auto_offset, freq1, octaves, freq2):
    """Test an Octave can be added to a frequency correctly"""
    freq1 = freq1 * module_registry_auto_offset.Hz
    shift = octaves * module_registry_auto_offset.Octave
    new_freq = freq1 + shift
    assert new_freq.units == freq1.units
    assert new_freq.magnitude == pytest.approx(freq2)


def test_db_db_addition(log_module_registry):
    """Test a dB value can be added to a dB and the answer is correct."""
    # adding two dB units
    power = (5 * log_module_registry.dB) + (10 * log_module_registry.dB)
    assert power.magnitude == pytest.approx(11.19331048066)
    assert power.units == log_module_registry.dB

    # Adding two absolute dB units
    power = (5 * log_module_registry.dBW) + (10 * log_module_registry.dBW)
    assert power.magnitude == pytest.approx(11.19331048066)
    assert power.units == log_module_registry.dBW


class TestLogarithmicUnitMath(QuantityTestCase):
    @classmethod
    def setup_class(cls):
        cls.kwargs["autoconvert_offset_to_baseunit"] = True
        cls.kwargs["logarithmic_math"] = True
        super().setup_class()

    additions = [
        # --- input tuple --| -- expected result --| -- expected result (conversion to base units) --
        pytest.param(
            ((2, "dB"), (1, "decibel")),
            (4.5390189104386724, "decibel"),
            (4.5390189104386724, "decibel"),
            id="dB+dB",
        ),
        pytest.param(
            ((2, "dBW"), (1, "decibelwatt")),
            (4.5390189104386724, "decibelwatt"),
            (4.5390189104386724, "decibelwatt"),
            id="dBW+dBW",
        ),
        pytest.param(
            ((2, "delta_dBW"), (1, "delta_decibelwatt")),
            (3, "delta_decibelwatt"),
            (3, "delta_decibelwatt"),
            id="delta_dBW+delta_dBW",
        ),
        pytest.param(
            ((100, "dimensionless"), (2, "decibel")), "error", "error", id="'' + dB"
        ),
        pytest.param(
            ((2, "decibel"), (100, "dimensionless")), "error", "error", id="dB + ''"
        ),  # ensures symmetry
        pytest.param(
            ((100, "dimensionless"), (2, "dBW")), "error", "error", id="'' + dBW"
        ),
        pytest.param(
            ((2, "dBW"), (100, "dimensionless")), "error", "error", id="dBW + ''"
        ),
        pytest.param(((100, "watt"), (2, "dBW")), "error", "error", id="W + dBW"),
        pytest.param(((2, "dBW"), (100, "watt")), "error", "error", id="dBW + W"),
        pytest.param(
            ((2, "dBW"), (1, "decibel")), "error", "error", id="dBW+dB"
        ),  # dimensionality error
        pytest.param(
            ((2, "dB"), (1, "delta_decibel")),
            (3, "decibel"),
            (3, "decibel"),
            id="dB+delta_dB",
        ),
        pytest.param(
            ((2, "delta_dB"), (1, "decibel")),
            (3, "decibel"),
            (3, "decibel"),
            id="delta_dB+dB",
        ),
        pytest.param(
            ((2, "dBW"), (1, "delta_decibelwatt")),
            (3, "decibelwatt"),
            (3, "decibelwatt"),
            id="dBW+delta_dBW",
        ),
        pytest.param(
            ((2, "delta_dBW"), (10, "dimensionless")),
            "error",
            "error",
            id="delta_dBW + ''",
        ),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected", "expected_base_units"), additions
    )
    def test_addition(self, input_tuple, expected, expected_base_units):

        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        # update input tuple with new values to have correct values on failure
        input_tuple = q1, q2

        self.ureg.autoconvert_offset_to_baseunit = False
        if expected == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.add(q1, q2)
        else:
            expected = self.Q_(*expected)
            assert op.add(q1, q2).units == expected.units
            helpers.assert_quantity_almost_equal(op.add(q1, q2), expected, atol=0.01)

        self.ureg.autoconvert_offset_to_baseunit = True
        if expected_base_units == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.add(q1, q2)
        else:
            expected_base_units = self.Q_(*expected_base_units)
            assert op.add(q1, q2).units == expected_base_units.units
            helpers.assert_quantity_almost_equal(
                op.add(q1, q2), expected_base_units, atol=0.01
            )

    subtractions = [
        # --- input tuple -------------------- | -- expected result -- | -- expected result (conversion to base units) --
        pytest.param(
            ((2, "dB"), (1, "decibel")),
            (1, "delta_decibel"),
            (1, "delta_decibel"),
            id="dB-dB",
        ),
        pytest.param(
            ((2, "dBW"), (1, "decibelwatt")),
            (1, "delta_decibelwatt"),
            (1, "delta_decibelwatt"),
            id="dBW-dBW",
        ),
        pytest.param(
            ((2, "delta_dBW"), (1, "delta_decibelwatt")),
            (1, "delta_decibelwatt"),
            (1, "delta_decibelwatt"),
            id="delta_dBW-delta_dBW",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "decibel")),
            (-8, "dimensionless"),
            (-8, "dimensionless"),
            id="'' - dB",
        ),
        pytest.param(
            ((10, "decibel"), (2, "dimensionless")),
            (6.9897000433601875, "delta_decibel"),
            (6.9897000433601875, "delta_decibel"),
            id="dB - ''",
        ),  # no symmetry
        pytest.param(
            ((2, "dimensionless"), (10, "dBW")), "error", "error", id="'' - dBW"
        ),
        pytest.param(
            ((10, "dBW"), (2, "dimensionless")), "error", "error", id="dBW - ''"
        ),
        pytest.param(
            ((15, "watt"), (10, "dBW")), (5, "watt"), (5, "watt"), id="W - dBW"
        ),
        pytest.param(
            ((10, "dBW"), (8, "watt")),
            (0.9691001300805642, "delta_decibelwatt"),
            (0.9691001300805642, "delta_decibelwatt"),
            id="dBW - W",
        ),
        pytest.param(
            ((2, "dBW"), (1, "decibel")), "error", "error", id="dBW-dB"
        ),  # dimensionality error
        pytest.param(
            ((2, "dB"), (1, "delta_decibel")),
            (1, "decibel"),
            (1, "decibel"),
            id="dB-delta_dB",
        ),
        pytest.param(
            ((2, "delta_dB"), (1, "decibel")),
            (1, "decibel"),
            (1, "decibel"),
            id="delta_dB-dB",
        ),
        pytest.param(
            ((4, "dBW"), (1, "delta_decibelwatt")),
            (3, "decibelwatt"),
            (3, "decibelwatt"),
            id="dBW-delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (2, "dimensionless")),
            "error",
            "error",
            id="delta_dBW - ''",
        ),
        pytest.param(
            ((10, "dimensionless"), (2, "delta_dBW")),
            "error",
            "error",
            id="'' - delta_dBW",
        ),
        pytest.param(
            ((15, "watt"), (10, "delta_dBW")),
            (5, "watt"),
            (5, "watt"),
            id="W - delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (8, "watt")),
            (2, "watt"),
            (2, "watt"),
            id="delta_dBW - W",
        ),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected", "expected_base_units"), subtractions
    )
    def test_subtraction(self, input_tuple, expected, expected_base_units):

        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2

        self.ureg.autoconvert_offset_to_baseunit = False
        if expected == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.sub(q1, q2)
        else:
            expected = self.Q_(*expected)
            assert op.sub(q1, q2).units == expected.units
            helpers.assert_quantity_almost_equal(op.sub(q1, q2), expected, atol=0.01)

        self.ureg.autoconvert_offset_to_baseunit = True
        if expected_base_units == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.sub(q1, q2)
        else:
            expected_base_units = self.Q_(*expected_base_units)
            assert op.sub(q1, q2).units == expected_base_units.units
            helpers.assert_quantity_almost_equal(
                op.sub(q1, q2), expected_base_units, atol=0.01
            )

    multiplications = [
        # --- input tuple --| -- expected result --| -- expected result (conversion to base units) --
        pytest.param(
            ((2, "dB"), (1, "decibel")), "error", (2, "dimensionless"), id="dB*dB"
        ),
        pytest.param(
            ((0.2, "dBm"), (0.1, "decibelmilliwatt")),
            "error",
            (1.07, "gram ** 2 * meter ** 4 / second ** 6"),
            id="dBm*dBm",
        ),
        pytest.param(
            ((0.2, "dB"), (0.1, "decibelmilliwatt")),
            "error",
            (1.07, "gram * meter ** 2 / second ** 3"),
            id="dB*dBm",
        ),
        pytest.param(
            ((2, "delta_dBW"), (1, "delta_decibelwatt")),
            (2, "delta_decibelwatt ** 2"),
            (2, "delta_decibelwatt ** 2"),
            id="delta_dBW*delta_dBW",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "decibel")),
            "error",
            (20, "dimensionless"),
            id="'' * dB",
        ),
        pytest.param(
            ((10, "decibel"), (2, "dimensionless")),
            "error",
            (20, "dimensionless"),
            id="dB * ''",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "dBW")),
            "error",
            (20 * 10**3, "gram * meter ** 2 / second ** 3"),
            id="'' * dBW",
        ),
        pytest.param(
            ((10, "dBW"), (2, "dimensionless")),
            "error",
            (20 * 10**3, "gram * meter ** 2 / second ** 3"),
            id="dBW * ''",
        ),
        pytest.param(
            ((15, "watt"), (10, "dBW")),
            "error",
            (150 * 10**3, "watt * gram * meter ** 2 / second ** 3"),
            id="W*dBW",
        ),
        pytest.param(
            ((10, "dBW"), (8, "watt")),
            "error",
            (80 * 10**3, "watt * gram * meter ** 2 / second ** 3"),
            id="dBW*W",
        ),
        pytest.param(
            ((2, "dBW"), (1, "decibel")),
            "error",
            (1.99526 * 10**3, "gram * meter ** 2 / second ** 3"),
            id="dBW*dB",
        ),
        pytest.param(
            ((2, "dB"), (1, "delta_decibel")),
            "error",
            (1.584, "delta_decibel"),
            id="dB*delta_dB",
        ),
        pytest.param(
            ((1, "delta_dB"), (2, "decibel")),
            "error",
            (1.584, "delta_decibel"),
            id="delta_dB*dB",
        ),
        pytest.param(
            ((4, "dBW"), (1, "delta_decibelwatt")),
            "error",
            (2511.88, "delta_decibelwatt * gram * meter ** 2 / second ** 3"),
            id="dBW*delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (2, "dimensionless")),
            (20, "delta_dBW"),
            (20, "delta_dBW"),
            id="delta_dBW * ''",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "delta_dBW")),
            (20, "delta_dBW"),
            (20, "delta_dBW"),
            id="''*delta_dBW",
        ),
        pytest.param(
            ((15, "watt"), (10, "delta_dBW")),
            (150, "delta_dBW*watt"),
            (150, "delta_dBW*watt"),
            id="W*delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (8, "watt")),
            (80, "delta_dBW*watt"),
            (80, "delta_dBW*watt"),
            id="delta_dBW*W",
        ),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected", "expected_base_units"), multiplications
    )
    def test_multiplication(self, input_tuple, expected, expected_base_units):

        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2

        self.ureg.autoconvert_offset_to_baseunit = False
        if expected == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.mul(q1, q2)
        else:
            expected = self.Q_(*expected)
            assert op.mul(q1, q2).units == expected.units
            helpers.assert_quantity_almost_equal(op.mul(q1, q2), expected, atol=0.01)

        self.ureg.autoconvert_offset_to_baseunit = True
        if expected_base_units == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.mul(q1, q2)
        else:
            expected_base_units = self.Q_(*expected_base_units)
            assert op.mul(q1, q2).units == expected_base_units.units
            helpers.assert_quantity_almost_equal(
                op.mul(q1, q2), expected_base_units, atol=0.01
            )

    divisions = [
        # --- input tuple --| -- expected result --| -- expected result (conversion to base units) --
        pytest.param(
            ((4, "dB"), (2, "decibel")), "error", (1.5849, "dimensionless"), id="dB/dB"
        ),
        pytest.param(
            ((4, "dBm"), (2, "decibelmilliwatt")),
            "error",
            (1.5849, "dimensionless"),
            id="dBm/dBm",
        ),
        pytest.param(
            ((4, "delta_dBW"), (2, "delta_decibelwatt")),
            (2, "dimensionless"),
            (2, "dimensionless"),
            id="delta_dBW/delta_dBW",
        ),
        pytest.param(
            ((20, "dimensionless"), (10, "decibel")),
            "error",
            (2, "dimensionless"),
            id="'' / dB",
        ),
        pytest.param(
            ((10, "decibel"), (2, "dimensionless")),
            "error",
            (5, "dimensionless"),
            id="dB / ''",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "dBW")),
            "error",
            (0.2 * 10**-3, "second ** 3 / gram / meter ** 2"),
            id="'' / dBW",
        ),
        pytest.param(
            ((10, "dBW"), (2, "dimensionless")),
            "error",
            (5 * 10**3, "gram * meter ** 2 / second ** 3"),
            id="dBW / ''",
        ),
        pytest.param(
            ((15, "watt"), (10, "dBW")),
            "error",
            (1.5 * 10**-3, "watt * second ** 3 / gram / meter ** 2"),
            id="W/dBW",
        ),
        pytest.param(
            ((10, "dBW"), (2, "watt")),
            "error",
            (5 * 10**3, "gram * meter ** 2 / second ** 3 / watt"),
            id="dBW/W",
        ),
        pytest.param(
            ((2, "dBW"), (1, "decibel")),
            "error",
            (1.25892 * 10**3, "gram * meter ** 2 / second ** 3"),
            id="dBW/dB",
        ),
        pytest.param(
            ((10, "dB"), (2, "decibelmilliwatt")),
            "error",
            (6.3095, "second ** 3 / gram / meter ** 2"),
            id="dB/dBm",
        ),
        pytest.param(
            ((10, "dB"), (2, "delta_decibel")),
            "error",
            (5, "1 / delta_decibel"),
            id="dB/delta_dB",
        ),
        pytest.param(
            ((20, "delta_dB"), (10, "decibel")),
            "error",
            (2, "delta_decibel"),
            id="delta_dB/dB",
        ),
        pytest.param(
            ((10, "dBW"), (2, "delta_decibelwatt")),
            "error",
            (5 * 10**3, "gram * meter ** 2 / second ** 3 / delta_decibelwatt"),
            id="dBW/delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (2, "dimensionless")),
            (5, "delta_dBW"),
            (5, "delta_dBW"),
            id="delta_dBW / ''",
        ),
        pytest.param(
            ((2, "dimensionless"), (10, "delta_dBW")),
            (0.2, "1 / delta_dBW"),
            (0.2, "1 / delta_dBW"),
            id="''/delta_dBW",
        ),
        pytest.param(
            ((10, "watt"), (5, "delta_dBW")),
            (2, "watt/delta_dBW"),
            (2, "watt/delta_dBW"),
            id="W/delta_dBW",
        ),
        pytest.param(
            ((10, "delta_dBW"), (5, "watt")),
            (2, "delta_dBW/watt"),
            (2, "delta_dBW/watt"),
            id="delta_dBW/W",
        ),
    ]

    @pytest.mark.parametrize(
        ("input_tuple", "expected", "expected_base_units"), divisions
    )
    def test_true_division(self, input_tuple, expected, expected_base_units):

        qin1, qin2 = input_tuple
        q1, q2 = self.Q_(*qin1), self.Q_(*qin2)
        input_tuple = q1, q2

        self.ureg.autoconvert_offset_to_baseunit = False
        if expected == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.truediv(q1, q2)
        else:
            expected = self.Q_(*expected)
            assert op.truediv(q1, q2).units == expected.units
            helpers.assert_quantity_almost_equal(
                op.truediv(q1, q2), expected, atol=0.01
            )

        self.ureg.autoconvert_offset_to_baseunit = True
        if expected_base_units == "error":
            with pytest.raises(
                (
                    LogarithmicUnitCalculusError,
                    OffsetUnitCalculusError,
                    DimensionalityError,
                )
            ):
                op.truediv(q1, q2)
        else:
            expected_base_units = self.Q_(*expected_base_units)
            assert op.truediv(q1, q2).units == expected_base_units.units
            helpers.assert_quantity_almost_equal(
                op.truediv(q1, q2), expected_base_units, atol=0.01
            )
