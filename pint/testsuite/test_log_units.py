import math
import unittest

import pytest

from pint import LogarithmicUnitCalculusError, OffsetUnitCalculusError, UnitRegistry
from pint.testsuite import QuantityTestCase
from pint.unit import UnitsContainer


@pytest.fixture(scope="module")
def auto_ureg():
    return UnitRegistry(autoconvert_offset_to_baseunit=True)


@pytest.fixture(scope="module")
def ureg():
    return UnitRegistry()


class TestLogarithmicQuantity(QuantityTestCase):

    FORCE_NDARRAY = False

    def test_log_quantity_creation(self):

        # Following Quantity Creation Pattern
        for args in (
            (4.2, "dBm"),
            (4.2, UnitsContainer(decibellmilliwatt=1)),
            (4.2, self.ureg.dBm),
        ):
            x = self.Q_(*args)
            self.assertEqual(x.magnitude, 4.2)
            self.assertEqual(x.units, UnitsContainer(decibellmilliwatt=1))

        x = self.Q_(self.Q_(4.2, "dBm"))
        self.assertEqual(x.magnitude, 4.2)
        self.assertEqual(x.units, UnitsContainer(decibellmilliwatt=1))

        x = self.Q_(4.2, UnitsContainer(decibellmilliwatt=1))
        y = self.Q_(x)
        self.assertEqual(x.magnitude, y.magnitude)
        self.assertEqual(x.units, y.units)
        self.assertIsNot(x, y)

        # Using multiplications for dB units requires autoconversion to baseunits
        new_reg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        x = new_reg.Quantity("4.2 * dBm")
        self.assertEqual(x.magnitude, 4.2)
        self.assertEqual(x.units, UnitsContainer(decibellmilliwatt=1))

        with self.capture_log() as buffer:
            self.assertEqual(4.2 * new_reg.dBm, new_reg.Quantity(4.2, 2 * new_reg.dBm))
            self.assertEqual(len(buffer), 1)

    def test_log_convert(self):

        # ## Test dB
        # 0 dB == 1
        self.assertQuantityAlmostEqual(
            self.Q_(0.0, "dB").to("dimensionless"), self.Q_(1.0)
        )
        # -10 dB == 0.1
        self.assertQuantityAlmostEqual(
            self.Q_(-10.0, "dB").to("dimensionless"), self.Q_(0.1)
        )
        # +10 dB == 10
        self.assertQuantityAlmostEqual(
            self.Q_(+10.0, "dB").to("dimensionless"), self.Q_(10.0)
        )
        # 30 dB == 1e3
        self.assertQuantityAlmostEqual(
            self.Q_(30.0, "dB").to("dimensionless"), self.Q_(1e3)
        )
        # 60 dB == 1e6
        self.assertQuantityAlmostEqual(
            self.Q_(60.0, "dB").to("dimensionless"), self.Q_(1e6)
        )
        # # 1 dB = 1/10 * bel
        # self.assertQuantityAlmostEqual(self.Q_(1.0, "dB").to("dimensionless"), self.Q_(1, "bell") / 10)
        # # Uncomment Bell unit in default_en.txt

        # ## Test decade
        # 1 decade == 10
        self.assertQuantityAlmostEqual(
            self.Q_(1.0, "decade").to("dimensionless"), self.Q_(10.0)
        )
        # 2 decade == 100
        self.assertQuantityAlmostEqual(
            self.Q_(2.0, "decade").to("dimensionless"), self.Q_(100.0)
        )

        # ## Test octave
        # 1 octave = 2
        self.assertQuantityAlmostEqual(
            self.Q_(1.0, "octave").to("dimensionless"), self.Q_(2.0)
        )

        # ## Test dB to dB units octave - decade
        # 1 decade = log2(10) octave
        self.assertQuantityAlmostEqual(
            self.Q_(1.0, "decade"), self.Q_(math.log(10, 2), "octave")
        )

        # ## Test dBm
        # 0 dBm = 1 mW
        self.assertQuantityAlmostEqual(self.Q_(0.0, "dBm").to("mW"), self.Q_(1.0, "mW"))
        self.assertQuantityAlmostEqual(
            self.Q_(0.0, "dBm"), self.Q_(1.0, "mW").to("dBm")
        )
        # 10 dBm = 10 mW
        self.assertQuantityAlmostEqual(
            self.Q_(10.0, "dBm").to("mW"), self.Q_(10.0, "mW")
        )
        self.assertQuantityAlmostEqual(
            self.Q_(10.0, "dBm"), self.Q_(10.0, "mW").to("dBm")
        )
        # 20 dBm = 100 mW
        self.assertQuantityAlmostEqual(
            self.Q_(20.0, "dBm").to("mW"), self.Q_(100.0, "mW")
        )
        self.assertQuantityAlmostEqual(
            self.Q_(20.0, "dBm"), self.Q_(100.0, "mW").to("dBm")
        )
        # -10 dBm = 0.1 mW
        self.assertQuantityAlmostEqual(
            self.Q_(-10.0, "dBm").to("mW"), self.Q_(0.1, "mW")
        )
        self.assertQuantityAlmostEqual(
            self.Q_(-10.0, "dBm"), self.Q_(0.1, "mW").to("dBm")
        )
        # -20 dBm = 0.01 mW
        self.assertQuantityAlmostEqual(
            self.Q_(-20.0, "dBm").to("mW"), self.Q_(0.01, "mW")
        )
        self.assertQuantityAlmostEqual(
            self.Q_(-20.0, "dBm"), self.Q_(0.01, "mW").to("dBm")
        )

        # ## Test dB to dB units dBm - dBu
        # 0 dBm = 1mW = 1e3 uW = 30 dBu
        self.assertAlmostEqual(self.Q_(0.0, "dBm"), self.Q_(29.999999999999996, "dBu"))

    def test_mix_regular_log_units(self):
        # Test regular-logarithmic mixed definition, such as dB/km or dB/cm

        # Multiplications and divisions with a mix of Logarithmic Units and regular Units is normally not possible.
        # The reason is that dB are considered by pint like offset units.
        # Multiplications and divisions that involve offset units are badly defined, so pint raises an error
        with self.assertRaises(OffsetUnitCalculusError):
            (-10.0 * self.ureg.dB) / (1 * self.ureg.cm)

        # However, if the flag autoconvert_offset_to_baseunit=True is given to UnitRegistry, then pint converts the unit to base.
        # With this flag on multiplications and divisions are now possible:
        new_reg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        self.assertQuantityAlmostEqual(-10 * new_reg.dB / new_reg.cm, 0.1 / new_reg.cm)


def test_compound_log_unit_multiply_definition(auto_ureg):
    """Check that compound log units can be defined using multiply.
    """
    Q_ = auto_ureg.Quantity
    canonical_def = Q_(-161, "dBm") / auto_ureg.Hz
    mult_def = -161 * auto_ureg["dBm/Hz"]
    assert mult_def == canonical_def


def test_compound_log_unit_quantity_definition(auto_ureg):
    """Check that compound log units can be defined using ``Quantity()``.
    """
    Q_ = auto_ureg.Quantity
    canonical_def = Q_(-161, "dBm") / auto_ureg.Hz
    quantity_def = Q_(-161, "dBm/Hz")
    assert quantity_def == canonical_def


def test_compound_log_unit_parse_definition(auto_ureg):
    """Check that compound log units can be defined using ``parse_expression()``.
    """
    Q_ = auto_ureg.Quantity
    canonical_def = Q_(-161, "dBm") / auto_ureg.Hz
    parse_def = auto_ureg.parse_expression("-161 dBm/Hz")
    assert canonical_def == parse_def


class TestLogarithmicQuantityBasicMath(QuantityTestCase):

    FORCE_NDARRAY = False

    @unittest.expectedFailure
    def _test_log_quantity_add_sub_raises_exception(self, unit, func):
        # Warning should be provided when trying to ....
        self.assertRaises(LogarithmicUnitCalculusError)

    @unittest.expectedFailure
    def _test_log_quantity_add_sub(self, unit, func):

        # Pure dB arithmetic
        # 5 dBm + 10 dB = 15 dBm
        self.assertQuantityAlmostEqual(
            5 * self.ureg.dBm + 10 * self.ureg.dB, 15 * self.ureg.dBm
        )
        # 100*dBm -10*dB = 90*dB
        self.assertQuantityAlmostEqual(
            100 * self.ureg.dB - 10 * self.ureg.dB, 90 * self.ureg.dB
        )
        # 100 dBW - 5 dBW = 95 dB
        self.assertQuantityAlmostEqual(
            100 * self.ureg.dBm - 5 * self.ureg.dBm, 95 * self.ureg.dB
        )
        # 20 dB + 0 dBW == 20 dBW

        # 100 Hz + 1 octave = 200 Hz
        self.assertQuantityAlmostEqual(
            100 * self.ureg.Hz + 1 * self.ureg.octave, 200 * self.ureg.Hz
        )
