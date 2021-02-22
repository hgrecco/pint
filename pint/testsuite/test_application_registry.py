"""Tests for global UnitRegistry, Unit, and Quantity
"""
import pickle

import pytest

from pint import (
    Measurement,
    Quantity,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
    get_application_registry,
    set_application_registry,
)
from pint.testsuite import helpers


class TestDefaultApplicationRegistry:
    @helpers.allprotos
    def test_unit(self, protocol):
        u = Unit("kg")
        assert str(u) == "kilogram"
        u = pickle.loads(pickle.dumps(u, protocol))
        assert str(u) == "kilogram"

    @helpers.allprotos
    def test_quantity_1arg(self, protocol):
        q = Quantity("123 kg")
        assert str(q.units) == "kilogram"
        assert q.to("t").magnitude == 0.123
        q = pickle.loads(pickle.dumps(q, protocol))
        assert str(q.units) == "kilogram"
        assert q.to("t").magnitude == 0.123

    @helpers.allprotos
    def test_quantity_2args(self, protocol):
        q = Quantity(123, "kg")
        assert str(q.units) == "kilogram"
        assert q.to("t").magnitude == 0.123
        q = pickle.loads(pickle.dumps(q, protocol))
        assert str(q.units) == "kilogram"
        assert q.to("t").magnitude == 0.123

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_2args(self, protocol):
        m = Measurement(Quantity(123, "kg"), Quantity(15, "kg"))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 15
        assert str(m.units) == "kilogram"
        m = pickle.loads(pickle.dumps(m, protocol))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 15
        assert str(m.units) == "kilogram"

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_3args(self, protocol):
        m = Measurement(123, 15, "kg")
        assert m.value.magnitude == 123
        assert m.error.magnitude == 15
        assert str(m.units) == "kilogram"
        m = pickle.loads(pickle.dumps(m, protocol))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 15
        assert str(m.units) == "kilogram"

    def test_get_application_registry(self):
        ureg = get_application_registry()
        u = ureg.Unit("kg")
        assert str(u) == "kilogram"

    @helpers.allprotos
    def test_pickle_crash(self, protocol):
        ureg = UnitRegistry(None)
        ureg.define("foo = []")
        q = ureg.Quantity(123, "foo")
        b = pickle.dumps(q, protocol)
        with pytest.raises(UndefinedUnitError):
            pickle.loads(b)
        b = pickle.dumps(q.units, protocol)
        with pytest.raises(UndefinedUnitError):
            pickle.loads(b)

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_pickle_crash_measurement(self, protocol):
        ureg = UnitRegistry(None)
        ureg.define("foo = []")
        m = ureg.Quantity(123, "foo").plus_minus(10)
        b = pickle.dumps(m, protocol)
        with pytest.raises(UndefinedUnitError):
            pickle.loads(b)


class TestCustomApplicationRegistry:
    @classmethod
    def setup_class(cls):
        cls.ureg_bak = get_application_registry()
        cls.ureg = UnitRegistry(None)
        cls.ureg.define("foo = []")
        cls.ureg.define("bar = foo / 2")
        set_application_registry(cls.ureg)
        assert get_application_registry() is cls.ureg

    @classmethod
    def teardown_class(cls):
        set_application_registry(cls.ureg_bak)

    @helpers.allprotos
    def test_unit(self, protocol):
        u = Unit("foo")
        assert str(u) == "foo"
        u = pickle.loads(pickle.dumps(u, protocol))
        assert str(u) == "foo"

    @helpers.allprotos
    def test_quantity_1arg(self, protocol):
        q = Quantity("123 foo")
        assert str(q.units) == "foo"
        assert q.to("bar").magnitude == 246
        q = pickle.loads(pickle.dumps(q, protocol))
        assert str(q.units) == "foo"
        assert q.to("bar").magnitude == 246

    @helpers.allprotos
    def test_quantity_2args(self, protocol):
        q = Quantity(123, "foo")
        assert str(q.units) == "foo"
        assert q.to("bar").magnitude == 246
        q = pickle.loads(pickle.dumps(q, protocol))
        assert str(q.units) == "foo"
        assert q.to("bar").magnitude == 246

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_2args(self, protocol):
        m = Measurement(Quantity(123, "foo"), Quantity(10, "bar"))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 5
        assert str(m.units) == "foo"
        m = pickle.loads(pickle.dumps(m, protocol))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 5
        assert str(m.units) == "foo"

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_3args(self, protocol):
        m = Measurement(123, 5, "foo")
        assert m.value.magnitude == 123
        assert m.error.magnitude == 5
        assert str(m.units) == "foo"
        m = pickle.loads(pickle.dumps(m, protocol))
        assert m.value.magnitude == 123
        assert m.error.magnitude == 5
        assert str(m.units) == "foo"


class TestSwapApplicationRegistry:
    """Test that the constructors of Quantity, Unit, and Measurement capture
    the registry that is set as the application registry at creation time

    Parameters
    ----------

    Returns
    -------

    """

    @classmethod
    def setup_class(cls):
        cls.ureg_bak = get_application_registry()
        cls.ureg1 = UnitRegistry(None)
        cls.ureg1.define("foo = [dim1]")
        cls.ureg1.define("bar = foo / 2")
        cls.ureg2 = UnitRegistry(None)
        cls.ureg2.define("foo = [dim2]")
        cls.ureg2.define("bar = foo / 3")

    @classmethod
    def teardown_class(cls):
        set_application_registry(cls.ureg_bak)

    @helpers.allprotos
    def test_quantity_1arg(self, protocol):
        set_application_registry(self.ureg1)
        q1 = Quantity("1 foo")
        set_application_registry(self.ureg2)
        q2 = Quantity("1 foo")
        q3 = pickle.loads(pickle.dumps(q1, protocol))
        assert q1.dimensionality == {"[dim1]": 1}
        assert q2.dimensionality == {"[dim2]": 1}
        assert q3.dimensionality == {"[dim2]": 1}
        assert q1.to("bar").magnitude == 2
        assert q2.to("bar").magnitude == 3
        assert q3.to("bar").magnitude == 3

    @helpers.allprotos
    def test_quantity_2args(self, protocol):
        set_application_registry(self.ureg1)
        q1 = Quantity(1, "foo")
        set_application_registry(self.ureg2)
        q2 = Quantity(1, "foo")
        q3 = pickle.loads(pickle.dumps(q1, protocol))
        assert q1.dimensionality == {"[dim1]": 1}
        assert q2.dimensionality == {"[dim2]": 1}
        assert q3.dimensionality == {"[dim2]": 1}
        assert q1.to("bar").magnitude == 2
        assert q2.to("bar").magnitude == 3
        assert q3.to("bar").magnitude == 3

    @helpers.allprotos
    def test_unit(self, protocol):
        set_application_registry(self.ureg1)
        u1 = Unit("bar")
        set_application_registry(self.ureg2)
        u2 = Unit("bar")
        u3 = pickle.loads(pickle.dumps(u1, protocol))
        assert u1.dimensionality == {"[dim1]": 1}
        assert u2.dimensionality == {"[dim2]": 1}
        assert u3.dimensionality == {"[dim2]": 1}

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_2args(self, protocol):
        set_application_registry(self.ureg1)
        m1 = Measurement(Quantity(10, "foo"), Quantity(1, "foo"))
        set_application_registry(self.ureg2)
        m2 = Measurement(Quantity(10, "foo"), Quantity(1, "foo"))
        m3 = pickle.loads(pickle.dumps(m1, protocol))

        assert m1.dimensionality == {"[dim1]": 1}
        assert m2.dimensionality == {"[dim2]": 1}
        assert m3.dimensionality == {"[dim2]": 1}
        assert m1.to("bar").value.magnitude == 20
        assert m2.to("bar").value.magnitude == 30
        assert m3.to("bar").value.magnitude == 30
        assert m1.to("bar").error.magnitude == 2
        assert m2.to("bar").error.magnitude == 3
        assert m3.to("bar").error.magnitude == 3

    @helpers.requires_uncertainties()
    @helpers.allprotos
    def test_measurement_3args(self, protocol):
        set_application_registry(self.ureg1)
        m1 = Measurement(10, 1, "foo")
        set_application_registry(self.ureg2)
        m2 = Measurement(10, 1, "foo")
        m3 = pickle.loads(pickle.dumps(m1, protocol))

        assert m1.dimensionality == {"[dim1]": 1}
        assert m2.dimensionality == {"[dim2]": 1}
        assert m1.to("bar").value.magnitude == 20
        assert m2.to("bar").value.magnitude == 30
        assert m3.to("bar").value.magnitude == 30
        assert m1.to("bar").error.magnitude == 2
        assert m2.to("bar").error.magnitude == 3
        assert m3.to("bar").error.magnitude == 3
