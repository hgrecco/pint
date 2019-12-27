"""Tests for global UnitRegistry, Unit, and Quantity
"""
import pickle

from pint import (
    Measurement,
    Quantity,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
    get_application_registry,
    set_application_registry,
)
from pint.testsuite import BaseTestCase
from pint.testsuite.helpers import requires_uncertainties


class TestDefaultApplicationRegistry(BaseTestCase):
    def test_unit(self):
        u = Unit("kg")
        self.assertEqual(str(u), "kilogram")
        u = pickle.loads(pickle.dumps(u))
        self.assertEqual(str(u), "kilogram")

    def test_quantity_1arg(self):
        q = Quantity("123 kg")
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)

    def test_quantity_2args(self):
        q = Quantity(123, "kg")
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)

    @requires_uncertainties()
    def test_measurement_2args(self):
        m = Measurement(Quantity(123, "kg"), Quantity(15, "kg"))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 15)
        self.assertEqual(str(m.units), "kilogram")
        m = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 15)
        self.assertEqual(str(m.units), "kilogram")

    @requires_uncertainties()
    def test_measurement_3args(self):
        m = Measurement(123, 15, "kg")
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 15)
        self.assertEqual(str(m.units), "kilogram")
        m = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 15)
        self.assertEqual(str(m.units), "kilogram")

    def test_get_application_registry(self):
        ureg = get_application_registry()
        u = ureg.Unit("kg")
        self.assertEqual(str(u), "kilogram")

    def test_pickle_crash(self):
        ureg = UnitRegistry(None)
        ureg.define("foo = []")
        q = ureg.Quantity(123, "foo")
        b = pickle.dumps(q)
        self.assertRaises(UndefinedUnitError, pickle.loads, b)
        b = pickle.dumps(q.units)
        self.assertRaises(UndefinedUnitError, pickle.loads, b)

    @requires_uncertainties()
    def test_pickle_crash_measurement(self):
        ureg = UnitRegistry(None)
        ureg.define("foo = []")
        m = ureg.Quantity(123, "foo").plus_minus(10)
        b = pickle.dumps(m)
        self.assertRaises(UndefinedUnitError, pickle.loads, b)


class TestCustomApplicationRegistry(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.ureg_bak = get_application_registry()
        self.ureg = UnitRegistry(None)
        self.ureg.define("foo = []")
        self.ureg.define("bar = foo / 2")
        set_application_registry(self.ureg)
        assert get_application_registry() is self.ureg

    def tearDown(self):
        super().tearDown()
        set_application_registry(self.ureg_bak)

    def test_unit(self):
        u = Unit("foo")
        self.assertEqual(str(u), "foo")
        u = pickle.loads(pickle.dumps(u))
        self.assertEqual(str(u), "foo")

    def test_quantity_1arg(self):
        q = Quantity("123 foo")
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)

    def test_quantity_2args(self):
        q = Quantity(123, "foo")
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)

    @requires_uncertainties()
    def test_measurement_2args(self):
        m = Measurement(Quantity(123, "foo"), Quantity(10, "bar"))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")
        m = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")

    @requires_uncertainties()
    def test_measurement_3args(self):
        m = Measurement(123, 5, "foo")
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")
        m = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")


class TestSwapApplicationRegistry(BaseTestCase):
    """Test that the constructors of Quantity, Unit, and Measurement capture
    the registry that is set as the application registry at creation time

    Parameters
    ----------

    Returns
    -------

    """

    def setUp(self):
        super().setUp()
        self.ureg_bak = get_application_registry()
        self.ureg1 = UnitRegistry(None)
        self.ureg1.define("foo = [dim1]")
        self.ureg1.define("bar = foo / 2")
        self.ureg2 = UnitRegistry(None)
        self.ureg2.define("foo = [dim2]")
        self.ureg2.define("bar = foo / 3")

    def tearDown(self):
        set_application_registry(self.ureg_bak)

    def test_quantity_1arg(self):
        set_application_registry(self.ureg1)
        q1 = Quantity("1 foo")
        set_application_registry(self.ureg2)
        q2 = Quantity("1 foo")
        q3 = pickle.loads(pickle.dumps(q1))
        assert q1.dimensionality == {"[dim1]": 1}
        assert q2.dimensionality == {"[dim2]": 1}
        assert q3.dimensionality == {"[dim2]": 1}
        assert q1.to("bar").magnitude == 2
        assert q2.to("bar").magnitude == 3
        assert q3.to("bar").magnitude == 3

    def test_quantity_2args(self):
        set_application_registry(self.ureg1)
        q1 = Quantity(1, "foo")
        set_application_registry(self.ureg2)
        q2 = Quantity(1, "foo")
        q3 = pickle.loads(pickle.dumps(q1))
        assert q1.dimensionality == {"[dim1]": 1}
        assert q2.dimensionality == {"[dim2]": 1}
        assert q3.dimensionality == {"[dim2]": 1}
        assert q1.to("bar").magnitude == 2
        assert q2.to("bar").magnitude == 3
        assert q3.to("bar").magnitude == 3

    def test_unit(self):
        set_application_registry(self.ureg1)
        u1 = Unit("bar")
        set_application_registry(self.ureg2)
        u2 = Unit("bar")
        u3 = pickle.loads(pickle.dumps(u1))
        assert u1.dimensionality == {"[dim1]": 1}
        assert u2.dimensionality == {"[dim2]": 1}
        assert u3.dimensionality == {"[dim2]": 1}

    @requires_uncertainties()
    def test_measurement_2args(self):
        set_application_registry(self.ureg1)
        m1 = Measurement(Quantity(10, "foo"), Quantity(1, "foo"))
        set_application_registry(self.ureg2)
        m2 = Measurement(Quantity(10, "foo"), Quantity(1, "foo"))
        m3 = pickle.loads(pickle.dumps(m1))

        assert m1.dimensionality == {"[dim1]": 1}
        assert m2.dimensionality == {"[dim2]": 1}
        assert m3.dimensionality == {"[dim2]": 1}
        self.assertEqual(m1.to("bar").value.magnitude, 20)
        self.assertEqual(m2.to("bar").value.magnitude, 30)
        self.assertEqual(m3.to("bar").value.magnitude, 30)
        self.assertEqual(m1.to("bar").error.magnitude, 2)
        self.assertEqual(m2.to("bar").error.magnitude, 3)
        self.assertEqual(m3.to("bar").error.magnitude, 3)

    @requires_uncertainties()
    def test_measurement_3args(self):
        set_application_registry(self.ureg1)
        m1 = Measurement(10, 1, "foo")
        set_application_registry(self.ureg2)
        m2 = Measurement(10, 1, "foo")
        m3 = pickle.loads(pickle.dumps(m1))

        assert m1.dimensionality == {"[dim1]": 1}
        assert m2.dimensionality == {"[dim2]": 1}
        self.assertEqual(m1.to("bar").value.magnitude, 20)
        self.assertEqual(m2.to("bar").value.magnitude, 30)
        self.assertEqual(m3.to("bar").value.magnitude, 30)
        self.assertEqual(m1.to("bar").error.magnitude, 2)
        self.assertEqual(m2.to("bar").error.magnitude, 3)
        self.assertEqual(m3.to("bar").error.magnitude, 3)
