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
    set_application_registry
)

from pint.testsuite import BaseTestCase
from pint.testsuite.helpers import requires_uncertainties


class TestDefaultApplicationRegistry(BaseTestCase):
    def test_unit(self):
        u = Unit("kg")
        self.assertEqual(str(u), "kilogram")
        u = pickle.loads(pickle.dumps(u))
        self.assertEqual(str(u), "kilogram")

    def test_quantity(self):
        q = Quantity("123 kg")
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "kilogram")
        self.assertEqual(q.to("t").magnitude, 0.123)

    @requires_uncertainties()
    def test_measurement(self):
        m = Measurement(Quantity(123, 'kg'), Quantity(15, 'kg'))
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
        super(TestCustomApplicationRegistry, self).setUp()
        ureg = UnitRegistry(None)
        ureg.define("foo = []")
        ureg.define("bar = foo / 2")

        self.ureg = ureg
        self.ureg_bak = get_application_registry()
        set_application_registry(ureg)
        assert get_application_registry() is ureg

    def tearDown(self):
        super(TestCustomApplicationRegistry, self).tearDown()
        set_application_registry(self.ureg_bak)

    def test_unit(self):
        u = Unit("foo")
        self.assertEqual(str(u), "foo")
        u = pickle.loads(pickle.dumps(u))
        self.assertEqual(str(u), "foo")

    def test_quantity(self):
        q = Quantity("123 foo")
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)
        q = pickle.loads(pickle.dumps(q))
        self.assertEqual(str(q.units), "foo")
        self.assertEqual(q.to("bar").magnitude, 246)

    @requires_uncertainties()
    def test_measurement(self):
        m = Measurement(Quantity(123, 'foo'), Quantity(10, 'bar'))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")
        m = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.value.magnitude, 123)
        self.assertEqual(m.error.magnitude, 5)
        self.assertEqual(str(m.units), "foo")
