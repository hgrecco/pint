from __future__ import annotations

import pytest

from pint import get_application_registry
from pint.metrology import METROLOGY_API_VERSION, MetrologyNamespace
from pint.util import UnitsContainer


@pytest.fixture
def mn(func_registry):
    return func_registry.__metrology_namespace__()


def test_namespace_from_objects(func_registry):
    q = func_registry.Quantity(3, "m")
    for obj in (func_registry, q, q.unit):
        mn = obj.__metrology_namespace__()
        assert isinstance(mn, MetrologyNamespace)
        assert mn._registry is func_registry


def test_dimension_namespace_uses_application_registry():
    mn = UnitsContainer({"[length]": 1}).__metrology_namespace__()
    assert mn._registry is get_application_registry()


def test_api_version(func_registry):
    q = func_registry.Quantity(3, "m")
    assert isinstance(
        q.__metrology_namespace__(api_version=METROLOGY_API_VERSION),
        MetrologyNamespace,
    )
    with pytest.raises(ValueError, match="Unsupported metrology API version"):
        q.__metrology_namespace__(api_version="1999.01")


def test_quantity_value_and_unit(func_registry):
    q = func_registry.Quantity(3, "m/s")
    assert q.value == 3
    assert q.unit == func_registry.Unit("m/s")
    assert isinstance(q.unit, func_registry.Unit)


def test_unit_dimension(func_registry):
    u = func_registry.Unit("m/s")
    assert u.dimension == UnitsContainer({"[length]": 1, "[time]": -1})
    assert u.dimension == u.dimensionality


def test_asdimension(mn):
    length = mn.asdimension("[length]")
    time = mn.asdimension("[time]")
    assert length == UnitsContainer({"[length]": 1})
    assert mn.asdimension(length) == length
    assert length / time == mn.asdimension("[length] / [time]")
    assert length * length == length**2
    assert time.__rmul__(length) == length * time
    assert time.__rtruediv__(length) == length / time


def test_asunit(mn, func_registry):
    m = mn.asunit("m")
    assert m == func_registry.meter
    assert mn.asunit(m) == m
    assert m.dimension == mn.asdimension("[length]")
    assert m * m == m**2
    assert func_registry.s.__rtruediv__(m) == m / func_registry.s
    assert (m / func_registry.s).dimension == mn.asdimension("[length]/[time]")


def test_asquantity(mn, func_registry):
    q = mn.asquantity(3, unit="m")
    assert isinstance(q, func_registry.Quantity)
    assert q.value == 3
    assert q.unit == func_registry.meter
    converted = mn.asquantity(q, unit=mn.asunit("cm"))
    assert converted.value == 300
    assert converted.unit == func_registry.centimeter


def test_metrology_apis_protocols(func_registry):
    mapis = pytest.importorskip("metrology_apis")
    q = func_registry.Quantity(3, "m")
    assert isinstance(q.__metrology_namespace__(), mapis.MetrologyNamespace)
    assert isinstance(q, mapis.Quantity)
    assert isinstance(q.unit, mapis.Unit)
    assert isinstance(q.unit.dimension, mapis.Dimension)
