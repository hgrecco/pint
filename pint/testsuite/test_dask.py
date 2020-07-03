import pytest

from pint import UnitRegistry

# Conditionally import NumPy and Dask
np = pytest.importorskip("numpy", reason="NumPy is not available")
da = pytest.importorskip("dask.array", reason="Dask is not available")

ureg = UnitRegistry(force_ndarray_like=True)
units_ = "kilogram"


@pytest.fixture
def dask_array():
    return da.arange(0, 25, chunks=5, dtype=float).reshape((5, 5))


@pytest.fixture
def numpy_array():
    return np.arange(0, 25, dtype=float).reshape((5, 5)) + 5


def test_has_dask_graph(dask_array):
    """Test that a pint.Quantity wrapped Dask array has a Dask graph."""
    q = ureg.Quantity(dask_array, units_)
    assert q.__dask_graph__() == dask_array.__dask_graph__()


def test_has_no_dask_graph(numpy_array):
    """Test that a pint.Quantity wrapped NumPy array does not have a Dask graph,
    and that attempting to access it returns None.
    """
    q = ureg.Quantity(numpy_array, units_)
    assert q.__dask_graph__() is None


def test_has_dask_keys(dask_array):
    """Test that a pint.Quantity wrapped Dask array has Dask keys."""
    q = ureg.Quantity(dask_array, units_)
    assert q.__dask_keys__() == dask_array.__dask_keys__()


def test_compute(dask_array, numpy_array):
    """Test the compute() method on a pint.Quantity wrapped Dask array."""
    q = ureg.Quantity(dask_array, units_)
    comps = q + 5 * ureg(units_)
    res = comps.compute()

    assert np.all(res.m == numpy_array)
    assert res.units == units_
    assert q.magnitude is dask_array


def test_persist(dask_array, numpy_array):
    """Test the persist() method on a pint.Quantity wrapped Dask array.

    For single machines, persist() is expected to return the computed result(s).
    """
    q = ureg.Quantity(dask_array, units_)
    comps = q + 5 * ureg(units_)
    res = comps.persist()

    assert np.all(res.m == numpy_array)
    assert res.units == units_
    assert q.magnitude is dask_array


def test_visualize():
    pass


def test_compute_exception(numpy_array):
    """Test exception handling for calling compute() on a pint.Quantity wrapped object
    that is not a dask.array.core.Array object.
    """
    q = ureg.Quantity(numpy_array, units_)
    comps = q + 5 * ureg(units_)
    with pytest.raises(AttributeError) as excinfo:
        comps.compute()

    exctruth = "Method compute only implemented for objects of <class 'dask.array.core.Array'>, not <class 'numpy.ndarray'>"
    assert str(excinfo.value) == exctruth


def test_persist_exception(numpy_array):
    """Test exception handling for calling persist() on a pint.Quantity wrapped object
    that is not a dask.array.core.Array object.
    """
    q = ureg.Quantity(numpy_array, units_)
    comps = q + 5 * ureg(units_)
    with pytest.raises(AttributeError) as excinfo:
        comps.persist()

    exctruth = "Method persist only implemented for objects of <class 'dask.array.core.Array'>, not <class 'numpy.ndarray'>"
    assert str(excinfo.value) == exctruth


def test_visualize_exception(numpy_array):
    """Test exception handling for calling visualize() on a pint.Quantity wrapped object
    that is not a dask.array.core.Array object.
    """
    q = ureg.Quantity(numpy_array, units_)
    comps = q + 5 * ureg(units_)
    with pytest.raises(AttributeError) as excinfo:
        comps.visualize()

    exctruth = "Method visualize only implemented for objects of <class 'dask.array.core.Array'>, not <class 'numpy.ndarray'>"
    assert str(excinfo.value) == exctruth
