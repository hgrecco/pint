import os

import pytest

from pint import UnitRegistry

# Conditionally import NumPy and Dask
np = pytest.importorskip("numpy", reason="NumPy is not available")
dask = pytest.importorskip("dask", reason="Dask is not available")

ureg = UnitRegistry(force_ndarray_like=True)
units_ = "kilogram"


def add_five(q):
    return q + 5 * ureg(units_)


@pytest.fixture
def dask_array():
    return dask.array.arange(0, 25, chunks=5, dtype=float).reshape((5, 5))


@pytest.fixture
def numpy_array():
    return np.arange(0, 25, dtype=float).reshape((5, 5)) + 5


@pytest.mark.parametrize("component", ["__dask_graph__", "__dask_keys__"])
def test_has_dask_components(dask_array, component):
    """Test that a pint.Quantity wrapped Dask array has a Dask graph and keys"""
    q = ureg.Quantity(dask_array, units_)

    q_method = getattr(q, component)
    component_q = q_method()

    truth_method = getattr(dask_array, component)
    component_truth = truth_method()

    assert component_q == component_truth


def test_has_no_dask_graph(numpy_array):
    """Test that a pint.Quantity wrapped NumPy array does not have a Dask graph
    and that attempting to access it returns None.
    """
    q = ureg.Quantity(numpy_array, units_)
    assert q.__dask_graph__() is None


def test_dask_scheduler(dask_array):
    """Test that a pint.Quantity wrapped Dask array has the correct default scheduler."""
    q = ureg.Quantity(dask_array, units_)

    scheduler = q.__dask_scheduler__
    scheduler_name = f"{scheduler.__module__}.{scheduler.__name__}"

    true_name = "dask.threaded.get"

    assert scheduler == dask.array.Array.__dask_scheduler__
    assert scheduler_name == true_name


def test_dask_optimize(dask_array):
    """Test that a pint.Quantity wrapped Dask array can be optimized."""
    q = ureg.Quantity(dask_array, units_)

    assert q.__dask_optimize__ == dask.array.Array.__dask_optimize__


@pytest.mark.parametrize("method", ["compute", "persist"])
def test_convenience_methods(dask_array, numpy_array, method):
    """Test convenience methods compute() and persist() on a pint.Quantity
    wrapped Dask array.
    """
    q = ureg.Quantity(dask_array, units_)

    comps = add_five(q)
    obj_method = getattr(comps, method)
    res = obj_method()

    assert np.all(res.m == numpy_array)
    assert res.units == units_
    assert q.magnitude is dask_array


def test_compute_persist_equivalent_single_machine(dask_array, numpy_array):
    """Test that compute() and persist() return the same result for calls made
    on a single machine.
    """
    q = ureg.Quantity(dask_array, units_)

    comps = add_five(q)
    res_compute = comps.compute()
    res_persist = comps.persist()

    assert np.all(res_compute == res_persist)
    assert res_compute.units == res_persist.units == units_


def test_visualize(dask_array):
    """Test the visualize() method on a pint.Quantity wrapped Dask array."""
    q = ureg.Quantity(dask_array, units_)

    comps = add_five(q)
    res = comps.visualize()

    assert res is None
    # These commands only work on Unix and Windows
    assert os.path.exists("mydask.png")
    os.remove("mydask.png")


@pytest.mark.parametrize("method", ["compute", "persist", "visualize"])
def test_exception_method_not_implemented(numpy_array, method):
    """Test exception handling for convenience methods on a pint.Quantity wrapped
    object that is not a dask.array.Array object.
    """
    q = ureg.Quantity(numpy_array, units_)

    exctruth = (
        f"Method {method} only implemented for objects of"
        " <class 'dask.array.core.Array'>, not"
        " <class 'numpy.ndarray'>"
    )
    with pytest.raises(AttributeError, match=exctruth):
        obj_method = getattr(q, method)
        obj_method()
