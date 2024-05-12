from __future__ import annotations

import importlib
import pathlib

import pytest

from pint import UnitRegistry

# Conditionally import NumPy, Dask, and Distributed
np = pytest.importorskip("numpy", reason="NumPy is not available")
dask = pytest.importorskip("dask", reason="Dask is not available")
distributed = pytest.importorskip("distributed", reason="Distributed is not available")

from dask.distributed import Client
from distributed.client import futures_of
from distributed.utils_test import (  # noqa: F401
    cleanup,
    cluster,
    gen_cluster,
    loop,
    loop_in_thread,
)

loop = loop  # flake8

units_ = "kilogram"


@pytest.fixture(scope="module")
def local_registry():
    # Set up unit registry and sample
    return UnitRegistry(force_ndarray_like=True)


def add_five(local_registry, q):
    return q + 5 * local_registry(units_)


@pytest.fixture
def dask_array():
    return dask.array.arange(0, 25, chunks=5, dtype=float).reshape((5, 5))


@pytest.fixture
def numpy_array():
    return np.arange(0, 25, dtype=float).reshape((5, 5)) + 5


def test_is_dask_collection(local_registry, dask_array):
    """Test that a pint.Quantity wrapped Dask array is a Dask collection."""
    q = local_registry.Quantity(dask_array, units_)
    assert dask.is_dask_collection(q)


def test_is_not_dask_collection(local_registry, numpy_array):
    """Test that other pint.Quantity wrapped objects are not Dask collections."""
    q = local_registry.Quantity(numpy_array, units_)
    assert not dask.is_dask_collection(q)


def test_dask_scheduler(local_registry, dask_array):
    """Test that a pint.Quantity wrapped Dask array has the correct default scheduler."""
    q = local_registry.Quantity(dask_array, units_)

    scheduler = q.__dask_scheduler__
    scheduler_name = f"{scheduler.__module__}.{scheduler.__name__}"

    true_name = "dask.threaded.get"

    assert scheduler == dask.array.Array.__dask_scheduler__
    assert scheduler_name == true_name


@pytest.mark.parametrize(
    "item",
    (
        pytest.param(1, id="int"),
        pytest.param(1.3, id="float"),
        pytest.param(np.array([1, 2]), id="numpy"),
        pytest.param(
            dask.array.arange(0, 25, chunks=5, dtype=float).reshape((5, 5)), id="dask"
        ),
    ),
)
def test_dask_tokenize(local_registry, item):
    """Test that a pint.Quantity wrapping something has a unique token."""
    dask_token = dask.base.tokenize(item)
    q = local_registry.Quantity(item, units_)

    assert dask.base.tokenize(item) != dask.base.tokenize(q)
    assert dask.base.tokenize(item) == dask_token


def test_dask_optimize(local_registry, dask_array):
    """Test that a pint.Quantity wrapped Dask array can be optimized."""
    q = local_registry.Quantity(dask_array, units_)

    assert q.__dask_optimize__ == dask.array.Array.__dask_optimize__


def test_compute(local_registry, dask_array, numpy_array):
    """Test the compute() method on a pint.Quantity wrapped Dask array."""
    q = local_registry.Quantity(dask_array, units_)

    comps = add_five(local_registry, q)
    res = comps.compute()

    assert np.all(res.m == numpy_array)
    assert not dask.is_dask_collection(res)
    assert res.units == units_
    assert q.magnitude is dask_array


def test_persist(local_registry, dask_array, numpy_array):
    """Test the persist() method on a pint.Quantity wrapped Dask array."""
    q = local_registry.Quantity(dask_array, units_)

    comps = add_five(local_registry, q)
    res = comps.persist()

    assert np.all(res.m == numpy_array)
    assert dask.is_dask_collection(res)
    assert res.units == units_
    assert q.magnitude is dask_array


@pytest.mark.skipif(
    importlib.util.find_spec("graphviz") is None, reason="GraphViz is not available"
)
def test_visualize(local_registry, dask_array):
    """Test the visualize() method on a pint.Quantity wrapped Dask array."""
    q = local_registry.Quantity(dask_array, units_)

    comps = add_five(local_registry, q)
    res = comps.visualize()

    assert res is None
    # These commands only work on Unix and Windows
    assert pathlib.Path("mydask.png").exists()
    pathlib.Path("mydask.png").unlink()


def test_compute_persist_equivalent(local_registry, dask_array, numpy_array):
    """Test that compute() and persist() return the same numeric results."""
    q = local_registry.Quantity(dask_array, units_)

    comps = add_five(local_registry, q)
    res_compute = comps.compute()
    res_persist = comps.persist()

    assert np.all(res_compute == res_persist)
    assert res_compute.units == res_persist.units == units_
    assert type(res_compute) == local_registry.Quantity
    assert type(res_persist) == local_registry.Quantity


@pytest.mark.parametrize("method", ["compute", "persist", "visualize"])
def test_exception_method_not_implemented(local_registry, numpy_array, method):
    """Test exception handling for convenience methods on a pint.Quantity wrapped
    object that is not a dask.array.Array object.
    """
    q = local_registry.Quantity(numpy_array, units_)

    exctruth = (
        f"Method {method} only implemented for objects of"
        " <class 'dask.array.core.Array'>, not"
        " <class 'numpy.ndarray'>"
    )
    with pytest.raises(AttributeError, match=exctruth):
        obj_method = getattr(q, method)
        obj_method()


def test_distributed_compute(local_registry, loop, dask_array, numpy_array):
    """Test compute() for distributed machines."""
    q = local_registry.Quantity(dask_array, units_)

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            comps = add_five(local_registry, q)
            res = comps.compute()

            assert np.all(res.m == numpy_array)
            assert not dask.is_dask_collection(res)
            assert res.units == units_

    assert q.magnitude is dask_array


def test_distributed_persist(local_registry, loop, dask_array):
    """Test persist() for distributed machines."""
    q = local_registry.Quantity(dask_array, units_)

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            comps = add_five(local_registry, q)
            persisted_q = comps.persist()

            comps_truth = dask_array + 5
            persisted_truth = comps_truth.persist()

            assert np.all(persisted_q.m == persisted_truth)
            assert dask.is_dask_collection(persisted_q)
            assert persisted_q.units == units_

    assert q.magnitude is dask_array


@gen_cluster(client=True)
async def test_async(c, s, a, b):
    """Test asynchronous operations."""

    # TODO: use a fixture for this.
    local_registry = UnitRegistry(force_ndarray_like=True)

    da = dask.array.arange(0, 25, chunks=5, dtype=float).reshape((5, 5))
    q = local_registry.Quantity(da, units_)

    x = q + local_registry.Quantity(5, units_)
    y = x.persist()
    assert str(y)

    assert dask.is_dask_collection(y)
    assert len(x.__dask_graph__()) > len(y.__dask_graph__())

    assert not futures_of(x)
    assert futures_of(y)

    future = c.compute(y)
    w = await future
    assert not dask.is_dask_collection(w)

    truth = np.arange(0, 25, dtype=float).reshape((5, 5)) + 5
    assert np.all(truth == w.m)
