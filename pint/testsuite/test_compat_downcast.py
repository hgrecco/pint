import pytest

from pint import UnitRegistry

# Conditionally import NumPy and any upcast type libraries
np = pytest.importorskip("numpy", reason="NumPy is not available")
sparse = pytest.importorskip("sparse", reason="sparse is not available")
da = pytest.importorskip("dask.array", reason="Dask is not available")


def WR(func):
    """Function to wrap another containing 1 argument.
    Used to parametrize tests in which some cases depend
    on the registry while avoiding to create it at the module level
    """
    return lambda ureg, x: func(x)


def WR2(func):
    """Function to wrap another containing 2 argument.
    Used to parametrize tests in which some cases depend
    on the registry while avoiding to create it at the module level
    """
    return lambda ureg, x, y: func(x, y)


@pytest.fixture(scope="module")
def local_registry():
    # Set up unit registry and sample
    return UnitRegistry(force_ndarray_like=True)


@pytest.fixture(scope="module")
def q_base(local_registry):
    # Set up unit registry and sample
    return (np.arange(25).reshape(5, 5).T + 1) * local_registry.kg


# Define identity function for use in tests
def identity(ureg, x):
    return x


@pytest.fixture(params=["sparse", "masked_array", "dask_array"])
def array(request):
    """Generate 5x5 arrays of given type for tests."""
    if request.param == "sparse":
        # Create sample sparse COO as a permutation matrix.
        coords = [[0, 1, 2, 3, 4], [1, 3, 0, 2, 4]]
        data = [1.0] * 5
        return sparse.COO(coords, data, shape=(5, 5))
    elif request.param == "masked_array":
        # Create sample masked array as an upper triangular matrix.
        return np.ma.masked_array(
            np.arange(25, dtype=float).reshape((5, 5)),
            mask=np.logical_not(np.triu(np.ones((5, 5)))),
        )
    elif request.param == "dask_array":
        return da.arange(25, chunks=5, dtype=float).reshape((5, 5))


@pytest.mark.parametrize(
    "op, magnitude_op, unit_op",
    [
        pytest.param(identity, identity, identity, id="identity"),
        pytest.param(
            lambda ureg, x: x + 1 * ureg.m,
            lambda ureg, x: x + 1,
            identity,
            id="addition",
        ),
        pytest.param(
            lambda ureg, x: x - 20 * ureg.cm,
            lambda ureg, x: x - 0.2,
            identity,
            id="subtraction",
        ),
        pytest.param(
            lambda ureg, x: x * (2 * ureg.s),
            lambda ureg, x: 2 * x,
            lambda ureg, u: u * ureg.s,
            id="multiplication",
        ),
        pytest.param(
            lambda ureg, x: x / (1 * ureg.s),
            identity,
            lambda ureg, u: u / ureg.s,
            id="division",
        ),
        pytest.param(
            WR(lambda x: x**2),
            WR(lambda x: x**2),
            WR(lambda u: u**2),
            id="square",
        ),
        pytest.param(WR(lambda x: x.T), WR(lambda x: x.T), identity, id="transpose"),
        pytest.param(WR(np.mean), WR(np.mean), identity, id="mean ufunc"),
        pytest.param(WR(np.sum), WR(np.sum), identity, id="sum ufunc"),
        pytest.param(WR(np.sqrt), WR(np.sqrt), WR(lambda u: u**0.5), id="sqrt ufunc"),
        pytest.param(
            WR(lambda x: np.reshape(x, (25,))),
            WR(lambda x: np.reshape(x, (25,))),
            identity,
            id="reshape function",
        ),
        pytest.param(WR(np.amax), WR(np.amax), identity, id="amax function"),
    ],
)
def test_univariate_op_consistency(
    local_registry, q_base, op, magnitude_op, unit_op, array
):
    q = local_registry.Quantity(array, "meter")
    res = op(local_registry, q)
    assert np.all(
        res.magnitude == magnitude_op(local_registry, array)
    )  # Magnitude check
    assert res.units == unit_op(local_registry, q.units)  # Unit check
    assert q.magnitude is array  # Immutability check


@pytest.mark.parametrize(
    "op, unit",
    [
        pytest.param(
            lambda x, y: x * y, lambda ureg: ureg("kg m"), id="multiplication"
        ),
        pytest.param(lambda x, y: x / y, lambda ureg: ureg("m / kg"), id="division"),
        pytest.param(np.multiply, lambda ureg: ureg("kg m"), id="multiply ufunc"),
    ],
)
def test_bivariate_op_consistency(local_registry, q_base, op, unit, array):
    # This is to avoid having a ureg built at the module level.
    unit = unit(local_registry)

    q = local_registry.Quantity(array, "meter")
    res = op(q, q_base)
    assert np.all(res.magnitude == op(array, q_base.magnitude))  # Magnitude check
    assert res.units == unit  # Unit check
    assert q.magnitude is array  # Immutability check


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            WR2(lambda a, u: a * u),
            id="array-first",
            marks=pytest.mark.xfail(reason="upstream issue numpy/numpy#15200"),
        ),
        pytest.param(WR2(lambda a, u: u * a), id="unit-first"),
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        pytest.param(lambda ureg: ureg.m, id="Unit"),
        pytest.param(lambda ureg: ureg("meter"), id="Quantity"),
    ],
)
def test_array_quantity_creation_by_multiplication(
    local_registry, q_base, op, unit, array
):
    # This is to avoid having a ureg built at the module level.
    unit = unit(local_registry)

    assert type(op(local_registry, array, unit)) == local_registry.Quantity
