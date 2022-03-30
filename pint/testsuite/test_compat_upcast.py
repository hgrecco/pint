import pytest

# Conditionally import NumPy and any upcast type libraries
np = pytest.importorskip("numpy", reason="NumPy is not available")
xr = pytest.importorskip("xarray", reason="xarray is not available")


@pytest.fixture(scope="module")
def q_base(module_registry):
    # Set up unit registry and sample
    return [[1.0, 2.0], [3.0, 4.0]] * module_registry.m


@pytest.fixture
def da(q_base):
    return xr.DataArray(q_base.copy())


@pytest.fixture
def ds():
    return xr.Dataset(
        {
            "a": (("x", "y"), [[0, 1], [2, 3], [4, 5]], {"units": "K"}),
            "b": ("x", [0, 2, 4], {"units": "degC"}),
            "c": ("y", [-1, 1], {"units": "hPa"}),
        },
        coords={
            "x": ("x", [-1, 0, 1], {"units": "degree"}),
            "y": ("y", [0, 1], {"units": "degree"}),
        },
    )


def test_xarray_quantity_creation(module_registry, q_base):
    with pytest.raises(TypeError) as exc:
        module_registry.Quantity(xr.DataArray(np.arange(4)), "m")
        assert "Quantity cannot wrap upcast type" in str(exc)
    assert xr.DataArray(q_base).data is q_base


def test_quantification(module_registry, ds):
    da = ds["a"]
    da.data = module_registry.Quantity(da.values, da.attrs.pop("units"))
    mean = da.mean().item()
    assert mean.units == module_registry.K
    assert np.isclose(mean, 2.5 * module_registry.K)


@pytest.mark.parametrize(
    "op",
    [
        lambda x, y: x + y,
        lambda x, y: x - (-y),
        lambda x, y: x * y,
        lambda x, y: x / (y**-1),
    ],
)
@pytest.mark.parametrize(
    "pair",
    [
        (lambda ureg, q: q, lambda ureg, q: xr.DataArray(q)),
        (
            lambda ureg, q: xr.DataArray([1.0, 2.0] * ureg.m, dims=("y",)),
            lambda ureg, q: xr.DataArray(
                np.arange(6, dtype="float").reshape(3, 2, 1), dims=("z", "y", "x")
            )
            * ureg.km,
        ),
        (lambda ureg, q: 1 * ureg.m, lambda ureg, q: xr.DataArray(q)),
    ],
)
def test_binary_arithmetic_commutativity(module_registry, q_base, op, pair):
    pair = tuple(p(module_registry, q_base) for p in pair)
    z0 = op(*pair)
    z1 = op(*pair[::-1])
    z1 = z1.transpose(*z0.dims)
    assert np.all(np.isclose(z0.data, z1.data.to(z0.data.units)))


def test_eq_commutativity(da, q_base):
    assert np.all((q_base.T == da) == (da.transpose() == q_base))


def test_ne_commutativity(da, q_base):
    assert np.all((q_base != da.transpose()) == (da != q_base.T))


def test_dataset_operation_with_unit(ds, module_registry):
    ds0 = module_registry.K * ds.isel(x=0)
    ds1 = (ds * module_registry.K).isel(x=0)
    xr.testing.assert_identical(ds0, ds1)
    assert np.isclose(ds0["a"].mean().item(), 0.5 * module_registry.K)


def test_dataarray_inplace_arithmetic_roundtrip(da, module_registry, q_base):
    da_original = da.copy()
    q_to_modify = q_base.copy()
    da += q_base
    xr.testing.assert_identical(da, xr.DataArray([[2, 4], [6, 8]] * module_registry.m))
    da -= q_base
    xr.testing.assert_identical(da, da_original)
    da *= module_registry.m
    xr.testing.assert_identical(da, xr.DataArray(q_base * module_registry.m))
    da /= module_registry.m
    xr.testing.assert_identical(da, da_original)
    # Operating inplace with DataArray converts to DataArray
    q_to_modify += da
    q_to_modify -= da
    assert np.all(np.isclose(q_to_modify.data, q_base))


def test_dataarray_inequalities(da, module_registry):
    xr.testing.assert_identical(
        2 * module_registry.m > da, xr.DataArray([[True, False], [False, False]])
    )
    xr.testing.assert_identical(
        2 * module_registry.m < da, xr.DataArray([[False, False], [True, True]])
    )
    with pytest.raises(ValueError) as exc:
        da > 2
        assert "Cannot compare Quantity and <class 'int'>" in str(exc)


def test_array_function_deferral(da, module_registry):
    lower = 2 * module_registry.m
    upper = 3 * module_registry.m
    args = (da, lower, upper)
    assert (
        lower.__array_function__(
            np.clip, tuple(set(type(arg) for arg in args)), args, {}
        )
        is NotImplemented
    )


def test_array_ufunc_deferral(da, module_registry):
    lower = 2 * module_registry.m
    assert lower.__array_ufunc__(np.maximum, "__call__", lower, da) is NotImplemented
