import copy
import operator as op
import pickle
import warnings

import pytest

from pint import DimensionalityError, OffsetUnitCalculusError, UnitStrippedWarning
from pint.compat import np
from pint.testsuite import helpers
from pint.testsuite.test_umath import TestUFuncs


@helpers.requires_numpy
class TestNumpyMethods:
    @classmethod
    def setup_class(cls):
        from pint import _DEFAULT_REGISTRY

        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity

    @classmethod
    def teardown_class(cls):
        cls.ureg = None
        cls.Q_ = None

    @property
    def q(self):
        return [[1, 2], [3, 4]] * self.ureg.m

    @property
    def q_scalar(self):
        return np.array(5) * self.ureg.m

    @property
    def q_nan(self):
        return [[1, 2], [3, np.nan]] * self.ureg.m

    @property
    def q_zero_or_nan(self):
        return [[0, 0], [0, np.nan]] * self.ureg.m

    @property
    def q_temperature(self):
        return self.Q_([[1, 2], [3, 4]], self.ureg.degC)

    def assertNDArrayEqual(self, actual, desired):
        # Assert that the given arrays are equal, and are not Quantities
        np.testing.assert_array_equal(actual, desired)
        assert not isinstance(actual, self.Q_)
        assert not isinstance(desired, self.Q_)


class TestNumpyArrayCreation(TestNumpyMethods):
    # https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html

    @helpers.requires_array_function_protocol()
    def test_ones_like(self):
        self.assertNDArrayEqual(np.ones_like(self.q), np.array([[1, 1], [1, 1]]))

    @helpers.requires_array_function_protocol()
    def test_zeros_like(self):
        self.assertNDArrayEqual(np.zeros_like(self.q), np.array([[0, 0], [0, 0]]))

    @helpers.requires_array_function_protocol()
    def test_empty_like(self):
        ret = np.empty_like(self.q)
        assert ret.shape == (2, 2)
        assert isinstance(ret, np.ndarray)

    @helpers.requires_array_function_protocol()
    def test_full_like(self):
        helpers.assert_quantity_equal(
            np.full_like(self.q, self.Q_(0, self.ureg.degC)),
            self.Q_([[0, 0], [0, 0]], self.ureg.degC),
        )
        self.assertNDArrayEqual(np.full_like(self.q, 2), np.array([[2, 2], [2, 2]]))


class TestNumpyArrayManipulation(TestNumpyMethods):
    # TODO
    # https://www.numpy.org/devdocs/reference/routines.array-manipulation.html
    # copyto
    # broadcast , broadcast_arrays
    # asarray	asanyarray	asmatrix	asfarray	asfortranarray	ascontiguousarray	asarray_chkfinite	asscalar	require

    # Changing array shape

    def test_flatten(self):
        helpers.assert_quantity_equal(self.q.flatten(), [1, 2, 3, 4] * self.ureg.m)

    def test_flat(self):
        for q, v in zip(self.q.flat, [1, 2, 3, 4]):
            assert q == v * self.ureg.m

    def test_reshape(self):
        helpers.assert_quantity_equal(
            self.q.reshape([1, 4]), [[1, 2, 3, 4]] * self.ureg.m
        )

    def test_ravel(self):
        helpers.assert_quantity_equal(self.q.ravel(), [1, 2, 3, 4] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_ravel_numpy_func(self):
        helpers.assert_quantity_equal(np.ravel(self.q), [1, 2, 3, 4] * self.ureg.m)

    # Transpose-like operations

    @helpers.requires_array_function_protocol()
    def test_moveaxis(self):
        helpers.assert_quantity_equal(
            np.moveaxis(self.q, 1, 0), np.array([[1, 2], [3, 4]]).T * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_rollaxis(self):
        helpers.assert_quantity_equal(
            np.rollaxis(self.q, 1), np.array([[1, 2], [3, 4]]).T * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_swapaxes(self):
        helpers.assert_quantity_equal(
            np.swapaxes(self.q, 1, 0), np.array([[1, 2], [3, 4]]).T * self.ureg.m
        )

    def test_transpose(self):
        helpers.assert_quantity_equal(
            self.q.transpose(), [[1, 3], [2, 4]] * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_transpose_numpy_func(self):
        helpers.assert_quantity_equal(
            np.transpose(self.q), [[1, 3], [2, 4]] * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_flip_numpy_func(self):
        helpers.assert_quantity_equal(
            np.flip(self.q, axis=0), [[3, 4], [1, 2]] * self.ureg.m
        )

    # Changing number of dimensions

    @helpers.requires_array_function_protocol()
    def test_atleast_1d(self):
        actual = np.atleast_1d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (self.Q_(np.array([0]), self.ureg.degC), self.q.flatten())
        for ind_actual, ind_expected in zip(actual, expected):
            helpers.assert_quantity_equal(ind_actual, ind_expected)
        helpers.assert_quantity_equal(np.atleast_1d(self.q), self.q)

    @helpers.requires_array_function_protocol()
    def test_atleast_2d(self):
        actual = np.atleast_2d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (
            self.Q_(np.array([[0]]), self.ureg.degC),
            np.array([[1, 2, 3, 4]]) * self.ureg.m,
        )
        for ind_actual, ind_expected in zip(actual, expected):
            helpers.assert_quantity_equal(ind_actual, ind_expected)
        helpers.assert_quantity_equal(np.atleast_2d(self.q), self.q)

    @helpers.requires_array_function_protocol()
    def test_atleast_3d(self):
        actual = np.atleast_3d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (
            self.Q_(np.array([[[0]]]), self.ureg.degC),
            np.array([[[1], [2], [3], [4]]]) * self.ureg.m,
        )
        for ind_actual, ind_expected in zip(actual, expected):
            helpers.assert_quantity_equal(ind_actual, ind_expected)
        helpers.assert_quantity_equal(
            np.atleast_3d(self.q), np.array([[[1], [2]], [[3], [4]]]) * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_broadcast_to(self):
        helpers.assert_quantity_equal(
            np.broadcast_to(self.q[:, 1], (2, 2)),
            np.array([[2, 4], [2, 4]]) * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_expand_dims(self):
        helpers.assert_quantity_equal(
            np.expand_dims(self.q, 0), np.array([[[1, 2], [3, 4]]]) * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_squeeze(self):
        helpers.assert_quantity_equal(np.squeeze(self.q), self.q)
        helpers.assert_quantity_equal(
            self.q.reshape([1, 4]).squeeze(), [1, 2, 3, 4] * self.ureg.m
        )

    # Changing number of dimensions
    # Joining arrays
    @helpers.requires_array_function_protocol()
    def test_concat_stack(self, subtests):
        for func in (np.concatenate, np.stack, np.hstack, np.vstack, np.dstack):
            with subtests.test(func=func):
                helpers.assert_quantity_equal(
                    func([self.q] * 2), self.Q_(func([self.q.m] * 2), self.ureg.m)
                )
                # One or more of the args is a bare array full of zeros or NaNs
                helpers.assert_quantity_equal(
                    func([self.q_zero_or_nan.m, self.q]),
                    self.Q_(func([self.q_zero_or_nan.m, self.q.m]), self.ureg.m),
                )
                # One or more of the args is a bare array with at least one non-zero,
                # non-NaN element
                nz = self.q_zero_or_nan
                nz.m[0, 0] = 1
                with pytest.raises(DimensionalityError):
                    func([nz.m, self.q])

    @helpers.requires_array_function_protocol()
    def test_block_column_stack(self, subtests):
        for func in (np.block, np.column_stack):
            with subtests.test(func=func):

                helpers.assert_quantity_equal(
                    func([self.q[:, 0], self.q[:, 1]]),
                    self.Q_(func([self.q[:, 0].m, self.q[:, 1].m]), self.ureg.m),
                )

                # One or more of the args is a bare array full of zeros or NaNs
                helpers.assert_quantity_equal(
                    func(
                        [
                            self.q_zero_or_nan[:, 0].m,
                            self.q[:, 0],
                            self.q_zero_or_nan[:, 1].m,
                        ]
                    ),
                    self.Q_(
                        func(
                            [
                                self.q_zero_or_nan[:, 0].m,
                                self.q[:, 0].m,
                                self.q_zero_or_nan[:, 1].m,
                            ]
                        ),
                        self.ureg.m,
                    ),
                )
                # One or more of the args is a bare array with at least one non-zero,
                # non-NaN element
                nz = self.q_zero_or_nan
                nz.m[0, 0] = 1
                with pytest.raises(DimensionalityError):
                    func([nz[:, 0].m, self.q[:, 0]])

    @helpers.requires_array_function_protocol()
    def test_append(self):
        helpers.assert_quantity_equal(
            np.append(self.q, [[0, 0]] * self.ureg.m, axis=0),
            [[1, 2], [3, 4], [0, 0]] * self.ureg.m,
        )

    def test_astype(self):
        actual = self.q.astype(np.float32)
        expected = self.Q_(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "m")
        helpers.assert_quantity_equal(actual, expected)
        assert actual.m.dtype == expected.m.dtype

    def test_item(self):
        helpers.assert_quantity_equal(self.Q_([[0]], "m").item(), 0 * self.ureg.m)


class TestNumpyMathematicalFunctions(TestNumpyMethods):
    # https://www.numpy.org/devdocs/reference/routines.math.html
    # Trigonometric functions
    @helpers.requires_array_function_protocol()
    def test_unwrap(self):
        helpers.assert_quantity_equal(
            np.unwrap([0, 3 * np.pi] * self.ureg.radians), [0, np.pi]
        )
        helpers.assert_quantity_equal(
            np.unwrap([0, 540] * self.ureg.deg), [0, 180] * self.ureg.deg
        )

    # Rounding

    @helpers.requires_array_function_protocol()
    def test_fix(self):
        helpers.assert_quantity_equal(np.fix(3.14 * self.ureg.m), 3.0 * self.ureg.m)
        helpers.assert_quantity_equal(np.fix(3.0 * self.ureg.m), 3.0 * self.ureg.m)
        helpers.assert_quantity_equal(
            np.fix([2.1, 2.9, -2.1, -2.9] * self.ureg.m),
            [2.0, 2.0, -2.0, -2.0] * self.ureg.m,
        )

    # Sums, products, differences

    @helpers.requires_array_function_protocol()
    def test_prod(self):
        axis = 0
        where = [[True, False], [True, True]]

        helpers.assert_quantity_equal(self.q.prod(), 24 * self.ureg.m ** 4)
        helpers.assert_quantity_equal(self.q.prod(axis=axis), [3, 8] * self.ureg.m ** 2)
        helpers.assert_quantity_equal(self.q.prod(where=where), 12 * self.ureg.m ** 3)

    @helpers.requires_array_function_protocol()
    def test_prod_numpy_func(self):
        axis = 0
        where = [[True, False], [True, True]]

        helpers.assert_quantity_equal(np.prod(self.q), 24 * self.ureg.m ** 4)
        helpers.assert_quantity_equal(
            np.prod(self.q, axis=axis), [3, 8] * self.ureg.m ** 2
        )
        helpers.assert_quantity_equal(
            np.prod(self.q, where=where), 12 * self.ureg.m ** 3
        )

        with pytest.raises(DimensionalityError):
            np.prod(self.q, axis=axis, where=where)
        helpers.assert_quantity_equal(
            np.prod(self.q, axis=axis, where=[[True, False], [False, True]]),
            [1, 4] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.prod(self.q, axis=axis, where=[True, False]), [3, 1] * self.ureg.m ** 2
        )

    def test_sum(self):
        assert self.q.sum() == 10 * self.ureg.m
        helpers.assert_quantity_equal(self.q.sum(0), [4, 6] * self.ureg.m)
        helpers.assert_quantity_equal(self.q.sum(1), [3, 7] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_sum_numpy_func(self):
        helpers.assert_quantity_equal(np.sum(self.q, axis=0), [4, 6] * self.ureg.m)
        with pytest.raises(OffsetUnitCalculusError):
            np.sum(self.q_temperature)

    @helpers.requires_array_function_protocol()
    def test_nansum_numpy_func(self):
        helpers.assert_quantity_equal(
            np.nansum(self.q_nan, axis=0), [4, 2] * self.ureg.m
        )

    def test_cumprod(self):
        with pytest.raises(DimensionalityError):
            self.q.cumprod()
        helpers.assert_quantity_equal((self.q / self.ureg.m).cumprod(), [1, 2, 6, 24])

    @helpers.requires_array_function_protocol()
    def test_cumprod_numpy_func(self):
        with pytest.raises(DimensionalityError):
            np.cumprod(self.q)
        with pytest.raises(DimensionalityError):
            np.cumproduct(self.q)
        helpers.assert_quantity_equal(np.cumprod(self.q / self.ureg.m), [1, 2, 6, 24])
        helpers.assert_quantity_equal(
            np.cumproduct(self.q / self.ureg.m), [1, 2, 6, 24]
        )
        helpers.assert_quantity_equal(
            np.cumprod(self.q / self.ureg.m, axis=1), [[1, 2], [3, 12]]
        )

    @helpers.requires_array_function_protocol()
    def test_nancumprod_numpy_func(self):
        with pytest.raises(DimensionalityError):
            np.nancumprod(self.q_nan)
        helpers.assert_quantity_equal(
            np.nancumprod(self.q_nan / self.ureg.m), [1, 2, 6, 6]
        )

    @helpers.requires_array_function_protocol()
    def test_diff(self):
        helpers.assert_quantity_equal(np.diff(self.q, 1), [[1], [1]] * self.ureg.m)
        helpers.assert_quantity_equal(
            np.diff(self.q_temperature, 1), [[1], [1]] * self.ureg.delta_degC
        )

    @helpers.requires_array_function_protocol()
    def test_ediff1d(self):
        helpers.assert_quantity_equal(np.ediff1d(self.q), [1, 1, 1] * self.ureg.m)
        helpers.assert_quantity_equal(
            np.ediff1d(self.q_temperature), [1, 1, 1] * self.ureg.delta_degC
        )

    @helpers.requires_array_function_protocol()
    def test_gradient(self):
        grad = np.gradient([[1, 1], [3, 4]] * self.ureg.m, 1 * self.ureg.J)
        helpers.assert_quantity_equal(
            grad[0], [[2.0, 3.0], [2.0, 3.0]] * self.ureg.m / self.ureg.J
        )
        helpers.assert_quantity_equal(
            grad[1], [[0.0, 0.0], [1.0, 1.0]] * self.ureg.m / self.ureg.J
        )

        grad = np.gradient(self.Q_([[1, 1], [3, 4]], self.ureg.degC), 1 * self.ureg.J)
        helpers.assert_quantity_equal(
            grad[0], [[2.0, 3.0], [2.0, 3.0]] * self.ureg.delta_degC / self.ureg.J
        )
        helpers.assert_quantity_equal(
            grad[1], [[0.0, 0.0], [1.0, 1.0]] * self.ureg.delta_degC / self.ureg.J
        )

    @helpers.requires_array_function_protocol()
    def test_cross(self):
        a = [[3, -3, 1]] * self.ureg.kPa
        b = [[4, 9, 2]] * self.ureg.m ** 2
        helpers.assert_quantity_equal(
            np.cross(a, b), [[-15, -2, 39]] * self.ureg.kPa * self.ureg.m ** 2
        )

    @helpers.requires_array_function_protocol()
    def test_trapz(self):
        helpers.assert_quantity_equal(
            np.trapz([1.0, 2.0, 3.0, 4.0] * self.ureg.J, dx=1 * self.ureg.m),
            7.5 * self.ureg.J * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_dot(self):
        helpers.assert_quantity_equal(
            self.q.ravel().dot(np.array([1, 0, 0, 1])), 5 * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_dot_numpy_func(self):
        helpers.assert_quantity_equal(
            np.dot(self.q.ravel(), [0, 0, 1, 0] * self.ureg.dimensionless),
            3 * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_einsum(self):
        a = np.arange(25).reshape(5, 5) * self.ureg.m
        b = np.arange(5) * self.ureg.m
        helpers.assert_quantity_equal(np.einsum("ii", a), 60 * self.ureg.m)
        helpers.assert_quantity_equal(
            np.einsum("ii->i", a), np.array([0, 6, 12, 18, 24]) * self.ureg.m
        )
        helpers.assert_quantity_equal(np.einsum("i,i", b, b), 30 * self.ureg.m ** 2)
        helpers.assert_quantity_equal(
            np.einsum("ij,j", a, b),
            np.array([30, 80, 130, 180, 230]) * self.ureg.m ** 2,
        )

    @helpers.requires_array_function_protocol()
    def test_solve(self):
        A = self.q
        b = [[3], [7]] * self.ureg.s
        x = np.linalg.solve(A, b)

        helpers.assert_quantity_almost_equal(x, self.Q_([[1], [1]], "s / m"))

        helpers.assert_quantity_almost_equal(np.dot(A, x), b)

    # Arithmetic operations
    def test_addition_with_scalar(self):
        a = np.array([0, 1, 2])
        b = 10.0 * self.ureg("gram/kilogram")
        helpers.assert_quantity_almost_equal(
            a + b, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless)
        )
        helpers.assert_quantity_almost_equal(
            b + a, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless)
        )

    def test_addition_with_incompatible_scalar(self):
        a = np.array([0, 1, 2])
        b = 1.0 * self.ureg.m
        with pytest.raises(DimensionalityError):
            op.add(a, b)
        with pytest.raises(DimensionalityError):
            op.add(b, a)

    def test_power(self):
        arr = np.array(range(3), dtype=np.float)
        q = self.Q_(arr, "meter")

        for op_ in [op.pow, op.ipow, np.power]:
            q_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(2.0, q_cp)
            arr_cp = copy.copy(arr)
            arr_cp = copy.copy(arr)
            q_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(q_cp, arr_cp)
            q_cp = copy.copy(q)
            q2_cp = copy.copy(q)
            with pytest.raises(DimensionalityError):
                op_(q_cp, q2_cp)

        helpers.assert_quantity_equal(
            np.power(self.q, self.Q_(2)), self.Q_([[1, 4], [9, 16]], "m**2")
        )
        helpers.assert_quantity_equal(
            self.q ** self.Q_(2), self.Q_([[1, 4], [9, 16]], "m**2")
        )
        self.assertNDArrayEqual(arr ** self.Q_(2), np.array([0, 1, 4]))

    def test_sqrt(self):
        q = self.Q_(100, "m**2")
        helpers.assert_quantity_equal(np.sqrt(q), self.Q_(10, "m"))

    def test_cbrt(self):
        q = self.Q_(1000, "m**3")
        helpers.assert_quantity_equal(np.cbrt(q), self.Q_(10, "m"))

    @pytest.mark.xfail
    @helpers.requires_numpy
    def test_exponentiation_array_exp_2(self):
        arr = np.array(range(3), dtype=np.float)
        # q = self.Q_(copy.copy(arr), None)
        q = self.Q_(copy.copy(arr), "meter")
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        # this fails as expected since numpy 1.8.0 but...
        with pytest.raises(DimensionalityError):
            op.pow(arr_cp, q_cp)
        # ..not for op.ipow !
        # q_cp is treated as if it is an array. The units are ignored.
        # Quantity.__ipow__ is never called
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        with pytest.raises(DimensionalityError):
            op.ipow(arr_cp, q_cp)


class TestNumpyUnclassified(TestNumpyMethods):
    def test_tolist(self):
        with pytest.raises(AttributeError):
            (5 * self.ureg.m).tolist()

        assert self.q.tolist() == [
            [1 * self.ureg.m, 2 * self.ureg.m],
            [3 * self.ureg.m, 4 * self.ureg.m],
        ]

    def test_fill(self):
        tmp = self.q
        tmp.fill(6 * self.ureg.ft)
        helpers.assert_quantity_equal(tmp, [[6, 6], [6, 6]] * self.ureg.ft)
        tmp.fill(5 * self.ureg.m)
        helpers.assert_quantity_equal(tmp, [[5, 5], [5, 5]] * self.ureg.m)

    def test_take(self):
        helpers.assert_quantity_equal(self.q.take([0, 1, 2, 3]), self.q.flatten())

    def test_put(self):
        q = [1.0, 2.0, 3.0, 4.0] * self.ureg.m
        q.put([0, 2], [10.0, 20.0] * self.ureg.m)
        helpers.assert_quantity_equal(q, [10.0, 2.0, 20.0, 4.0] * self.ureg.m)

        q = [1.0, 2.0, 3.0, 4.0] * self.ureg.m
        q.put([0, 2], [1.0, 2.0] * self.ureg.mm)
        helpers.assert_quantity_equal(q, [0.001, 2.0, 0.002, 4.0] * self.ureg.m)

        q = [1.0, 2.0, 3.0, 4.0] * self.ureg.m / self.ureg.mm
        q.put([0, 2], [1.0, 2.0])
        helpers.assert_quantity_equal(
            q, [0.001, 2.0, 0.002, 4.0] * self.ureg.m / self.ureg.mm
        )

        q = [1.0, 2.0, 3.0, 4.0] * self.ureg.m
        with pytest.raises(DimensionalityError):
            q.put([0, 2], [4.0, 6.0] * self.ureg.J)
        with pytest.raises(DimensionalityError):
            q.put([0, 2], [4.0, 6.0])

    def test_repeat(self):
        helpers.assert_quantity_equal(
            self.q.repeat(2), [1, 1, 2, 2, 3, 3, 4, 4] * self.ureg.m
        )

    def test_sort(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        q.sort()
        helpers.assert_quantity_equal(q, [1, 2, 3, 4, 5, 6] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_sort_numpy_func(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        helpers.assert_quantity_equal(np.sort(q), [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort(self):
        q = [1, 4, 5, 6, 2, 9] * self.ureg.MeV
        self.assertNDArrayEqual(q.argsort(), [0, 4, 1, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_argsort_numpy_func(self):
        self.assertNDArrayEqual(np.argsort(self.q, axis=0), np.array([[0, 0], [1, 1]]))

    def test_diagonal(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        helpers.assert_quantity_equal(q.diagonal(offset=1), [2, 3] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_diagonal_numpy_func(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        helpers.assert_quantity_equal(np.diagonal(q, offset=-1), [1, 2] * self.ureg.m)

    def test_compress(self):
        helpers.assert_quantity_equal(
            self.q.compress([False, True], axis=0), [[3, 4]] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            self.q.compress([False, True], axis=1), [[2], [4]] * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_compress_nep18(self):
        helpers.assert_quantity_equal(
            np.compress([False, True], self.q, axis=1), [[2], [4]] * self.ureg.m
        )

    def test_searchsorted(self):
        q = self.q.flatten()
        self.assertNDArrayEqual(q.searchsorted([1.5, 2.5] * self.ureg.m), [1, 2])
        q = self.q.flatten()
        with pytest.raises(DimensionalityError):
            q.searchsorted([1.5, 2.5])

    @helpers.requires_array_function_protocol()
    def test_searchsorted_numpy_func(self):
        """Test searchsorted as numpy function."""
        q = self.q.flatten()
        self.assertNDArrayEqual(np.searchsorted(q, [1.5, 2.5] * self.ureg.m), [1, 2])

    def test_nonzero(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(q.nonzero()[0], [0, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(np.nonzero(q)[0], [0, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_any_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert np.any(q)
        with pytest.raises(ValueError):
            np.any(self.q_temperature)

    @helpers.requires_array_function_protocol()
    def test_all_numpy_func(self):
        q = [0, 1] * self.ureg.m
        assert not np.all(q)
        with pytest.raises(ValueError):
            np.all(self.q_temperature)

    @helpers.requires_array_function_protocol()
    def test_count_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        assert np.count_nonzero(q) == 4

    def test_max(self):
        assert self.q.max() == 4 * self.ureg.m

    def test_max_numpy_func(self):
        assert np.max(self.q) == 4 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_max_with_axis_arg(self):
        helpers.assert_quantity_equal(np.max(self.q, axis=1), [2, 4] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_max_with_initial_arg(self):
        helpers.assert_quantity_equal(
            np.max(self.q[..., None], axis=2, initial=3 * self.ureg.m),
            [[3, 3], [3, 4]] * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_nanmax(self):
        assert np.nanmax(self.q_nan) == 3 * self.ureg.m

    def test_argmax(self):
        assert self.q.argmax() == 3

    @helpers.requires_array_function_protocol()
    def test_argmax_numpy_func(self):
        self.assertNDArrayEqual(np.argmax(self.q, axis=0), np.array([1, 1]))

    @helpers.requires_array_function_protocol()
    def test_nanargmax_numpy_func(self):
        self.assertNDArrayEqual(np.nanargmax(self.q_nan, axis=0), np.array([1, 0]))

    def test_maximum(self):
        helpers.assert_quantity_equal(
            np.maximum(self.q, self.Q_([0, 5], "m")), self.Q_([[1, 5], [3, 5]], "m")
        )

    def test_min(self):
        assert self.q.min() == 1 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_min_numpy_func(self):
        assert np.min(self.q) == 1 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_min_with_axis_arg(self):
        helpers.assert_quantity_equal(np.min(self.q, axis=1), [1, 3] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_min_with_initial_arg(self):
        helpers.assert_quantity_equal(
            np.min(self.q[..., None], axis=2, initial=3 * self.ureg.m),
            [[1, 2], [3, 3]] * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_nanmin(self):
        assert np.nanmin(self.q_nan) == 1 * self.ureg.m

    def test_argmin(self):
        assert self.q.argmin() == 0

    @helpers.requires_array_function_protocol()
    def test_argmin_numpy_func(self):
        self.assertNDArrayEqual(np.argmin(self.q, axis=0), np.array([0, 0]))

    @helpers.requires_array_function_protocol()
    def test_nanargmin_numpy_func(self):
        self.assertNDArrayEqual(np.nanargmin(self.q_nan, axis=0), np.array([0, 0]))

    def test_minimum(self):
        helpers.assert_quantity_equal(
            np.minimum(self.q, self.Q_([0, 5], "m")), self.Q_([[0, 2], [0, 4]], "m")
        )

    def test_ptp(self):
        assert self.q.ptp() == 3 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_ptp_numpy_func(self):
        helpers.assert_quantity_equal(np.ptp(self.q, axis=0), [2, 2] * self.ureg.m)

    def test_clip(self):
        helpers.assert_quantity_equal(
            self.q.clip(max=2 * self.ureg.m), [[1, 2], [2, 2]] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            self.q.clip(min=3 * self.ureg.m), [[3, 3], [3, 4]] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            self.q.clip(min=2 * self.ureg.m, max=3 * self.ureg.m),
            [[2, 2], [3, 3]] * self.ureg.m,
        )
        with pytest.raises(DimensionalityError):
            self.q.clip(self.ureg.J)
        with pytest.raises(DimensionalityError):
            self.q.clip(1)

    @helpers.requires_array_function_protocol()
    def test_clip_numpy_func(self):
        helpers.assert_quantity_equal(
            np.clip(self.q, 150 * self.ureg.cm, None), [[1.5, 2], [3, 4]] * self.ureg.m
        )

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * self.ureg.m
        helpers.assert_quantity_equal(q.round(0), [1, 1, 6, 22] * self.ureg.m)
        helpers.assert_quantity_equal(q.round(-1), [0, 0, 10, 20] * self.ureg.m)
        helpers.assert_quantity_equal(q.round(1), [1, 1.3, 5.7, 22] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_round_numpy_func(self):
        helpers.assert_quantity_equal(
            np.around(1.0275 * self.ureg.m, decimals=2), 1.03 * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.round_(1.0275 * self.ureg.m, decimals=2), 1.03 * self.ureg.m
        )

    def test_trace(self):
        assert self.q.trace() == (1 + 4) * self.ureg.m

    def test_cumsum(self):
        helpers.assert_quantity_equal(self.q.cumsum(), [1, 3, 6, 10] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_cumsum_numpy_func(self):
        helpers.assert_quantity_equal(
            np.cumsum(self.q, axis=0), [[1, 2], [4, 6]] * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_nancumsum_numpy_func(self):
        helpers.assert_quantity_equal(
            np.nancumsum(self.q_nan, axis=0), [[1, 2], [4, 2]] * self.ureg.m
        )

    def test_mean(self):
        assert self.q.mean() == 2.5 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_mean_numpy_func(self):
        assert np.mean(self.q) == 2.5 * self.ureg.m
        assert np.mean(self.q_temperature) == self.Q_(2.5, self.ureg.degC)

    @helpers.requires_array_function_protocol()
    def test_nanmean_numpy_func(self):
        assert np.nanmean(self.q_nan) == 2 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_average_numpy_func(self):
        helpers.assert_quantity_almost_equal(
            np.average(self.q, axis=0, weights=[1, 2]),
            [2.33333, 3.33333] * self.ureg.m,
            rtol=1e-5,
        )

    @helpers.requires_array_function_protocol()
    def test_median_numpy_func(self):
        assert np.median(self.q) == 2.5 * self.ureg.m

    @helpers.requires_array_function_protocol()
    def test_nanmedian_numpy_func(self):
        assert np.nanmedian(self.q_nan) == 2 * self.ureg.m

    def test_var(self):
        assert self.q.var() == 1.25 * self.ureg.m ** 2

    @helpers.requires_array_function_protocol()
    def test_var_numpy_func(self):
        assert np.var(self.q) == 1.25 * self.ureg.m ** 2

    @helpers.requires_array_function_protocol()
    def test_nanvar_numpy_func(self):
        helpers.assert_quantity_almost_equal(
            np.nanvar(self.q_nan), 0.66667 * self.ureg.m ** 2, rtol=1e-5
        )

    def test_std(self):
        helpers.assert_quantity_almost_equal(
            self.q.std(), 1.11803 * self.ureg.m, rtol=1e-5
        )

    @helpers.requires_array_function_protocol()
    def test_std_numpy_func(self):
        helpers.assert_quantity_almost_equal(
            np.std(self.q), 1.11803 * self.ureg.m, rtol=1e-5
        )
        with pytest.raises(OffsetUnitCalculusError):
            np.std(self.q_temperature)

    def test_cumprod(self):
        with pytest.raises(DimensionalityError):
            self.q.cumprod()
        helpers.assert_quantity_equal((self.q / self.ureg.m).cumprod(), [1, 2, 6, 24])

    @helpers.requires_array_function_protocol()
    def test_nanstd_numpy_func(self):
        helpers.assert_quantity_almost_equal(
            np.nanstd(self.q_nan), 0.81650 * self.ureg.m, rtol=1e-5
        )

    def test_conj(self):
        helpers.assert_quantity_equal((self.q * (1 + 1j)).conj(), self.q * (1 - 1j))
        helpers.assert_quantity_equal(
            (self.q * (1 + 1j)).conjugate(), self.q * (1 - 1j)
        )

    def test_getitem(self):
        with pytest.raises(IndexError):
            self.q.__getitem__((0, 10))
        helpers.assert_quantity_equal(self.q[0], [1, 2] * self.ureg.m)
        assert self.q[1, 1] == 4 * self.ureg.m

    def test_setitem(self):
        with pytest.raises(TypeError):
            self.q[0, 0] = 1
        with pytest.raises(DimensionalityError):
            self.q[0, 0] = 1 * self.ureg.J
        with pytest.raises(DimensionalityError):
            self.q[0] = 1
        with pytest.raises(DimensionalityError):
            self.q[0] = np.ndarray([1, 2])
        with pytest.raises(DimensionalityError):
            self.q[0] = 1 * self.ureg.J

        q = self.q.copy()
        q[0] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [3, 4]] * self.ureg.m)

        q = self.q.copy()
        q[...] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [1, 1]] * self.ureg.m)

        q = self.q.copy()
        q[:] = 1 * self.ureg.m
        helpers.assert_quantity_equal(q, [[1, 1], [1, 1]] * self.ureg.m)

        # check and see that dimensionless num  bers work correctly
        q = [0, 1, 2, 3] * self.ureg.dimensionless
        q[0] = 1
        helpers.assert_quantity_equal(q, np.asarray([1, 1, 2, 3]))
        q[0] = self.ureg.m / self.ureg.mm
        helpers.assert_quantity_equal(q, np.asarray([1000, 1, 2, 3]))

        q = [0.0, 1.0, 2.0, 3.0] * self.ureg.m / self.ureg.mm
        q[0] = 1.0
        helpers.assert_quantity_equal(q, [0.001, 1, 2, 3] * self.ureg.m / self.ureg.mm)

        # Check that this properly masks the first item without warning
        q = self.ureg.Quantity(
            np.ma.array([0.0, 1.0, 2.0, 3.0], mask=[False, True, False, False]), "m"
        )
        with warnings.catch_warnings(record=True) as w:
            q[0] = np.ma.masked
            # Check for no warnings
            assert not w
            assert q.mask[0]

    def test_iterator(self):
        for q, v in zip(self.q.flatten(), [1, 2, 3, 4]):
            assert q == v * self.ureg.m

    def test_iterable(self):
        assert np.iterable(self.q)
        assert not np.iterable(1 * self.ureg.m)

    def test_reversible_op(self):
        """ """
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))
        helpers.assert_quantity_equal(x / self.q, u * x / self.q)
        helpers.assert_quantity_equal(x * self.q, u * x * self.q)
        helpers.assert_quantity_equal(x + u, u + x)
        helpers.assert_quantity_equal(x - u, -(u - x))

    def test_pickle(self, subtests):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with subtests.test(protocol):
                q1 = [10, 20] * self.ureg.m
                q2 = pickle.loads(pickle.dumps(q1, protocol))
                self.assertNDArrayEqual(q1.magnitude, q2.magnitude)
                assert q1.units == q2.units

    def test_equal(self):
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))
        true = np.ones_like(x, dtype=np.bool_)
        false = np.zeros_like(x, dtype=np.bool_)

        helpers.assert_quantity_equal(u, u)
        helpers.assert_quantity_equal(u == u, u.magnitude == u.magnitude)
        helpers.assert_quantity_equal(u == 1, u.magnitude == 1)

        v = self.Q_(np.zeros(x.shape), "m")
        w = self.Q_(np.ones(x.shape), "m")
        self.assertNDArrayEqual(v == 1, false)
        self.assertNDArrayEqual(
            self.Q_(np.zeros_like(x), "m") == self.Q_(np.zeros_like(x), "s"),
            false,
        )
        self.assertNDArrayEqual(v == v, true)
        self.assertNDArrayEqual(v == w, false)
        self.assertNDArrayEqual(v == w.to("mm"), false)
        self.assertNDArrayEqual(u == v, false)

    def test_shape(self):
        u = self.Q_(np.arange(12))
        u.shape = 4, 3
        assert u.magnitude.shape == (4, 3)

    @helpers.requires_array_function_protocol()
    def test_shape_numpy_func(self):
        assert np.shape(self.q) == (2, 2)

    @helpers.requires_array_function_protocol()
    def test_alen_numpy_func(self):
        assert np.alen(self.q) == 2

    @helpers.requires_array_function_protocol()
    def test_ndim_numpy_func(self):
        assert np.ndim(self.q) == 2

    @helpers.requires_array_function_protocol()
    def test_copy_numpy_func(self):
        q_copy = np.copy(self.q)
        helpers.assert_quantity_equal(self.q, q_copy)
        assert self.q is not q_copy

    @helpers.requires_array_function_protocol()
    def test_trim_zeros_numpy_func(self):
        q = [0, 4, 3, 0, 2, 2, 0, 0, 0] * self.ureg.m
        helpers.assert_quantity_equal(np.trim_zeros(q), [4, 3, 0, 2, 2] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_result_type_numpy_func(self):
        assert np.result_type(self.q) == np.dtype("int")

    @helpers.requires_array_function_protocol()
    def test_nan_to_num_numpy_func(self):
        helpers.assert_quantity_equal(
            np.nan_to_num(self.q_nan, nan=-999 * self.ureg.mm),
            [[1, 2], [3, -0.999]] * self.ureg.m,
        )

    @helpers.requires_array_function_protocol()
    def test_meshgrid_numpy_func(self):
        x = [1, 2] * self.ureg.m
        y = [0, 50, 100] * self.ureg.mm
        xx, yy = np.meshgrid(x, y)
        helpers.assert_quantity_equal(xx, [[1, 2], [1, 2], [1, 2]] * self.ureg.m)
        helpers.assert_quantity_equal(yy, [[0, 0], [50, 50], [100, 100]] * self.ureg.mm)

    @helpers.requires_array_function_protocol()
    def test_isclose_numpy_func(self):
        q2 = [[1000.05, 2000], [3000.00007, 4001]] * self.ureg.mm
        self.assertNDArrayEqual(
            np.isclose(self.q, q2), np.array([[False, True], [True, False]])
        )
        self.assertNDArrayEqual(
            np.isclose(self.q, q2, atol=1e-5, rtol=1e-7),
            np.array([[False, True], [True, False]]),
        )

    @helpers.requires_array_function_protocol()
    def test_interp_numpy_func(self):
        x = [1, 4] * self.ureg.m
        xp = np.linspace(0, 3, 5) * self.ureg.m
        fp = self.Q_([0, 5, 10, 15, 20], self.ureg.degC)
        helpers.assert_quantity_almost_equal(
            np.interp(x, xp, fp), self.Q_([6.66667, 20.0], self.ureg.degC), rtol=1e-5
        )

        x_ = np.array([1, 4])
        xp_ = np.linspace(0, 3, 5)
        fp_ = [0, 5, 10, 15, 20]

        helpers.assert_quantity_almost_equal(
            np.interp(x_, xp_, fp), self.Q_([6.6667, 20.0], self.ureg.degC), rtol=1e-5
        )
        helpers.assert_quantity_almost_equal(
            np.interp(x, xp, fp_), [6.6667, 20.0], rtol=1e-5
        )

    def test_comparisons(self):
        self.assertNDArrayEqual(
            self.q > 2 * self.ureg.m, np.array([[False, False], [True, True]])
        )
        self.assertNDArrayEqual(
            self.q < 2 * self.ureg.m, np.array([[True, False], [False, False]])
        )

    @helpers.requires_array_function_protocol()
    def test_where(self):
        helpers.assert_quantity_equal(
            np.where(self.q >= 2 * self.ureg.m, self.q, 20 * self.ureg.m),
            [[20, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 2 * self.ureg.m, self.q, 0),
            [[0, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 2 * self.ureg.m, self.q, np.nan),
            [[np.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 3 * self.ureg.m, 0, self.q),
            [[1, 2], [0, 0]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 3 * self.ureg.m, np.nan, self.q),
            [[1, 2], [np.nan, np.nan]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 2 * self.ureg.m, self.q, np.array(np.nan)),
            [[np.nan, 2], [3, 4]] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.where(self.q >= 3 * self.ureg.m, np.array(np.nan), self.q),
            [[1, 2], [np.nan, np.nan]] * self.ureg.m,
        )
        with pytest.raises(DimensionalityError):
            np.where(
                self.q < 2 * self.ureg.m,
                self.q,
                0 * self.ureg.J,
            )

    @helpers.requires_array_function_protocol()
    def test_fabs(self):
        helpers.assert_quantity_equal(
            np.fabs(self.q - 2 * self.ureg.m), self.Q_([[1, 0], [1, 2]], "m")
        )

    @helpers.requires_array_function_protocol()
    def test_isin(self):
        self.assertNDArrayEqual(
            np.isin(self.q, self.Q_([0, 2, 4], "m")),
            np.array([[False, True], [False, True]]),
        )
        self.assertNDArrayEqual(
            np.isin(self.q, self.Q_([0, 2, 4], "J")),
            np.array([[False, False], [False, False]]),
        )
        self.assertNDArrayEqual(
            np.isin(self.q, [self.Q_(2, "m"), self.Q_(4, "J")]),
            np.array([[False, True], [False, False]]),
        )
        self.assertNDArrayEqual(
            np.isin(self.q, self.q.m), np.array([[False, False], [False, False]])
        )
        self.assertNDArrayEqual(
            np.isin(self.q / self.ureg.cm, [1, 3]),
            np.array([[True, False], [True, False]]),
        )
        with pytest.raises(ValueError):
            np.isin(self.q.m, self.q)

    @helpers.requires_array_function_protocol()
    def test_percentile(self):
        helpers.assert_quantity_equal(np.percentile(self.q, 25), self.Q_(1.75, "m"))

    @helpers.requires_array_function_protocol()
    def test_nanpercentile(self):
        helpers.assert_quantity_equal(
            np.nanpercentile(self.q_nan, 25), self.Q_(1.5, "m")
        )

    @helpers.requires_array_function_protocol()
    def test_quantile(self):
        helpers.assert_quantity_equal(np.quantile(self.q, 0.25), self.Q_(1.75, "m"))

    @helpers.requires_array_function_protocol()
    def test_nanquantile(self):
        helpers.assert_quantity_equal(
            np.nanquantile(self.q_nan, 0.25), self.Q_(1.5, "m")
        )

    @helpers.requires_array_function_protocol()
    def test_copyto(self):
        a = self.q.m
        q = copy.copy(self.q)
        np.copyto(q, 2 * q, where=[True, False])
        helpers.assert_quantity_equal(q, self.Q_([[2, 2], [6, 4]], "m"))
        np.copyto(q, 0, where=[[False, False], [True, False]])
        helpers.assert_quantity_equal(q, self.Q_([[2, 2], [0, 4]], "m"))
        np.copyto(a, q)
        self.assertNDArrayEqual(a, np.array([[2, 2], [0, 4]]))

    @helpers.requires_array_function_protocol()
    def test_tile(self):
        helpers.assert_quantity_equal(
            np.tile(self.q, 2), np.array([[1, 2, 1, 2], [3, 4, 3, 4]]) * self.ureg.m
        )

    @helpers.requires_numpy_at_least("1.20")
    @helpers.requires_array_function_protocol()
    def test_sliding_window_view(self):
        q = self.Q_([[1, 2, 2, 1], [2, 1, 1, 2], [1, 2, 2, 1]], "m")
        actual = np.lib.stride_tricks.sliding_window_view(q, window_shape=(3, 3))
        expected = self.Q_(
            [[[[1, 2, 2], [2, 1, 1], [1, 2, 2]], [[2, 2, 1], [1, 1, 2], [2, 2, 1]]]],
            "m",
        )
        helpers.assert_quantity_equal(actual, expected)

    @helpers.requires_array_function_protocol()
    def test_rot90(self):
        helpers.assert_quantity_equal(
            np.rot90(self.q), np.array([[2, 4], [1, 3]]) * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_insert(self):
        helpers.assert_quantity_equal(
            np.insert(self.q, 1, 0 * self.ureg.m, axis=1),
            np.array([[1, 0, 2], [3, 0, 4]]) * self.ureg.m,
        )

    def test_ndarray_downcast(self):
        with pytest.warns(UnitStrippedWarning):
            np.asarray(self.q)

    def test_ndarray_downcast_with_dtype(self):
        with pytest.warns(UnitStrippedWarning):
            qarr = np.asarray(self.q, dtype=np.float64)
            assert qarr.dtype == np.float64

    def test_array_protocol_unavailable(self):
        for attr in ("__array_struct__", "__array_interface__"):
            with pytest.raises(AttributeError):
                getattr(self.q, attr)

    @helpers.requires_array_function_protocol()
    def test_resize(self):
        helpers.assert_quantity_equal(
            np.resize(self.q, (2, 4)), [[1, 2, 3, 4], [1, 2, 3, 4]] * self.ureg.m
        )

    @helpers.requires_array_function_protocol()
    def test_pad(self):
        # Tests reproduced with modification from NumPy documentation
        a = [1, 2, 3, 4, 5] * self.ureg.m
        b = self.Q_([4.0, 6.0, 8.0, 9.0, -3.0], "degC")

        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "constant"), [0, 0, 1, 2, 3, 4, 5, 0, 0, 0] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "constant", constant_values=(0, 600 * self.ureg.cm)),
            [0, 0, 1, 2, 3, 4, 5, 6, 6, 6] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(
                b, (2, 1), "constant", constant_values=(np.nan, self.Q_(10, "degC"))
            ),
            self.Q_([np.nan, np.nan, 4, 6, 8, 9, -3, 10], "degC"),
        )
        with pytest.raises(DimensionalityError):
            np.pad(a, (2, 3), "constant", constant_values=4)
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "edge"), [1, 1, 1, 2, 3, 4, 5, 5, 5, 5] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "linear_ramp"),
            [0, 0, 1, 2, 3, 4, 5, 3, 1, 0] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "linear_ramp", end_values=(5, -4) * self.ureg.m),
            [5, 3, 1, 2, 3, 4, 5, 2, -1, -4] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2,), "maximum"), [5, 5, 1, 2, 3, 4, 5, 5, 5] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2,), "mean"), [3, 3, 1, 2, 3, 4, 5, 3, 3] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2,), "median"), [3, 3, 1, 2, 3, 4, 5, 3, 3] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(self.q, ((3, 2), (2, 3)), "minimum"),
            [
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [3, 3, 3, 4, 3, 3, 3],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
            ]
            * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "reflect"), [3, 2, 1, 2, 3, 4, 5, 4, 3, 2] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "reflect", reflect_type="odd"),
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "symmetric"), [2, 1, 1, 2, 3, 4, 5, 5, 4, 3] * self.ureg.m
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "symmetric", reflect_type="odd"),
            [0, 1, 1, 2, 3, 4, 5, 5, 6, 7] * self.ureg.m,
        )
        helpers.assert_quantity_equal(
            np.pad(a, (2, 3), "wrap"), [4, 5, 1, 2, 3, 4, 5, 1, 2, 3] * self.ureg.m
        )

        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get("padder", 10)
            vector[: pad_width[0]] = pad_value
            vector[-pad_width[1] :] = pad_value

        b = self.Q_(np.arange(6).reshape((2, 3)), "degC")
        helpers.assert_quantity_equal(
            np.pad(b, 2, pad_with),
            self.Q_(
                [
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 0, 1, 2, 10, 10],
                    [10, 10, 3, 4, 5, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                    [10, 10, 10, 10, 10, 10, 10],
                ],
                "degC",
            ),
        )
        helpers.assert_quantity_equal(
            np.pad(b, 2, pad_with, padder=100),
            self.Q_(
                [
                    [100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 0, 1, 2, 100, 100],
                    [100, 100, 3, 4, 5, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100],
                    [100, 100, 100, 100, 100, 100, 100],
                ],
                "degC",
            ),
        )  # Note: Does not support Quantity pad_with vectorized callable use

    @helpers.requires_array_function_protocol()
    def test_allclose(self):
        assert np.allclose([1e10, 1e-8] * self.ureg.m, [1.00001e10, 1e-9] * self.ureg.m)
        assert not np.allclose(
            [1e10, 1e-8] * self.ureg.m, [1.00001e10, 1e-9] * self.ureg.mm
        )

    @helpers.requires_array_function_protocol()
    def test_intersect1d(self):
        helpers.assert_quantity_equal(
            np.intersect1d([1, 3, 4, 3] * self.ureg.m, [3, 1, 2, 1] * self.ureg.m),
            [1, 3] * self.ureg.m,
        )


@pytest.mark.skip
class TestBitTwiddlingUfuncs(TestUFuncs):
    """Universal functions (ufuncs) >  Bittwiddling functions

    http://docs.scipy.org/doc/numpy/reference/ufuncs.html#bittwiddlingfunctions

    bitwise_and(x1, x2[, out])         Compute the bitwise AND of two arrays elementwise.
    bitwise_or(x1, x2[, out])  Compute the bitwise OR of two arrays elementwise.
    bitwise_xor(x1, x2[, out])         Compute the bitwise XOR of two arrays elementwise.
    invert(x[, out])   Compute bitwise inversion, or bitwise NOT, elementwise.
    left_shift(x1, x2[, out])  Shift the bits of an integer to the left.
    right_shift(x1, x2[, out])         Shift the bits of an integer to the right.

    Parameters
    ----------

    Returns
    -------

    """

    @property
    def qless(self):
        return np.asarray([1, 2, 3, 4], dtype=np.uint8) * self.ureg.dimensionless

    @property
    def qs(self):
        return 8 * self.ureg.J

    @property
    def q1(self):
        return np.asarray([1, 2, 3, 4], dtype=np.uint8) * self.ureg.J

    @property
    def q2(self):
        return 2 * self.q1

    @property
    def qm(self):
        return np.asarray([1, 2, 3, 4], dtype=np.uint8) * self.ureg.m

    def test_bitwise_and(self):
        self._test2(np.bitwise_and, self.q1, (self.q2, self.qs), (self.qm,), "same")

    def test_bitwise_or(self):
        self._test2(
            np.bitwise_or, self.q1, (self.q1, self.q2, self.qs), (self.qm,), "same"
        )

    def test_bitwise_xor(self):
        self._test2(
            np.bitwise_xor, self.q1, (self.q1, self.q2, self.qs), (self.qm,), "same"
        )

    def test_invert(self):
        self._test1(np.invert, (self.q1, self.q2, self.qs), (), "same")

    def test_left_shift(self):
        self._test2(
            np.left_shift, self.q1, (self.qless, 2), (self.q1, self.q2, self.qs), "same"
        )

    def test_right_shift(self):
        self._test2(
            np.right_shift,
            self.q1,
            (self.qless, 2),
            (self.q1, self.q2, self.qs),
            "same",
        )
