# -*- coding: utf-8 -*-

import copy
import operator as op
import unittest

from pint import DimensionalityError, OffsetUnitCalculusError, set_application_registry
from pint.compat import np
from pint.testsuite import QuantityTestCase, helpers
from pint.testsuite.test_umath import TestUFuncs


@helpers.requires_numpy()
class TestNumpyMethods(QuantityTestCase):

    FORCE_NDARRAY = True

    @classmethod
    def setUpClass(cls):
        from pint import _DEFAULT_REGISTRY
        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity

    @property
    def q(self):
        return [[1,2],[3,4]] * self.ureg.m
    @property
    def q_nan(self):
        return [[1,2],[3,np.nan]] * self.ureg.m
    @property
    def q_temperature(self):
        return self.Q_([[1,2],[3,4]], self.ureg.degC)

    def assertNDArrayEqual(self, actual, desired):
        # Assert that the given arrays are equal, and are not Quantities
        np.testing.assert_array_equal(actual, desired)
        self.assertFalse(isinstance(actual, self.Q_))
        self.assertFalse(isinstance(desired, self.Q_))


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
        self.assertEqual(ret.shape, (2, 2))
        self.assertTrue(isinstance(ret, np.ndarray))

    @helpers.requires_array_function_protocol()
    def test_full_like(self):
        self.assertQuantityEqual(np.full_like(self.q, self.Q_(0, self.ureg.degC)),
                                 self.Q_([[0, 0], [0, 0]], self.ureg.degC))
        self.assertNDArrayEqual(np.full_like(self.q, 2), np.array([[2, 2], [2, 2]]))

class TestNumpyArrayManipulation(TestNumpyMethods):
    #TODO
    # https://www.numpy.org/devdocs/reference/routines.array-manipulation.html
    # copyto
    # broadcast , broadcast_arrays
    # asarray	asanyarray	asmatrix	asfarray	asfortranarray	ascontiguousarray	asarray_chkfinite	asscalar	require
    
    # Changing array shape
    
    def test_flatten(self):
        self.assertQuantityEqual(self.q.flatten(), [1, 2, 3, 4] * self.ureg.m)

    def test_flat(self):
        for q, v in zip(self.q.flat, [1, 2, 3, 4]):
            self.assertEqual(q, v * self.ureg.m)

    def test_reshape(self):
        self.assertQuantityEqual(self.q.reshape([1,4]), [[1, 2, 3, 4]] * self.ureg.m)
    
    def test_ravel(self):
        self.assertQuantityEqual(self.q.ravel(), [1, 2, 3, 4] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_ravel_numpy_func(self):
        self.assertQuantityEqual(np.ravel(self.q), [1, 2, 3, 4] * self.ureg.m)

    # Transpose-like operations

    @helpers.requires_array_function_protocol()
    def test_moveaxis(self):
        self.assertQuantityEqual(np.moveaxis(self.q, 1,0), np.array([[1,2],[3,4]]).T * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_rollaxis(self):
        self.assertQuantityEqual(np.rollaxis(self.q, 1), np.array([[1,2],[3,4]]).T * self.ureg.m)
    
    @helpers.requires_array_function_protocol()
    def test_swapaxes(self):
        self.assertQuantityEqual(np.swapaxes(self.q, 1,0), np.array([[1,2],[3,4]]).T * self.ureg.m)

    def test_transpose(self):
        self.assertQuantityEqual(self.q.transpose(), [[1, 3], [2, 4]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_transpose_numpy_func(self):
        self.assertQuantityEqual(np.transpose(self.q), [[1, 3], [2, 4]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_flip_numpy_func(self):
        self.assertQuantityEqual(np.flip(self.q, axis=0), [[3, 4], [1, 2]] * self.ureg.m)
    
    # Changing number of dimensions
    
    @helpers.requires_array_function_protocol()
    def test_atleast_1d(self):
        actual = np.atleast_1d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (self.Q_(np.array([0]), self.ureg.degC), self.q.flatten())
        for ind_actual, ind_expected in zip(actual, expected):
            self.assertQuantityEqual(ind_actual, ind_expected)
        self.assertQuantityEqual(np.atleast_1d(self.q), self.q)

    @helpers.requires_array_function_protocol()
    def test_atleast_2d(self):
        actual = np.atleast_2d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (self.Q_(np.array([[0]]), self.ureg.degC), np.array([[1, 2, 3, 4]]) * self.ureg.m)
        for ind_actual, ind_expected in zip(actual, expected):
            self.assertQuantityEqual(ind_actual, ind_expected)
        self.assertQuantityEqual(np.atleast_2d(self.q), self.q)

    @helpers.requires_array_function_protocol()
    def test_atleast_3d(self):
        actual = np.atleast_3d(self.Q_(0, self.ureg.degC), self.q.flatten())
        expected = (self.Q_(np.array([[[0]]]), self.ureg.degC), np.array([[[1], [2], [3], [4]]]) * self.ureg.m)
        for ind_actual, ind_expected in zip(actual, expected):
            self.assertQuantityEqual(ind_actual, ind_expected)
        self.assertQuantityEqual(np.atleast_3d(self.q), np.array([[[1],[2]],[[3],[4]]])* self.ureg.m)
    
    @helpers.requires_array_function_protocol()
    def test_broadcast_to(self):
        self.assertQuantityEqual(np.broadcast_to(self.q[:,1], (2,2)), np.array([[2,4],[2,4]]) * self.ureg.m)
    
    @helpers.requires_array_function_protocol()    
    def test_expand_dims(self):
        self.assertQuantityEqual(np.expand_dims(self.q, 0), np.array([[[1, 2],[3, 4]]])* self.ureg.m)
    
    @helpers.requires_array_function_protocol()    
    def test_squeeze(self):
        self.assertQuantityEqual(np.squeeze(self.q), self.q)
        self.assertQuantityEqual(
            self.q.reshape([1,4]).squeeze(),
            [1, 2, 3, 4] * self.ureg.m
        )
        
    # Changing number of dimensions
    # Joining arrays
    @helpers.requires_array_function_protocol()
    def test_concatentate(self):
        self.assertQuantityEqual(
            np.concatenate([self.q]*2),
            self.Q_(np.concatenate([self.q.m]*2), self.ureg.m)
        )
    
    @helpers.requires_array_function_protocol()    
    def test_stack(self):
        self.assertQuantityEqual(
            np.stack([self.q]*2),
            self.Q_(np.stack([self.q.m]*2), self.ureg.m)
        )
    
    @helpers.requires_array_function_protocol()    
    def test_column_stack(self):
        self.assertQuantityEqual(
            np.column_stack([self.q[:,0],self.q[:,1]]),
            self.q
        )
    
    @helpers.requires_array_function_protocol()    
    def test_dstack(self):
        self.assertQuantityEqual(
            np.dstack([self.q]*2),
            self.Q_(np.dstack([self.q.m]*2), self.ureg.m)
        )
    
    @helpers.requires_array_function_protocol()    
    def test_hstack(self):
        self.assertQuantityEqual(
            np.hstack([self.q]*2),
            self.Q_(np.hstack([self.q.m]*2), self.ureg.m)
        )

    @helpers.requires_array_function_protocol()
    def test_vstack(self):
        self.assertQuantityEqual(
            np.vstack([self.q]*2),
            self.Q_(np.vstack([self.q.m]*2), self.ureg.m)
        )

    @helpers.requires_array_function_protocol()
    def test_block(self):
        self.assertQuantityEqual(
            np.block([self.q[0,:],self.q[1,:]]),
            self.Q_([1,2,3,4], self.ureg.m)
        )

    @helpers.requires_array_function_protocol()
    def test_append(self):
        self.assertQuantityEqual(np.append(self.q, [[0, 0]] * self.ureg.m, axis=0),
                                 [[1, 2], [3, 4], [0, 0]] * self.ureg.m)

    def test_astype(self):
        actual = self.q.astype(np.float32)
        expected = self.Q_(np.array([[1., 2.], [3., 4.]], dtype=np.float32), 'm')
        self.assertQuantityEqual(actual, expected)
        self.assertEqual(actual.m.dtype, expected.m.dtype)

    def test_item(self):
        self.assertQuantityEqual(self.Q_([[0]], 'm').item(), 0 * self.ureg.m)
        
class TestNumpyMathematicalFunctions(TestNumpyMethods):
    # https://www.numpy.org/devdocs/reference/routines.math.html
    # Trigonometric functions
    @helpers.requires_array_function_protocol()
    def test_unwrap(self):
        self.assertQuantityEqual(np.unwrap([0,3*np.pi]*self.ureg.radians), [0,np.pi])
        self.assertQuantityEqual(np.unwrap([0,540]*self.ureg.deg), [0,180]*self.ureg.deg)
        
    # Rounding

    @helpers.requires_array_function_protocol()
    def test_fix(self):
        self.assertQuantityEqual(np.fix(3.14 * self.ureg.m), 3.0 * self.ureg.m)
        self.assertQuantityEqual(np.fix(3.0 * self.ureg.m), 3.0 * self.ureg.m)
        self.assertQuantityEqual(
            np.fix([2.1, 2.9, -2.1, -2.9] * self.ureg.m),
            [2., 2., -2., -2.] * self.ureg.m
        )
    # Sums, products, differences

    def test_prod(self):
        self.assertEqual(self.q.prod(), 24 * self.ureg.m**4)
        
    def test_sum(self):
        self.assertEqual(self.q.sum(), 10*self.ureg.m)
        self.assertQuantityEqual(self.q.sum(0), [4,     6]*self.ureg.m)
        self.assertQuantityEqual(self.q.sum(1), [3, 7]*self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_sum_numpy_func(self):
        self.assertQuantityEqual(np.sum(self.q, axis=0), [4, 6] * self.ureg.m)
        self.assertRaises(OffsetUnitCalculusError, np.sum, self.q_temperature)

    @helpers.requires_array_function_protocol()
    def test_nansum_numpy_func(self):
        self.assertQuantityEqual(np.nansum(self.q_nan, axis=0), [4, 2] * self.ureg.m)

    def test_cumprod(self):
        self.assertRaises(ValueError, self.q.cumprod)
        self.assertQuantityEqual((self.q / self.ureg.m).cumprod(), [1, 2, 6, 24])

    @helpers.requires_array_function_protocol()
    def test_cumprod_numpy_func(self):
        self.assertRaises(DimensionalityError, np.cumprod, self.q)
        self.assertRaises(DimensionalityError, np.cumproduct, self.q)
        self.assertQuantityEqual(np.cumprod(self.q / self.ureg.m), [1, 2, 6, 24])
        self.assertQuantityEqual(np.cumproduct(self.q / self.ureg.m), [1, 2, 6, 24])

    @helpers.requires_array_function_protocol()
    def test_nancumprod_numpy_func(self):
        self.assertRaises(DimensionalityError, np.nancumprod, self.q_nan)
        self.assertQuantityEqual(np.nancumprod(self.q_nan / self.ureg.m), [1, 2, 6, 6])

    @helpers.requires_array_function_protocol()
    def test_diff(self):
        self.assertQuantityEqual(np.diff(self.q, 1), [[1], [1]] * self.ureg.m)
        self.assertQuantityEqual(np.diff(self.q_temperature, 1), [[1], [1]] * self.ureg.delta_degC)

    @helpers.requires_array_function_protocol()
    def test_ediff1d(self):
        self.assertQuantityEqual(np.ediff1d(self.q), [1, 1, 1] * self.ureg.m)
        self.assertQuantityEqual(np.ediff1d(self.q_temperature), [1, 1, 1] * self.ureg.delta_degC)

    @helpers.requires_array_function_protocol()
    def test_gradient(self):
        l = np.gradient([[1,1],[3,4]] * self.ureg.m, 1 * self.ureg.J)
        self.assertQuantityEqual(l[0], [[2., 3.], [2., 3.]] * self.ureg.m / self.ureg.J)
        self.assertQuantityEqual(l[1], [[0., 0.], [1., 1.]] * self.ureg.m / self.ureg.J)

        l = np.gradient(self.Q_([[1,1],[3,4]] , self.ureg.degC), 1 * self.ureg.J)
        self.assertQuantityEqual(l[0], [[2., 3.], [2., 3.]] * self.ureg.delta_degC / self.ureg.J)
        self.assertQuantityEqual(l[1], [[0., 0.], [1., 1.]] * self.ureg.delta_degC / self.ureg.J)

    @helpers.requires_array_function_protocol()
    def test_cross(self):
        a = [[3,-3, 1]] * self.ureg.kPa
        b = [[4, 9, 2]] * self.ureg.m**2
        self.assertQuantityEqual(np.cross(a, b), [[-15, -2, 39]] * self.ureg.kPa * self.ureg.m**2)

    @helpers.requires_array_function_protocol()
    def test_trapz(self):
        self.assertQuantityEqual(np.trapz([1. ,2., 3., 4.] * self.ureg.J, dx=1*self.ureg.m), 7.5 * self.ureg.J*self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_dot_numpy_func(self):
        self.assertQuantityEqual(np.dot(self.q.ravel(), [0, 0, 1, 0] * self.ureg.dimensionless), 3 * self.ureg.m)

    # Arithmetic operations
    def test_addition_with_scalar(self):
        a = np.array([0, 1, 2])
        b = 10. * self.ureg('gram/kilogram')
        self.assertQuantityAlmostEqual(a + b, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless))
        self.assertQuantityAlmostEqual(b + a, self.Q_([0.01, 1.01, 2.01], self.ureg.dimensionless))

    def test_addition_with_incompatible_scalar(self):
        a = np.array([0, 1, 2])
        b = 1. * self.ureg.m
        self.assertRaises(DimensionalityError, op.add, a, b)
        self.assertRaises(DimensionalityError, op.add, b, a)

    def test_power(self):
        arr = np.array(range(3), dtype=np.float)
        q = self.Q_(arr, 'meter')

        for op_ in [op.pow, op.ipow, np.power]:
            q_cp = copy.copy(q)
            self.assertRaises(DimensionalityError, op_, 2., q_cp)
            arr_cp = copy.copy(arr)
            arr_cp = copy.copy(arr)
            q_cp = copy.copy(q)
            self.assertRaises(DimensionalityError, op_, q_cp, arr_cp)
            q_cp = copy.copy(q)
            q2_cp = copy.copy(q)
            self.assertRaises(DimensionalityError, op_, q_cp, q2_cp)

    @unittest.expectedFailure
    @helpers.requires_numpy()
    def test_exponentiation_array_exp_2(self):
        arr = np.array(range(3), dtype=np.float)
        #q = self.Q_(copy.copy(arr), None)
        q = self.Q_(copy.copy(arr), 'meter')
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        # this fails as expected since numpy 1.8.0 but...
        self.assertRaises(DimensionalityError, op.pow, arr_cp, q_cp)
        # ..not for op.ipow !
        # q_cp is treated as if it is an array. The units are ignored.
        # Quantity.__ipow__ is never called
        arr_cp = copy.copy(arr)
        q_cp = copy.copy(q)
        self.assertRaises(DimensionalityError, op.ipow, arr_cp, q_cp)

class TestNumpyUnclassified(TestNumpyMethods):
    def test_tolist(self):
        self.assertEqual(self.q.tolist(), [[1*self.ureg.m, 2*self.ureg.m], [3*self.ureg.m, 4*self.ureg.m]])

    def test_fill(self):
        tmp = self.q
        tmp.fill(6 * self.ureg.ft)
        self.assertQuantityEqual(tmp, [[6, 6], [6, 6]] * self.ureg.ft)
        tmp.fill(5 * self.ureg.m)
        self.assertQuantityEqual(tmp, [[5, 5], [5, 5]] * self.ureg.m)
        
    def test_take(self):
        self.assertQuantityEqual(self.q.take([0,1,2,3]), self.q.flatten())

    def test_put(self):
        q =  [1., 2., 3., 4.] * self.ureg.m
        q.put([0, 2], [10.,20.]*self.ureg.m)
        self.assertQuantityEqual(q, [10., 2., 20., 4.]*self.ureg.m)

        q =  [1., 2., 3., 4.] * self.ureg.m
        q.put([0, 2], [1., 2.]*self.ureg.mm)
        self.assertQuantityEqual(q, [0.001, 2., 0.002, 4.]*self.ureg.m)

        q =  [1., 2., 3., 4.] * self.ureg.m / self.ureg.mm
        q.put([0, 2], [1., 2.])
        self.assertQuantityEqual(q, [0.001, 2., 0.002, 4.]*self.ureg.m/self.ureg.mm)

        q =  [1., 2., 3., 4.] * self.ureg.m
        with self.assertRaises(DimensionalityError):
            q.put([0, 2], [4., 6.] * self.ureg.J)
        with self.assertRaises(DimensionalityError):
            q.put([0, 2], [4., 6.])

    def test_repeat(self):
        self.assertQuantityEqual(self.q.repeat(2), [1,1,2,2,3,3,4,4]*self.ureg.m)

    def test_sort(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        q.sort()
        self.assertQuantityEqual(q, [1, 2, 3, 4, 5, 6] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_sort_numpy_func(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        self.assertQuantityEqual(np.sort(q), [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort(self):
        q = [1, 4, 5, 6, 2, 9] * self.ureg.MeV
        self.assertNDArrayEqual(q.argsort(), [0, 4, 1, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_argsort_numpy_func(self):
        self.assertNDArrayEqual(np.argsort(self.q, axis=0), np.array([[0, 0], [1, 1]]))

    def test_diagonal(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        self.assertQuantityEqual(q.diagonal(offset=1), [2, 3] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_diagonal_numpy_func(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        self.assertQuantityEqual(np.diagonal(q, offset=-1), [1, 2] * self.ureg.m)

    def test_compress(self):
        self.assertQuantityEqual(self.q.compress([False, True], axis=0),
                                 [[3, 4]] * self.ureg.m)
        self.assertQuantityEqual(self.q.compress([False, True], axis=1),
                                 [[2], [4]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_compress(self):
        self.assertQuantityEqual(np.compress([False, True], self.q, axis=1),
                                 [[2], [4]] * self.ureg.m)

    def test_searchsorted(self):
        q = self.q.flatten()
        self.assertNDArrayEqual(q.searchsorted([1.5, 2.5] * self.ureg.m),
                                      [1, 2])
        q = self.q.flatten()
        self.assertRaises(DimensionalityError, q.searchsorted, [1.5, 2.5])

    @helpers.requires_array_function_protocol()
    def test_searchsorted_numpy_func(self):
        """Test searchsorted as numpy function."""
        q = self.q.flatten()
        self.assertNDArrayEqual(np.searchsorted(q, [1.5, 2.5] * self.ureg.m),
                                      [1, 2])

    def test_nonzero(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(q.nonzero()[0], [0, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertNDArrayEqual(np.nonzero(q)[0], [0, 2, 3, 5])

    @helpers.requires_array_function_protocol()
    def test_count_nonzero_numpy_func(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertEqual(np.count_nonzero(q), 4)

    def test_max(self):
        self.assertEqual(self.q.max(), 4*self.ureg.m)

    def test_max_numpy_func(self):
        self.assertEqual(np.max(self.q), 4 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_max_with_axis_arg(self):
        self.assertQuantityEqual(np.max(self.q, axis=1), [2, 4] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_max_with_initial_arg(self):
        self.assertQuantityEqual(np.max(self.q[..., None], axis=2, initial=3 * self.ureg.m), [[3, 3], [3, 4]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_nanmax(self):
        self.assertEqual(np.nanmax(self.q_nan), 3 * self.ureg.m)

    def test_argmax(self):
        self.assertEqual(self.q.argmax(), 3)

    @helpers.requires_array_function_protocol()
    def test_argmax_numpy_func(self):
        self.assertNDArrayEqual(np.argmax(self.q, axis=0), np.array([1, 1]))

    @helpers.requires_array_function_protocol()
    def test_nanargmax_numpy_func(self):
        self.assertNDArrayEqual(np.nanargmax(self.q_nan, axis=0), np.array([1, 0]))

    def test_maximum(self):
        self.assertQuantityEqual(np.maximum(self.q, self.Q_([0, 5], 'm')),
                                 self.Q_([[1, 5], [3, 5]], 'm'))

    def test_min(self):
        self.assertEqual(self.q.min(), 1 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_min_numpy_func(self):
        self.assertEqual(np.min(self.q), 1 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_min_with_axis_arg(self):
        self.assertQuantityEqual(np.min(self.q, axis=1), [1, 3] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_min_with_initial_arg(self):
        self.assertQuantityEqual(np.min(self.q[..., None], axis=2, initial=3 * self.ureg.m), [[1, 2], [3, 3]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_nanmin(self):
        self.assertEqual(np.nanmin(self.q_nan), 1 * self.ureg.m)

    def test_argmin(self):
        self.assertEqual(self.q.argmin(), 0)

    @helpers.requires_array_function_protocol()
    def test_argmin_numpy_func(self):
        self.assertNDArrayEqual(np.argmin(self.q, axis=0), np.array([0, 0]))

    @helpers.requires_array_function_protocol()
    def test_nanargmin_numpy_func(self):
        self.assertNDArrayEqual(np.nanargmin(self.q_nan, axis=0), np.array([0, 0]))

    def test_minimum(self):
        self.assertQuantityEqual(np.minimum(self.q, self.Q_([0, 5], 'm')),
                                 self.Q_([[0, 2], [0, 4]], 'm'))

    def test_ptp(self):
        self.assertEqual(self.q.ptp(), 3 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_ptp_numpy_func(self):
        self.assertQuantityEqual(np.ptp(self.q, axis=0), [2, 2] * self.ureg.m)

    def test_clip(self):
        self.assertQuantityEqual(
            self.q.clip(max=2*self.ureg.m),
            [[1, 2], [2, 2]] * self.ureg.m
        )
        self.assertQuantityEqual(
            self.q.clip(min=3*self.ureg.m),
            [[3, 3], [3, 4]] * self.ureg.m
        )
        self.assertQuantityEqual(
            self.q.clip(min=2*self.ureg.m, max=3*self.ureg.m),
            [[2, 2], [3, 3]] * self.ureg.m
        )
        self.assertRaises(DimensionalityError, self.q.clip, self.ureg.J)
        self.assertRaises(DimensionalityError, self.q.clip, 1)

    @helpers.requires_array_function_protocol()
    def test_clip_numpy_func(self):
        self.assertQuantityEqual(np.clip(self.q, 150 * self.ureg.cm, None), [[1.5, 2], [3, 4]] * self.ureg.m)

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * self.ureg.m
        self.assertQuantityEqual(q.round(0), [1, 1, 6, 22] * self.ureg.m)
        self.assertQuantityEqual(q.round(-1), [0, 0, 10, 20] * self.ureg.m)
        self.assertQuantityEqual(q.round(1), [1, 1.3, 5.7, 22] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_round_numpy_func(self):
        self.assertQuantityEqual(np.around(1.0275 * self.ureg.m, decimals=2), 1.03 * self.ureg.m)
        self.assertQuantityEqual(np.round_(1.0275 * self.ureg.m, decimals=2), 1.03 * self.ureg.m)

    def test_trace(self):
        self.assertEqual(self.q.trace(), (1+4) * self.ureg.m)

    def test_cumsum(self):
        self.assertQuantityEqual(self.q.cumsum(), [1, 3, 6, 10] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_cumsum_numpy_func(self):
        self.assertQuantityEqual(np.cumsum(self.q, axis=0), [[1, 2], [4, 6]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_nancumsum_numpy_func(self):
        self.assertQuantityEqual(np.nancumsum(self.q_nan, axis=0), [[1, 2], [4, 2]] * self.ureg.m)

    def test_mean(self):
        self.assertEqual(self.q.mean(), 2.5 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_mean_numpy_func(self):
        self.assertEqual(np.mean(self.q), 2.5 * self.ureg.m)
        self.assertEqual(np.mean(self.q_temperature), self.Q_(2.5, self.ureg.degC))

    @helpers.requires_array_function_protocol()
    def test_nanmean_numpy_func(self):
        self.assertEqual(np.nanmean(self.q_nan), 2 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_average_numpy_func(self):
        self.assertQuantityAlmostEqual(np.average(self.q, axis=0, weights=[1, 2]), [2.33333, 3.33333] * self.ureg.m, rtol=1e-5)

    @helpers.requires_array_function_protocol()
    def test_median_numpy_func(self):
        self.assertEqual(np.median(self.q), 2.5 * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_nanmedian_numpy_func(self):
        self.assertEqual(np.nanmedian(self.q_nan), 2 * self.ureg.m)

    def test_var(self):
        self.assertEqual(self.q.var(), 1.25*self.ureg.m**2)

    @helpers.requires_array_function_protocol()
    def test_var_numpy_func(self):
        self.assertEqual(np.var(self.q), 1.25*self.ureg.m**2)

    @helpers.requires_array_function_protocol()
    def test_nanvar_numpy_func(self):
        self.assertQuantityAlmostEqual(np.nanvar(self.q_nan), 0.66667*self.ureg.m**2, rtol=1e-5)

    def test_std(self):
        self.assertQuantityAlmostEqual(self.q.std(), 1.11803*self.ureg.m, rtol=1e-5)

    @helpers.requires_array_function_protocol()
    def test_std_numpy_func(self):
        self.assertQuantityAlmostEqual(np.std(self.q), 1.11803*self.ureg.m, rtol=1e-5)
        self.assertRaises(OffsetUnitCalculusError, np.std, self.q_temperature)

    def test_prod(self):
        self.assertEqual(self.q.prod(), 24 * self.ureg.m**4)

    def test_cumprod(self):
        self.assertRaises(DimensionalityError, self.q.cumprod)
        self.assertQuantityEqual((self.q / self.ureg.m).cumprod(), [1, 2, 6, 24])

    @helpers.requires_array_function_protocol()
    def test_nanstd_numpy_func(self):
        self.assertQuantityAlmostEqual(np.nanstd(self.q_nan), 0.81650 * self.ureg.m, rtol=1e-5)
        
    @helpers.requires_numpy_previous_than('1.10')
    def test_integer_div(self):
        a = [1] * self.ureg.m
        b = [2] * self.ureg.m
        c = a/b  # Should be float division
        self.assertEqual(c.magnitude[0], 0.5)

        a /= b  # Should be integer division
        self.assertEqual(a.magnitude[0], 0)

    def test_conj(self):
        self.assertQuantityEqual((self.q*(1+1j)).conj(), self.q*(1-1j))
        self.assertQuantityEqual((self.q*(1+1j)).conjugate(), self.q*(1-1j))

    def test_getitem(self):
        self.assertRaises(IndexError, self.q.__getitem__, (0,10))
        self.assertQuantityEqual(self.q[0], [1,2]*self.ureg.m)
        self.assertEqual(self.q[1,1], 4*self.ureg.m)

    def test_setitem(self):
        with self.assertRaises(TypeError):
            self.q[0, 0] = 1
        with self.assertRaises(DimensionalityError):
            self.q[0, 0] = 1 * self.ureg.J
        with self.assertRaises(DimensionalityError):
            self.q[0] = 1
        with self.assertRaises(DimensionalityError):
            self.q[0] = np.ndarray([1, 2])
        with self.assertRaises(DimensionalityError):
            self.q[0] = 1 * self.ureg.J

        q = self.q.copy()
        q[0] = 1 * self.ureg.m
        self.assertQuantityEqual(q, [[1, 1], [3, 4]] * self.ureg.m)

        q = self.q.copy()
        q[...] = 1 * self.ureg.m
        self.assertQuantityEqual(q, [[1, 1], [1, 1]] * self.ureg.m)

        q = self.q.copy()
        q[:] = 1 * self.ureg.m
        self.assertQuantityEqual(q, [[1, 1], [1, 1]] * self.ureg.m)

        # check and see that dimensionless num  bers work correctly
        q = [0, 1, 2, 3] * self.ureg.dimensionless
        q[0] = 1
        self.assertQuantityEqual(q, np.asarray([1, 1, 2, 3]))
        q[0] = self.ureg.m/self.ureg.mm
        self.assertQuantityEqual(q, np.asarray([1000, 1, 2, 3]))

        q = [0., 1., 2., 3.] * self.ureg.m / self.ureg.mm
        q[0] = 1.0
        self.assertQuantityEqual(q, [0.001, 1, 2, 3] * self.ureg.m / self.ureg.mm)

    def test_iterator(self):
        for q, v in zip(self.q.flatten(), [1, 2, 3, 4]):
            self.assertEqual(q, v * self.ureg.m)

    def test_iterable(self):
        self.assertTrue(np.iterable(self.q))
        self.assertFalse(np.iterable(1 * self.ureg.m))

    def test_reversible_op(self):
        """
        """
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))
        self.assertQuantityEqual(x / self.q, u * x / self.q)
        self.assertQuantityEqual(x * self.q, u * x * self.q)
        self.assertQuantityEqual(x + u, u + x)
        self.assertQuantityEqual(x - u, -(u - x))

    def test_pickle(self):
        import pickle
        set_application_registry(self.ureg)
        def pickle_test(q):
            pq = pickle.loads(pickle.dumps(q))
            self.assertNDArrayEqual(q.magnitude, pq.magnitude)
            self.assertEqual(q.units, pq.units)

        pickle_test([10,20]*self.ureg.m)

    def test_equal(self):
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))

        self.assertQuantityEqual(u, u)
        self.assertQuantityEqual(u == u, u.magnitude == u.magnitude)
        self.assertQuantityEqual(u == 1, u.magnitude == 1)

    def test_shape(self):
        u = self.Q_(np.arange(12))
        u.shape = 4, 3
        self.assertEqual(u.magnitude.shape, (4, 3))

    @helpers.requires_array_function_protocol()
    def test_shape_numpy_func(self):
        self.assertEqual(np.shape(self.q), (2, 2))

    @helpers.requires_array_function_protocol()
    def test_alen_numpy_func(self):
        self.assertEqual(np.alen(self.q), 2)

    @helpers.requires_array_function_protocol()
    def test_ndim_numpy_func(self):
        self.assertEqual(np.ndim(self.q), 2)

    @helpers.requires_array_function_protocol()
    def test_copy_numpy_func(self):
        q_copy = np.copy(self.q)
        self.assertQuantityEqual(self.q, q_copy)
        self.assertIsNot(self.q, q_copy)

    @helpers.requires_array_function_protocol()
    def test_trim_zeros_numpy_func(self):
        q = [0, 4, 3, 0, 2, 2, 0, 0, 0] * self.ureg.m
        self.assertQuantityEqual(np.trim_zeros(q), [4, 3, 0, 2, 2] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_result_type_numpy_func(self):
        self.assertEqual(np.result_type(self.q), np.dtype('int64'))

    @helpers.requires_array_function_protocol()
    def test_nan_to_num_numpy_func(self):
        self.assertQuantityEqual(np.nan_to_num(self.q_nan, nan=-999 * self.ureg.mm),
                                 [[1, 2], [3, -0.999]] * self.ureg.m)

    @helpers.requires_array_function_protocol()
    def test_meshgrid_numpy_func(self):
        x = [1, 2] * self.ureg.m
        y = [0, 50, 100] * self.ureg.mm
        xx, yy = np.meshgrid(x, y)
        self.assertQuantityEqual(xx, [[1, 2], [1, 2], [1, 2]] * self.ureg.m)
        self.assertQuantityEqual(yy, [[0, 0], [50, 50], [100, 100]] * self.ureg.mm)

    @helpers.requires_array_function_protocol()
    def test_isclose_numpy_func(self):
        q2 = [[1000.05, 2000], [3000.00007, 4001]] * self.ureg.mm
        self.assertNDArrayEqual(np.isclose(self.q, q2), np.array([[False, True], [True, False]]))

    @helpers.requires_array_function_protocol()
    def test_interp_numpy_func(self):
        x = [1, 4] * self.ureg.m
        xp = np.linspace(0, 3, 5) * self.ureg.m
        fp = self.Q_([0, 5, 10, 15, 20], self.ureg.degC)
        self.assertQuantityAlmostEqual(np.interp(x, xp, fp), self.Q_([6.66667, 20.], self.ureg.degC), rtol=1e-5)

    def test_comparisons(self):
        self.assertNDArrayEqual(self.q > 2 * self.ureg.m, np.array([[False, False], [True, True]]))
        self.assertNDArrayEqual(self.q < 2 * self.ureg.m, np.array([[True, False], [False, False]]))

    @helpers.requires_array_function_protocol()
    def test_where(self):
        self.assertQuantityEqual(np.where(self.q >= 2 * self.ureg.m, self.q, 0 * self.ureg.m),
                                 [[0, 2], [3, 4]] * self.ureg.m)
        self.assertRaises(DimensionalityError, np.where, self.q < 2 * self.ureg.m, self.q, 0)

    def test_fabs(self):
        self.assertQuantityEqual(np.fabs(self.q - 2 * self.ureg.m), self.Q_([[1, 0], [1, 2]], 'm'))


@unittest.skip
class TestBitTwiddlingUfuncs(TestUFuncs):
    """Universal functions (ufuncs) >  Bittwiddling functions

    http://docs.scipy.org/doc/numpy/reference/ufuncs.html#bittwiddlingfunctions

    bitwise_and(x1, x2[, out])         Compute the bitwise AND of two arrays elementwise.
    bitwise_or(x1, x2[, out])  Compute the bitwise OR of two arrays elementwise.
    bitwise_xor(x1, x2[, out])         Compute the bitwise XOR of two arrays elementwise.
    invert(x[, out])   Compute bitwise inversion, or bitwise NOT, elementwise.
    left_shift(x1, x2[, out])  Shift the bits of an integer to the left.
    right_shift(x1, x2[, out])         Shift the bits of an integer to the right.
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
        self._test2(np.bitwise_and,
                    self.q1,
                    (self.q2, self.qs,),
                    (self.qm, ),
                    'same')

    def test_bitwise_or(self):
        self._test2(np.bitwise_or,
                    self.q1,
                    (self.q1, self.q2, self.qs, ),
                    (self.qm,),
                    'same')

    def test_bitwise_xor(self):
        self._test2(np.bitwise_xor,
                    self.q1,
                    (self.q1, self.q2, self.qs, ),
                    (self.qm, ),
                    'same')

    def test_invert(self):
        self._test1(np.invert,
                    (self.q1, self.q2, self.qs, ),
                    (),
                    'same')

    def test_left_shift(self):
        self._test2(np.left_shift,
                    self.q1,
                    (self.qless, 2),
                    (self.q1, self.q2, self.qs, ),
                    'same')

    def test_right_shift(self):
        self._test2(np.right_shift,
                    self.q1,
                    (self.qless, 2),
                    (self.q1, self.q2, self.qs, ),
                    'same')
