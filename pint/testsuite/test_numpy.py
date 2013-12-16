# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

from pint.testsuite import TestCase, HAS_NUMPY, np

@unittest.skipUnless(HAS_NUMPY, 'Numpy not present')
class TestNumpyMethods(TestCase):

    FORCE_NDARRAY = True

    @property
    def q(self):
        return [[1,2],[3,4]] * self.ureg.m

    def test_tolist(self):
        self.assertEqual(self.q.tolist(), [[1*self.ureg.m, 2*self.ureg.m], [3*self.ureg.m, 4*self.ureg.m]])

    def test_sum(self):
        self.assertEqual(self.q.sum(), 10*self.ureg.m)
        self.assertSequenceEqual(self.q.sum(0), [4,     6]*self.ureg.m)
        self.assertSequenceEqual(self.q.sum(1), [3, 7]*self.ureg.m)

    def test_fill(self):
        tmp = self.q
        tmp.fill(6 * self.ureg.ft)
        self.assertSequenceEqual(tmp, [[6, 6], [6, 6]] * self.ureg.ft)
        tmp.fill(5 * self.ureg.m)
        self.assertSequenceEqual(tmp, [[5, 5], [5, 5]] * self.ureg.m)

    def test_reshape(self):
        self.assertSequenceEqual(self.q.reshape([1,4]), [[1, 2, 3, 4]] * self.ureg.m)

    def test_transpose(self):
        self.assertSequenceEqual(self.q.transpose(), [[1, 3], [2, 4]] * self.ureg.m)

    def test_flatten(self):
        self.assertSequenceEqual(self.q.flatten(), [1, 2, 3, 4] * self.ureg.m)

    def test_ravel(self):
        self.assertSequenceEqual(self.q.ravel(), [1, 2, 3, 4] * self.ureg.m)

    def test_squeeze(self):
        self.assertSequenceEqual(
            self.q.reshape([1,4]).squeeze(),
            [1, 2, 3, 4] * self.ureg.m
        )

    def test_take(self):
        self.assertSequenceEqual(self.q.take([0,1,2,3]), self.q.flatten())

    def test_put(self):
        q =  [1., 2., 3., 4.] * self.ureg.m
        q.put([0, 2], [10.,20.]*self.ureg.m)
        self.assertSequenceEqual(q, [10., 2., 20., 4.]*self.ureg.m)

        q =  [1., 2., 3., 4.] * self.ureg.m
        q.put([0, 2], [1., 2.]*self.ureg.mm)
        self.assertSequenceEqual(q, [0.001, 2., 0.002, 4.]*self.ureg.m)

        q =  [1., 2., 3., 4.] * self.ureg.m / self.ureg.mm
        q.put([0, 2], [1., 2.])
        self.assertSequenceEqual(q, [0.001, 2., 0.002, 4.]*self.ureg.m/self.ureg.mm)

        q =  [1., 2., 3., 4.] * self.ureg.m
        self.assertRaises(ValueError, q.put, [0, 2], [4., 6.] * self.ureg.J)
        self.assertRaises(ValueError, q.put, [0, 2], [4., 6.])

    def test_repeat(self):
        self.assertSequenceEqual(self.q.repeat(2), [1,1,2,2,3,3,4,4]*self.ureg.m)

    def test_sort(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        q.sort()
        self.assertSequenceEqual(q, [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort(self):
        q = [1, 4, 5, 6, 2, 9] * self.ureg.MeV
        self.assertSequenceEqual(q.argsort(), [0, 4, 1, 2, 3, 5])

    def test_diagonal(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        self.assertSequenceEqual(q.diagonal(offset=1), [2, 3] * self.ureg.m)

    def test_compress(self):
        self.assertSequenceEqual(self.q.compress([False, True], axis=0),
                                 [[3, 4]] * self.ureg.m)
        self.assertSequenceEqual(self.q.compress([False, True], axis=1),
                                 [[2], [4]] * self.ureg.m)

    def test_searchsorted(self):
        q = self.q.flatten()
        self.assertSequenceEqual(q.searchsorted([1.5, 2.5] * self.ureg.m),
                                 [1, 2])
        q = self.q.flatten()
        self.assertRaises(ValueError, q.searchsorted, [1.5, 2.5])

    def test_nonzero(self):
        q = [1, 0, 5, 6, 0, 9] * self.ureg.m
        self.assertSequenceEqual(q.nonzero()[0], [0, 2, 3, 5])

    def test_max(self):
        self.assertEqual(self.q.max(), 4*self.ureg.m)

    def test_argmax(self):
        self.assertEqual(self.q.argmax(), 3)

    def test_min(self):
        self.assertEqual(self.q.min(), 1 * self.ureg.m)

    def test_argmin(self):
        self.assertEqual(self.q.argmin(), 0)

    def test_ptp(self):
        self.assertEqual(self.q.ptp(), 3 * self.ureg.m)

    def test_clip(self):
        self.assertSequenceEqual(
            self.q.clip(max=2*self.ureg.m),
            [[1, 2], [2, 2]] * self.ureg.m
        )
        self.assertSequenceEqual(
            self.q.clip(min=3*self.ureg.m),
            [[3, 3], [3, 4]] * self.ureg.m
        )
        self.assertSequenceEqual(
            self.q.clip(min=2*self.ureg.m, max=3*self.ureg.m),
            [[2, 2], [3, 3]] * self.ureg.m
        )
        self.assertRaises(ValueError, self.q.clip, self.ureg.J)
        self.assertRaises(ValueError, self.q.clip, 1)

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * self.ureg.m
        self.assertSequenceEqual(q.round(0), [1, 1, 6, 22] * self.ureg.m)
        self.assertSequenceEqual(q.round(-1), [0, 0, 10, 20] * self.ureg.m)
        self.assertSequenceEqual(q.round(1), [1, 1.3, 5.7, 22] * self.ureg.m)

    def test_trace(self):
        self.assertEqual(self.q.trace(), (1+4) * self.ureg.m)

    def test_cumsum(self):
        self.assertSequenceEqual(self.q.cumsum(), [1, 3, 6, 10] * self.ureg.m)

    def test_mean(self):
        self.assertEqual(self.q.mean(), 2.5 * self.ureg.m)

    def test_var(self):
        self.assertEqual(self.q.var(), 1.25*self.ureg.m**2)

    def test_std(self):
        self.assertAlmostEqual(self.q.std(), 1.11803*self.ureg.m, delta=1e-5)

    def test_prod(self):
        self.assertEqual(self.q.prod(), 24 * self.ureg.m**4)

    def test_cumprod(self):
        self.assertRaises(ValueError, self.q.cumprod)
        self.assertSequenceEqual((self.q / self.ureg.m).cumprod(), [1, 2, 6, 24])

    def test_conj(self):
        self.assertSequenceEqual((self.q*(1+1j)).conj(), self.q*(1-1j))
        self.assertSequenceEqual((self.q*(1+1j)).conjugate(), self.q*(1-1j))

    def test_getitem(self):
        self.assertRaises(IndexError, self.q.__getitem__, (0,10))
        self.assertSequenceEqual(self.q[0], [1,2]*self.ureg.m)
        self.assertEqual(self.q[1,1], 4*self.ureg.m)

    def test_setitem (self):
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1)
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1*self.ureg.J)
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1)
        self.assertRaises(ValueError, self.q.__setitem__, 0, np.ndarray([1, 2]))
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1*self.ureg.J)

        q = self.q.copy()
        q[0] = 1*self.ureg.m
        self.assertSequenceEqual(q, [[1,1],[3,4]]*self.ureg.m)

        q[0] = (1,2)*self.ureg.m
        self.assertSequenceEqual(q, self.q)

        q[:] = 1*self.ureg.m
        self.assertSequenceEqual(q, [[1,1],[1,1]]*self.ureg.m)

        # check and see that dimensionless num  bers work correctly
        q = [0,1,2,3]*self.ureg.dimensionless
        q[0] = 1
        self.assertSequenceEqual(q, [1,1,2,3])
        q[0] = self.ureg.m/self.ureg.mm
        self.assertSequenceEqual(q, [1000, 1,2,3])

        q = [0.,1.,2.,3.] * self.ureg.m/self.ureg.mm
        q[0] = 1
        self.assertSequenceEqual(q, [0.001,1,2,3]*self.ureg.m/self.ureg.mm)

    def test_iterator(self):
        for q, v in zip(self.q.flatten(), [1, 2, 3, 4]):
            self.assertEqual(q, v * self.ureg.m)

    def test_reversible_op(self):
        """
        """
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))
        self.assertSequenceEqual(x / self.q, u * x / self.q)
        self.assertSequenceEqual(x * self.q, u * x * self.q)
        self.assertSequenceEqual(x + u, u + x)
        self.assertSequenceEqual(x - u, -(u - x))

    def test_pickle(self):
        import pickle

        def pickle_test(q):
            pq = pickle.loads(pickle.dumps(q))
            self.assertSequenceEqual(q.magnitude, pq.magnitude)
            self.assertEqual(q.units, pq.units)

        pickle_test([10,20]*self.ureg.m)

    def test_equal(self):
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))

        self.assertSequenceEqual(u, u)
        self.assertSequenceEqual(u == u, u.magnitude == u.magnitude)
        self.assertSequenceEqual(u == 1, u.magnitude == 1)


from pint.testsuite.test_umath import TestUFuncs
@unittest.skip
class TestNumpyNotSupported(TestUFuncs):

    FORCE_NDARRAY = True


    def test_unwrap(self):
        """unwrap depends on diff
        """
        self.assertEqual(np.unwrap([0,3*np.pi]*self.ureg.radians), [0,np.pi])
        self.assertEqual(np.unwrap([0,540]*self.ureg.deg), [0,180]*self.ureg.deg)

    def test_trapz(self):
        """Units are erased by asanyarray, Quantity does not inherit from NDArray
        """
        self.assertEqual(np.trapz(self.q, dx = 1*self.ureg.m), 7.5 * self.ureg.J*self.ureg.m)

    def test_diff(self):
        """Units are erased by asanyarray, Quantity does not inherit from NDArray
        """
        self.assertSequenceEqual(np.diff(self.q, 1), [1, 1, 1] * self.ureg.J)

    def test_ediff1d(self):
        """Units are erased by asanyarray, Quantity does not inherit from NDArray
        """
        self.assertEqual(np.ediff1d(self.q, 1), [1, 1, 1] * self.ureg.J)

    def test_fix(self):
        """Units are erased by asanyarray, Quantity does not inherit from NDArray
        """
        self.assertEqual(np.fix(3.14 * self.ureg.m), 3.0 * self.ureg.m)
        self.assertEqual(np.fix(3.0 * self.ureg.m), 3.0 * self.ureg.m)
        self.assertSequenceEqual(
            np.fix([2.1, 2.9, -2.1, -2.9] * self.ureg.m),
            [2., 2., -2., -2.] * self.ureg.m
        )

    def test_gradient(self):
        """shape is a property not a function
        """
        l = np.gradient([[1,1],[3,4]] * self.ureg.J, 1 * self.ureg.m)
        self.assertEqual(l[0], [[2., 3.], [2., 3.]] * self.ureg.J / self.ureg.m)
        self.assertEqual(l[1], [[0., 0.], [1., 1.]] * self.ureg.J / self.ureg.m)

    def test_cross(self):
        """Units are erased by asarray, Quantity does not inherit from NDArray
        """
        a = [3,-3, 1] * self.ureg.kPa
        b = [4, 9, 2] * self.ureg.m**2
        self.assertSequenceEqual(np.cross(a,b), [-15,-2,39]*self.ureg.kPa*self.ureg.m**2)

    def test_power(self):
        """This is not supported as different elements might end up with different units

        eg. ([1, 1] * m) ** [2, 3]

        Must force exponent to single value
        """
        self._test2(np.power, self.q1,
                    (self.qless, np.asarray([1., 2, 3, 4])),
                    (self.q2, ),)

    def test_ones_like(self):
        """Units are erased by emptyarra, Quantity does not inherit from NDArray
        """
        self._test1(np.ones_like,
                    (self.q2, self.qs, self.qless, self.qi),
                    (),
                    2)


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

