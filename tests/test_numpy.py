# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import unittest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from tests import TestCase

@unittest.skipIf(not HAS_NUMPY, 'Numpy not present')
class TestQuantityMethods(TestCase):

    FORCE_NDARRAY = True

    @property
    def q(self):
        return [[1,2],[3,4]] * self.ureg.m

    def test_tolist(self):
        self.assertEqual(self.q.tolist(), [[1*self.ureg.m, 2*self.ureg.m], [3*self.ureg.m, 4*self.ureg.m]])

    def test_sum(self):
        self.assertEqual(self.q.sum(), 10*self.ureg.m)
        self.assertEqual(self.q.sum(0), [4,     6]*self.ureg.m)
        self.assertEqual(self.q.sum(1), [3, 7]*self.ureg.m)

    def test_fill(self):
        self.q.fill(6 * self.ureg.ft)
        self.assertEqual(self.q, [[6, 6], [6, 6]] * self.ureg.ft)
        self.q.fill(5)
        self.assertEqual(self.q, [[5, 5], [5, 5]] * self.ureg.ft)

    def test_reshape(self):
        self.assertEqual(self.q.reshape([1,4]), [[1, 2, 3, 4]] * self.ureg.m)

    def test_transpose(self):
        self.assertEqual(self.q.transpose(), [[1, 3], [2, 4]] * self.ureg.m)

    def test_flatten(self):
        self.assertEqual(self.q.flatten(), [1, 2, 3, 4] * self.ureg.m)

    def test_ravel(self):
        self.assertEqual(self.q.ravel(), [1, 2, 3, 4] * self.ureg.m)

    def test_squeeze(self):
        self.assertEqual(
            self.q.reshape([1,4]).squeeze(),
            [1, 2, 3, 4] * self.ureg.m
        )

    def test_take(self):
        self.assertEqual(self.q.take([0,1,2,3]), self.q.flatten())

    def test_put(self):
        q = self.q.flatten()
        q.put([0,2], [10,20]*self.ureg.m)
        self.assertEqual(q, [10, 2, 20, 4]*self.ureg.m)

        q = self.q.flatten()
        q.put([0, 2], [1, 2]*self.ureg.mm)
        self.assertEqual(q, [0.001, 2, 0.002, 4]*self.ureg.m)

        q = self.q.flatten()/self.ureg.mm
        q.put([0, 2], [1, 2])
        self.assertEqual(q.simplified, [1, 2000, 2, 4000])
        self.assertEqual(q, [0.001, 2, 0.002, 4]*self.ureg.m/self.ureg.mm)

        q = self.q.flatten()
        self.assertRaises(ValueError, q.put, [0, 2], [4, 6] * self.ureg.J)
        self.assertRaises(ValueError, q.put, [0, 2], [4, 6])

    def test_repeat(self):
        self.assertEqual(self.q.repeat(2), [1,1,2,2,3,3,4,4]*self.ureg.m)

    def test_sort(self):
        q = [4, 5, 2, 3, 1, 6] * self.ureg.m
        q.sort()
        self.assertEqual(q, [1, 2, 3, 4, 5, 6] * self.ureg.m)

    def test_argsort(self):
        q = [1, 4, 5, 6, 2, 9] * self.ureg.MeV
        self.assertSequenceEqual(q.argsort(), [0, 4, 1, 2, 3, 5])

    def test_diagonal(self):
        q = [[1, 2, 3], [1, 2, 3], [1, 2, 3]] * self.ureg.m
        self.assertEqual(q.diagonal(offset=1), [2, 3] * self.ureg.m)

    def test_compress(self):
        self.assertEqual(self.q.compress([False, True], axis=0),
                         [[3, 4]] * self.ureg.m)
        self.assertEqual(self.q.compress([False, True], axis=1),
                         [[2], [4]] * self.ureg.m)

    def test_searchsorted(self):
        self.assertEqual(self.q.flatten().searchsorted([1.5, 2.5] * self.ureg.m),
                         [1, 2])

        self.assertRaises(ValueError, self.q.flatten().searchsorted, [1.5, 2.5])

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
        self.assertEqual(
            self.q.copy().clip(max=2*self.ureg.m),
            [[1, 2], [2, 2]] * self.ureg.m
        )
        self.assertEqual(
            self.q.copy().clip(min=3*self.ureg.m),
            [[3, 3], [3, 4]] * self.ureg.m
        )
        self.assertEqual(
            self.q.copy().clip(min=2*self.ureg.m, max=3*self.ureg.m),
            [[2, 2], [3, 3]] * self.ureg.m
        )
        self.assertRaises(ValueError, self.q.clip, self.ureg.J)
        self.assertRaises(ValueError, self.q.clip, 1)

    def test_round(self):
        q = [1, 1.33, 5.67, 22] * self.ureg.m
        self.assertEqual(q.round(0), [1, 1, 6, 22] * self.ureg.m)
        self.assertEqual(q.round(-1), [0, 0, 10, 20] * self.ureg.m)
        self.assertEqual(q.round(1), [1, 1.3, 5.7, 22] * self.ureg.m)

    def test_trace(self):
        self.assertEqual(self.q.trace(), (1+4) * self.ureg.m)

    def test_cumsum(self):
        self.assertEqual(self.q.cumsum(), [1, 3, 6, 10] * self.ureg.m)

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
        self.assertEqual((self.q*(1+1j)).conj(), self.q*(1-1j))
        self.assertEqual((self.q*(1+1j)).conjugate(), self.q*(1-1j))

    def test_getitem(self):
        self.assertRaises(IndexError, self.q.__getitem__, (0,10))
        self.assertEqual(self.q[0], [1,2]*self.ureg.m)
        self.assertEqual(self.q[1,1], 4*self.ureg.m)

    def test_setitem (self):
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1)
        self.assertRaises(ValueError, self.q.__setitem__, (0,0), 1*self.ureg.J)
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1)
        self.assertRaises(ValueError, self.q.__setitem__, 0, np.ndarray([1, 2]))
        self.assertRaises(ValueError, self.q.__setitem__, 0, 1*self.ureg.J)

        q = self.q.copy()
        q[0] = 1*self.ureg.m
        self.assertEqual(q, [[1,1],[3,4]]*self.ureg.m)

        q[0] = (1,2)*self.ureg.m
        self.assertEqual(q, self.q)

        q[:] = 1*self.ureg.m
        self.assertEqual(q, [[1,1],[1,1]]*self.ureg.m)

        # check and see that dimensionless num  bers work correctly
        q = [0,1,2,3]*self.ureg.dimensionless
        q[0] = 1
        self.assertEqual(q, [1,1,2,3])
        q[0] = self.ureg.m/self.ureg.mm
        self.assertEqual(q, [1000, 1,2,3])

        q = [0.,1.,2.,3.] * self.ureg.m/self.ureg.mm
        q[0] = 1
        self.assertEqual(q, [0.001,1,2,3]*self.ureg.m/self.ureg.mm)

    def test_iterator(self):
        for q, v in zip(self.q.flatten(), [1, 2, 3, 4]):
            self.assertEqual(q, v * self.ureg.m)

    def test_reversible_op(self):
        """This fails because x / self.q returns an array, not Quantity.
        Division is handled by the ndarray code.
        """
        x = self.q.magnitude
        u = self.Q_(np.ones(x.shape))
        self.assertEqual(x / self.q, u * x / self.q)
        self.assertEqual(x * self.q, u * x * self.q)
        self.assertEqual(x + u, u + x)
        self.assertEqual(x - u, -(u - x))

    def test_pickle(self):
        import pickle

        def pickle_test(q):
            self.assertEqual(q, pickle.loads(pickle.dumps(q)))

        pickle_test([10,20]*self.ureg.m)
