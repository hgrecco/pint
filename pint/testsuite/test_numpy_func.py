# -*- coding: utf-8 -*-

import pint.numpy_func

from pint.testsuite import QuantityTestCase, helpers
from pint.numpy_func import implements
from unittest.mock import patch


@helpers.requires_numpy()
class TestNumPyFuncUtils(QuantityTestCase):

    FORCE_NDARRAY = True

    @classmethod
    def setUpClass(cls):
        from pint import _DEFAULT_REGISTRY
        cls.ureg = _DEFAULT_REGISTRY
        cls.Q_ = cls.ureg.Quantity

    @patch('pint.numpy_func.HANDLED_FUNCTIONS', {})
    @patch('pint.numpy_func.HANDLED_UFUNCS', {})
    def test_implements(self):
        # Test for functions
        @implements('test', 'function')
        def test_function():
            pass

        self.assertEqual(pint.numpy_func.HANDLED_FUNCTIONS['test'], test_function)

        # Test for ufuncs
        @implements('test', 'ufunc')
        def test_ufunc():
            pass

        self.assertEqual(pint.numpy_func.HANDLED_UFUNCS['test'], test_ufunc)

        # Test for invalid func type
        with self.assertRaises(ValueError):
            @implements('test', 'invalid')
            def test_invalid():
                pass

    # TODO: fill in other functions in numpy_func
