# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import sys
import logging
import unittest

try:
    import numpy as np
    HAS_NUMPY = True
    ndarray = np.ndarray
    NUMPY_VER = np.__version__
except ImportError:
    np = None
    HAS_NUMPY = False
    NUMPY_VER = 0
    class ndarray(object):
        pass

PYTHON3 = sys.version >= '3'

if PYTHON3:
    string_types = str
    def u(x):
        return x
else:
    import codecs
    string_types = basestring
    def u(x):
        return codecs.unicode_escape_decode(x)[0]

from pint import logger, UnitRegistry

h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")
h.setLevel(logging.DEBUG)
h.setFormatter(f)
logger.addHandler(h)
logger.setLevel(logging.DEBUG)


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ureg = UnitRegistry(force_ndarray=cls.FORCE_NDARRAY)
        cls.Q_ = cls.ureg.Quantity

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        if isinstance(seq1, self.Q_):
            if isinstance(seq2, self.Q_):
                seq2 = seq2.to(seq1).magnitude
                seq1 = seq1.magnitude
            else:
                seq1 = seq1.to('').magnitude
        if isinstance(seq2, self.Q_):
            if isinstance(seq1, self.Q_):
                seq1 = seq1.to(seq2).magnitude
                seq2 = seq2.magnitude
            else:
                seq2 = seq2.to('').magnitude
        if isinstance(seq1, ndarray):
            seq1 = seq1.tolist()
        if isinstance(seq2, ndarray):
            seq2 = seq2.tolist()
        unittest.TestCase.assertSequenceEqual(self, seq1, seq2, msg, seq_type)

    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        if isinstance(first, self.Q_) and isinstance(second, self.Q_):
            second = second.to(first)
            unittest.TestCase.assertAlmostEqual(self, first.magnitude, second.magnitude, places, msg, delta)
            self.assertEqual(first.units, second.units)
        elif isinstance(first, self.Q_):
            self.assertTrue(first.dimensionless)
            first = first.to('')
            unittest.TestCase.assertAlmostEqual(self, first.magnitude, second, places, msg, delta)
        elif isinstance(second, self.Q_):
            self.assertTrue(second.dimensionless)
            second = second.to('')
            unittest.TestCase.assertAlmostEqual(self, first, second.magnitude, places, msg, delta)
        else:
            unittest.TestCase.assertAlmostEqual(self, first, second, places, msg, delta)

    def assertAlmostEqualRelError(self, first, second, rel, msg=None):
        if isinstance(first, self.Q_) and isinstance(second, self.Q_):
            second = second.to(first)
            val = abs((second - first) / (second + first))
        elif isinstance(first, self.Q_):
            self.assertTrue(first.dimensionless)
            first = first.to('')
            val = abs((second - first) / (second + first))
        elif isinstance(second, self.Q_):
            self.assertTrue(second.dimensionless)
            second = second.to('')
            val = abs((second - first) / (second + first))
        else:
            val = abs((second - first) / (second + first))
        self.assertLess(val, rel, msg=msg)

def testsuite():
    """A testsuite that has all the pyflim tests.
    """
    return unittest.TestLoader().discover(os.path.dirname(__file__))


def main():
    """Runs the testsuite as command line application."""
    try:
        unittest.main()
    except Exception as e:
        print('Error: %s' % e)
