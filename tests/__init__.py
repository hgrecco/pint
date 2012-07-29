# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import


import sys
import logging
import unittest

import numpy as np

from pint import logger, UnitRegistry

logger.setLevel(logging.DEBUG)
h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")
h.setFormatter(f)
logger.addHandler(h)

if sys.version < '3':
    import codecs
    string_types = basestring
    def u(x):
        return codecs.unicode_escape_decode(x)[0]
else:
    string_types = str
    def u(x):
        return x

class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ureg = UnitRegistry(force_ndarray=cls.FORCE_NDARRAY)
        cls.Q_ = cls.ureg.Quantity

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        if isinstance(seq1, (tuple, list)) and isinstance(seq2, np.ndarray):
            unittest.TestCase.assertSequenceEqual(self, seq1, seq2.tolist(), msg, seq_type)
        elif isinstance(seq2, (tuple, list)) and isinstance(seq1, np.ndarray):
            unittest.TestCase.assertSequenceEqual(self, seq1.tolist(), seq2, msg, seq_type)
        else:
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

