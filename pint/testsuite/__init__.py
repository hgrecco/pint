# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import logging

from pint.compat import ndarray, unittest

from pint import logger, UnitRegistry
from pint.quantity import _Quantity
from logging.handlers import BufferingHandler

h = logging.StreamHandler()
f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")
h.setLevel(logging.DEBUG)
h.setFormatter(f)
logger.addHandler(h)
logger.setLevel(logging.DEBUG)


class TestHandler(BufferingHandler):
    def __init__(self):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        BufferingHandler.__init__(self, 0)

    def shouldFlush(self):
        return False

    def emit(self, record):
        self.buffer.append(record.__dict__)


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ureg = UnitRegistry(force_ndarray=cls.FORCE_NDARRAY)
        cls.Q_ = cls.ureg.Quantity

    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        if isinstance(seq1, _Quantity):
            if isinstance(seq2, _Quantity):
                seq2 = seq2.to(seq1).magnitude
                seq1 = seq1.magnitude
            else:
                seq1 = seq1.to('').magnitude
        if isinstance(seq2, _Quantity):
            if isinstance(seq1, _Quantity):
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
        if isinstance(first, _Quantity) and isinstance(second, _Quantity):
            second = second.to(first)
            unittest.TestCase.assertAlmostEqual(self, first.magnitude, second.magnitude, places, msg, delta)
            self.assertEqual(first.units, second.units)
        elif isinstance(first, _Quantity):
            self.assertTrue(first.dimensionless)
            first = first.to('')
            unittest.TestCase.assertAlmostEqual(self, first.magnitude, second, places, msg, delta)
        elif isinstance(second, _Quantity):
            self.assertTrue(second.dimensionless)
            second = second.to('')
            unittest.TestCase.assertAlmostEqual(self, first, second.magnitude, places, msg, delta)
        else:
            unittest.TestCase.assertAlmostEqual(self, first, second, places, msg, delta)

    def assertAlmostEqualRelError(self, first, second, rel, msg=None):
        if isinstance(first, _Quantity) and isinstance(second, _Quantity):
            second = second.to(first)
            val = abs((second - first) / (second + first))
        elif isinstance(first, _Quantity):
            self.assertTrue(first.dimensionless)
            first = first.to('')
            val = abs((second - first) / (second + first))
        elif isinstance(second, _Quantity):
            self.assertTrue(second.dimensionless)
            second = second.to('')
            val = abs((second - first) / (second + first))
        else:
            val = abs((second - first) / (second + first))
        self.assertLess(val, rel, msg=msg)


def testsuite():
    """A testsuite that has all the pint tests.
    """
    return unittest.TestLoader().discover(os.path.dirname(__file__))


def main():
    """Runs the testsuite as command line application."""
    try:
        unittest.main()
    except Exception as e:
        print('Error: %s' % e)


def run():
    """Run all tests i
    """
    test_runner = unittest.TextTestRunner()
    test_runner.run(testsuite())
