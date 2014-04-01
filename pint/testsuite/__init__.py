# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals, print_function, absolute_import

import os
import logging
from contextlib import contextmanager

from pint.compat import ndarray, unittest, np

from pint import logger, UnitRegistry
from pint.quantity import _Quantity
from logging.handlers import BufferingHandler

#h = logging.StreamHandler()
#f = logging.Formatter("%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s")
#h.setLevel(logging.DEBUG)
#h.setFormatter(f)
#logger.addHandler(h)
#logger.setLevel(logging.DEBUG)


class TestHandler(BufferingHandler):

    def __init__(self, only_warnings=False):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        self.only_warnings = only_warnings
        BufferingHandler.__init__(self, 0)

    def shouldFlush(self):
        return False

    def emit(self, record):
        if self.only_warnings and record.level != logging.WARNING:
            return
        self.buffer.append(record.__dict__)


class QuantityTestCase(unittest.TestCase):

    FORCE_NDARRAY = False

    CHECK_NO_WARNING = False

    @classmethod
    def setUpClass(cls):
        cls.ureg = UnitRegistry(force_ndarray=cls.FORCE_NDARRAY)
        cls.Q_ = cls.ureg.Quantity

    @contextmanager
    def capture_log(self, level=logging.DEBUG):
        th = TestHandler()
        th.setLevel(level)
        logger.addHandler(th)
        yield th.buffer

    def setUp(self):
        self._test_handler = None
        if self.CHECK_NO_WARNING:
            self._test_handler = th = TestHandler()
            th.setLevel(logging.WARNING)
            logger.addHandler(th)

    def tearDown(self):
        if self._test_handler is not None:
            l = len(self._test_handler.buffer)
            self.assertEqual(l, 0, msg='%d warnings raised.')

    def _get_comparable_magnitudes(self, first, second, msg):
        if isinstance(first, _Quantity) and isinstance(second, _Quantity):
            second = second.to(first)
            self.assertEqual(first.units, second.units, msg=msg + 'Units are not equal.')
            m1, m2 = first.magnitude, second.magnitude
        elif isinstance(first, _Quantity):
            self.assertTrue(first.dimensionless, msg=msg + 'The first is not dimensionless.')
            first = first.to('')
            m1, m2 = first.magnitude, second
        elif isinstance(second, _Quantity):
            self.assertTrue(second.dimensionless, msg=msg + 'The second is not dimensionless.')
            second = second.to('')
            m1, m2 = first, second.magnitude
        else:
            m1, m2 = first, second

        return m1, m2

    def assertQuantityEqual(self, first, second, msg=None):
        if msg is None:
            msg = 'Comparing %r and %r. ' % (first, second)

        m1, m2 = self._get_comparable_magnitudes(first, second, msg)

        if isinstance(m1, ndarray) or isinstance(m2, ndarray):
            np.testing.assert_array_equal(m1, m2, err_msg=msg)
        else:
            unittest.TestCase.assertEqual(self, m1, m2, msg)

    def assertQuantityAlmostEqual(self, first, second, rtol=1e-07, atol=0, msg=None):
        if msg is None:
            msg = 'Comparing %r and %r. ' % (first, second)

        m1, m2 = self._get_comparable_magnitudes(first, second, msg)

        if isinstance(m1, ndarray) or isinstance(m2, ndarray):
            np.testing.assert_allclose(m1, m2, rtol=rtol, atol=atol, err_msg=msg)
        else:
            self.assertLessEqual(abs(m1 - m2), atol + rtol * abs(m2))


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

