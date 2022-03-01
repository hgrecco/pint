import itertools
import logging

from pint import pi_theorem
from pint.testsuite import QuantityTestCase


# TODO: do not subclass from QuantityTestCase
class TestPiTheorem(QuantityTestCase):
    def test_simple(self, caplog):

        # simple movement
        with caplog.at_level(logging.DEBUG):
            assert pi_theorem({"V": "m/s", "T": "s", "L": "m"}) == [
                {"V": 1, "T": 1, "L": -1}
            ]

            # pendulum
            assert pi_theorem({"T": "s", "M": "grams", "L": "m", "g": "m/s**2"}) == [
                {"g": 1, "T": 2, "L": -1}
            ]
        assert len(caplog.records) == 7

    def test_inputs(self):
        V = "km/hour"
        T = "ms"
        L = "cm"

        f1 = lambda x: x
        f2 = lambda x: self.Q_(1, x)
        f3 = lambda x: self.Q_(1, x).units
        f4 = lambda x: self.Q_(1, x).dimensionality

        fs = f1, f2, f3, f4
        for fv, ft, fl in itertools.product(fs, fs, fs):
            qv = fv(V)
            qt = ft(T)
            ql = ft(L)
            assert self.ureg.pi_theorem({"V": qv, "T": qt, "L": ql}) == [
                {"V": 1.0, "T": 1.0, "L": -1.0}
            ]
