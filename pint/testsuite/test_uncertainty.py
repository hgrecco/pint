import pytest

from pint import DimensionalityError
from pint.testsuite import QuantityTestCase, helpers

from pint.compat import Uncertainty


# TODO: do not subclass from QuantityTestCase
@helpers.requires_autouncertainties()
class TestQuantity(QuantityTestCase):
    def test_simple(self):
        Q = self.ureg.Quantity
        Q(Uncertainty(4.0, 0.1), "s")

    def test_build(self):
        Q = self.ureg.Quantity
        v, u, w = self.Q_(4.0, "s"), self.Q_(0.1, "s"), self.Q_(0.1, "days")
        Q(Uncertainty(v.magnitude, u.magnitude), "s")
        (
            Q(Uncertainty(v.magnitude, u.magnitude), "s"),
            Q(Uncertainty.from_quantities(v, u)),
            v.plus_minus(u),
            v.plus_minus(w),
        )

    def test_raise_build(self):
        v, u = self.Q_(1.0, "s"), self.Q_(0.1, "s")
        o = self.Q_(0.1, "m")

        with pytest.raises(DimensionalityError):
            Uncertainty.from_quantities(v, u._magnitude)
        with pytest.raises(DimensionalityError):
            Uncertainty.from_quantities(v._magnitude, u)
        with pytest.raises(DimensionalityError):
            Uncertainty.from_quantities(v, o)
        with pytest.raises(DimensionalityError):
            v.plus_minus(o)
        with pytest.raises(DimensionalityError):
            v.plus_minus(u._magnitude)

    def test_propagate_linear(self):
        v = [0, 1, 2, 3, 4]
        e = [1, 2, 3, 4, 5]

        x_without_units = [Uncertainty(vi, ei) for vi, ei in zip(v, e)]
        x_with_units = [self.Q_(u, "s") for u in x_without_units]

        for x_nou, x_u in zip(x_without_units, x_with_units):
            for y_nou, y_u in zip(x_without_units, x_with_units):
                z_nou = x_nou + y_nou
                z_u = x_u + y_u
                assert z_nou.value == z_u.value.m
                assert z_nou.error == z_u.error.m

    def test_propagate_product(self):
        v = [1, 2, 3, 4]
        e = [1, 2, 3, 4, 5]

        x_without_units = [Uncertainty(vi, ei) for vi, ei in zip(v, e)]
        x_with_units = [self.Q_(u, "s") for u in x_without_units]

        for x_nou, x_u in zip(x_without_units, x_with_units):
            for y_nou, y_u in zip(x_without_units, x_with_units):
                z_nou = x_nou * y_nou
                z_u = x_u * y_u
                assert z_nou.value == z_u.value.m
                assert z_nou.error == z_u.error.m
