from __future__ import annotations

import subprocess
import sys
from decimal import Decimal
from fractions import Fraction

import pytest


def test_copy_without_numpy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["numpy"] = None
from pint import UnitRegistry
from pint.compat import HAS_NUMPY
assert not HAS_NUMPY
quantity = UnitRegistry().Quantity(3, "m")
result = quantity.copy()
assert result == quantity
assert result is not quantity
assert type(result.magnitude) is int
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("magnitude", [3, 3.5, Decimal("3.5"), Fraction(7, 2)])
def test_scalar_copy(func_registry, magnitude):
    quantity = func_registry.Quantity(magnitude, "m")

    result = quantity.copy()

    assert result is not quantity
    assert type(result) is type(quantity)
    assert result == quantity
    assert result.units == quantity.units
    result *= 2
    assert quantity.magnitude == magnitude
    assert result.magnitude == magnitude * 2
