from pint.numpy_util import (
    is_quantity,
    is_quantity_with_scalar_magnitude,
    is_quantity_with_sequence_magnitude,
    is_sequence_with_quantity_elements,
)
from pint.compat import np
import pytest
from pint import Quantity as Q_


@pytest.mark.parametrize(
    "obj,result",
    [
        (Q_(1, "m"), True),
        (Q_(np.nan, "m"), True),
        (Q_([1, 2], "m"), True),
        (Q_([1, np.nan], "m"), True),
        (Q_(np.array([1, 2]), "m"), True),
        (Q_(np.array([1, np.nan]), "m"), True),
        (np.array([Q_(1, "m"), Q_(2, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), Q_(np.nan, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), np.nan], dtype="object"), False),
    ],
)
def test_is_quantity(obj, result):
    assert is_quantity(obj) == result


@pytest.mark.parametrize(
    "obj,result",
    [
        (Q_(1, "m"), True),
        (Q_(np.nan, "m"), True),
        (Q_([1, 2], "m"), False),
        (Q_([1, np.nan], "m"), False),
        (Q_(np.array([1, 2]), "m"), False),
        (Q_(np.array([1, np.nan]), "m"), False),
        (np.array([Q_(1, "m"), Q_(2, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), Q_(np.nan, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), np.nan], dtype="object"), False),
    ],
)
def test_is_quantity_with_scalar_magnitude(obj, result):
    assert is_quantity_with_scalar_magnitude(obj) == result


@pytest.mark.parametrize(
    "obj,result",
    [
        (Q_(1, "m"), False),
        (Q_(np.nan, "m"), False),
        (Q_([1, 2], "m"), True),
        (Q_([1, np.nan], "m"), True),
        (Q_(np.array([1, 2]), "m"), True),
        (Q_(np.array([1, np.nan]), "m"), True),
        (np.array([Q_(1, "m"), Q_(2, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), Q_(np.nan, "m")], dtype="object"), False),
        (np.array([Q_(1, "m"), np.nan], dtype="object"), False),
    ],
)
def test_is_quantity_with_sequence_magnitude(obj, result):
    assert is_quantity_with_sequence_magnitude(obj) == result


@pytest.mark.parametrize(
    "obj,result",
    [
        (Q_(1, "m"), False),
        (Q_(np.nan, "m"), False),
        (Q_([1, 2], "m"), True),
        (Q_([1, np.nan], "m"), True),
        (Q_(np.array([1, 2]), "m"), True),
        (Q_(np.array([1, np.nan]), "m"), True),
        (np.array([Q_(1, "m"), Q_(2, "m")], dtype="object"), True),
        (np.array([Q_(1, "m"), Q_(np.nan, "m")], dtype="object"), True),
        (np.array([Q_(1, "m"), np.nan], dtype="object"), True),
    ],
)
def test_is_sequence_with_quantity_elements(obj, result):
    assert is_sequence_with_quantity_elements(obj) == result
