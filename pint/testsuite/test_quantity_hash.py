from __future__ import annotations

import pickle

import pytest

from pint import UnitRegistry


@pytest.fixture(scope="module")
def registry_pair():
    return UnitRegistry(), UnitRegistry()


@pytest.mark.parametrize(
    "first, second",
    [("1 s", "1 s"), ("1 m", "100 cm"), ("1 kg", "1000 g")],
)
def test_equal_quantities_from_distinct_registries_have_equal_hashes(
    first, second, registry_pair
):
    left_registry, right_registry = registry_pair
    left = left_registry(first)
    right = right_registry(second)

    assert left == right
    assert hash(left) == hash(right)
    assert right in {left}
    assert {left: "value"}[right] == "value"


def test_quantity_hash_preserves_dimensionality(sess_registry):
    registry = sess_registry
    assert len({registry("1 m"), registry("1 s")}) == 2
    assert len({registry("1 m"), registry("100 cm")}) == 1
    assert hash(registry("100 percent")) == hash(1)


def test_quantity_hash_after_unpickling_into_application_registry(registry_pair):
    original = registry_pair[0]("1 s")
    restored = pickle.loads(pickle.dumps(original))

    assert restored == original
    assert hash(restored) == hash(original)
    assert {original: "value"}[restored] == "value"
