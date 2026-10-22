import inspect
import math

import pytest

from pint import DimensionalityError


@pytest.fixture
def ureg(sess_registry):
    return sess_registry


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("unit_object", [False, True])
def test_fixed_varargs(ureg, count, unit_object):
    unit = ureg.meter if unit_object else "meter"

    @ureg.wraps(unit, unit)
    def total(*values):
        return sum(values)

    values = [ureg.Quantity(100, "cm"), ureg.Quantity(2, "m"), ureg.Quantity(300, "cm")]
    result = total(*values[:count])

    assert result.units == ureg.meter
    assert result.magnitude == sum([1, 2, 3][:count])


@pytest.mark.parametrize("count", [0, 1, 3])
def test_none_varargs_pass_through(ureg, count):
    @ureg.wraps(None, None)
    def collect(*values):
        return values

    values = [object(), ureg.Quantity(2, "m"), ureg.Quantity(3, "s")]
    result = collect(*values[:count])

    assert len(result) == count
    assert all(actual is expected for actual, expected in zip(result, values))


@pytest.mark.parametrize("count", [0, 1, 3])
def test_referenced_gcd(ureg, count):
    wrapped = ureg.wraps("=A", "=A")(math.gcd)
    magnitudes = [6, 4, 8][:count]
    result = wrapped(*(ureg.Quantity(value, "m") for value in magnitudes))

    assert result.magnitude == math.gcd(*magnitudes)
    assert result.units == (ureg.meter if count else ureg.dimensionless)


def test_references_reset_each_call(ureg):
    @ureg.wraps("=A", "=A")
    def total(*values):
        return sum(values)

    first = total(ureg.Quantity(100, "cm"), ureg.Quantity(2, "m"))
    second = total(ureg.Quantity(3, "s"), ureg.Quantity(2000, "ms"))
    empty = total()
    bare = total(2, 3)

    assert first.units == ureg.centimeter
    assert first.magnitude == 300
    assert second.units == ureg.second
    assert second.magnitude == 5
    assert empty.units == ureg.dimensionless
    assert empty.magnitude == 0
    assert bare.units == ureg.dimensionless
    assert bare.magnitude == 5


@pytest.mark.parametrize("reference", [False, True])
def test_zero_magnitude_keeps_units(ureg, reference):
    unit = "=A" if reference else "cm"

    @ureg.wraps(unit, unit)
    def total(*values):
        return sum(values)

    result = total(ureg.Quantity(0, "cm"), ureg.Quantity(3, "m"))

    assert result.units == ureg.centimeter
    assert result.magnitude == 300


@pytest.mark.parametrize("unit", ["m", "=A"])
def test_incompatible_later_vararg(ureg, unit):
    @ureg.wraps(unit, unit)
    def total(*values):
        return sum(values)

    with pytest.raises(DimensionalityError):
        total(ureg.Quantity(1, "m"), ureg.Quantity(2, "s"))


@pytest.mark.parametrize("invalid_position", [0, 1])
def test_strict_checks_every_vararg(ureg, invalid_position):
    @ureg.wraps("m", "m")
    def total(*values):
        return sum(values)

    values = [ureg.Quantity(1, "m"), ureg.Quantity(2, "m")]
    values[invalid_position] = 3
    with pytest.raises(ValueError, match="strict=True"):
        total(*values)


def test_strict_varargs_accept_strings(ureg):
    @ureg.wraps("m", "m")
    def total(*values):
        return sum(values)

    result = total("1 m", "200 cm", "3 m")

    assert result.units == ureg.meter
    assert result.magnitude == 6


def test_non_strict_varargs(ureg):
    @ureg.wraps("m", "m", strict=False)
    def total(*values):
        return sum(values)

    result = total(1, ureg.Quantity(200, "cm"), 3)

    assert result.units == ureg.meter
    assert result.magnitude == 6


@pytest.mark.parametrize("legacy_placeholder", [False, True])
@pytest.mark.parametrize("call", ["defaults", "many", "keywords", "keyword_head"])
def test_fixed_keyword_only_and_extra_keywords(ureg, legacy_placeholder, call):
    def collect(
        head=ureg.Quantity(100, "cm"),
        *tail,
        offset=ureg.Quantity(200, "cm"),
        note="default",
        **extras,
    ):
        return head, tail, offset, note, extras

    units = ("m", "m", "m", None)
    if legacy_placeholder:
        units += (None,)
    wrapped = ureg.wraps(None, units)(collect)
    extra = ureg.Quantity(5, "s")

    if call == "defaults":
        result = wrapped()
        expected = (1, (), 2, "default", {})
    elif call == "many":
        result = wrapped(
            ureg.Quantity(100, "cm"), ureg.Quantity(2, "m"), ureg.Quantity(300, "cm")
        )
        expected = (1, (2, 3), 2, "default", {})
    elif call == "keywords":
        result = wrapped(
            ureg.Quantity(100, "cm"),
            ureg.Quantity(2, "m"),
            ureg.Quantity(300, "cm"),
            offset=ureg.Quantity(400, "cm"),
            note="kept",
            extra=extra,
        )
        expected = (1, (2, 3), 4, "kept", {"extra": extra})
    else:
        result = wrapped(
            head=ureg.Quantity(100, "cm"), offset=ureg.Quantity(400, "cm"), extra=extra
        )
        expected = (1, (), 4, "default", {"extra": extra})

    assert result == expected
    if "extra" in result[-1]:
        assert result[-1]["extra"] is extra


@pytest.mark.parametrize("count", [0, 2])
def test_fixed_reference_with_varargs(ureg, count):
    @ureg.wraps("=A", ("=A", "=A"))
    def total(head, *tail):
        return head + sum(tail)

    values = [ureg.Quantity(300, "cm"), ureg.Quantity(4, "m")][:count]
    result = total(ureg.Quantity(2, "m"), *values)

    assert result.units == ureg.meter
    assert result.magnitude == 2 + sum([3, 4][:count])


@pytest.mark.parametrize("count", [0, 2])
def test_keyword_only_reference_for_varargs(ureg, count):
    @ureg.wraps("=A**2", ("=A**2", "=A"))
    def total(*areas, length):
        return sum(areas)

    values = [ureg.Quantity(10000, "cm**2"), ureg.Quantity(2, "m**2")][:count]
    result = total(*values, length=ureg.Quantity(3, "m"))

    assert result.units == ureg.meter**2
    assert result.magnitude == sum([1, 2][:count])


def test_empty_varargs_reference_does_not_rebind_to_keyword_dependency(ureg):
    @ureg.wraps(None, ("=A", "=A"))
    def collect(*values, tail):
        return values, tail

    with pytest.raises(DimensionalityError):
        collect(tail=ureg.Quantity(1, "m"))


def test_none_varargs_between_converted_parameters(ureg):
    @ureg.wraps(None, ("m", None, "s"))
    def collect(head, *tail, duration):
        return head, tail, duration

    value = ureg.Quantity(9, "kg")
    result = collect(
        ureg.Quantity(100, "cm"), value, duration=ureg.Quantity(2000, "ms")
    )

    assert result == (1, (value,), 2)
    assert result[1][0] is value


def test_positional_only_before_varargs(ureg):
    @ureg.wraps("m", ("m", "m", "m"))
    def total(head, /, *tail, offset=ureg.Quantity(100, "cm")):
        return head + sum(tail) + offset

    result = total(
        ureg.Quantity(1, "m"), ureg.Quantity(200, "cm"), offset=ureg.Quantity(300, "cm")
    )

    assert result.units == ureg.meter
    assert result.magnitude == 6


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unexpected"])
def test_invalid_python_calls_raise_typeerror(ureg, invalid):
    @ureg.wraps(None, ("m", "m", None))
    def collect(head, *tail, marker):
        return head, tail, marker

    with pytest.raises(TypeError):
        if invalid == "missing":
            collect(ureg.Quantity(1, "m"))
        elif invalid == "duplicate":
            collect(ureg.Quantity(1, "m"), head=ureg.Quantity(2, "m"), marker=True)
        else:
            collect(ureg.Quantity(1, "m"), marker=True, unexpected=True)


@pytest.mark.parametrize("reference", [False, True])
def test_array_varargs(ureg, reference):
    np = pytest.importorskip("numpy")
    unit = "=A" if reference else "m"

    @ureg.wraps(unit, unit)
    def total(*values):
        return sum(values)

    left = ureg.Quantity(np.array([1, 2]), "m")
    right = ureg.Quantity(np.array([300, 400]), "cm")
    result = total(left, right)

    assert result.units == ureg.meter
    np.testing.assert_array_equal(result.magnitude, [4, 6])
    np.testing.assert_array_equal(left.magnitude, [1, 2])
    np.testing.assert_array_equal(right.magnitude, [300, 400])


def test_referenced_multiple_returns(ureg):
    @ureg.wraps(("=A", "=A"), "=A")
    def extrema(*values):
        return min(values), max(values)

    result = extrema(ureg.Quantity(100, "cm"), ureg.Quantity(2, "m"))

    assert isinstance(result, tuple)
    assert all(value.units == ureg.centimeter for value in result)
    assert tuple(value.magnitude for value in result) == (100, 200)


def test_varargs_wrapper_metadata(ureg):
    def total(head, /, *values, offset=0, **kwargs):
        """Sum the supplied values."""
        return head + sum(values) + offset

    wrapped = ureg.wraps("m", ("m", "m", None))(total)

    assert inspect.signature(wrapped) == inspect.signature(total)
    assert wrapped.__name__ == total.__name__
    assert wrapped.__doc__ == total.__doc__
    assert wrapped.__wrapped__ is total
