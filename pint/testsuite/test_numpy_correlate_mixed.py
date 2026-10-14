import pytest

from pint.compat import np
from pint.testsuite import helpers


@pytest.mark.parametrize("quantity_position", ["a", "v", "both"])
@pytest.mark.parametrize("mode", ["valid", "same", "full"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@helpers.requires_numpy
def test_correlate_mixed_operands(sess_registry, quantity_position, mode, dtype):
    signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    template = np.array([0.5, 1.0], dtype=dtype)
    signal_before = signal.copy()
    template_before = template.copy()
    a = sess_registry.Quantity(signal, "meter") if quantity_position in ("a", "both") else signal
    v = sess_registry.Quantity(template, "second") if quantity_position in ("v", "both") else template
    result = np.correlate(a, v, mode=mode)
    units = (sess_registry.meter if quantity_position in ("a", "both") else 1) * (
        sess_registry.second if quantity_position in ("v", "both") else 1
    )
    expected = sess_registry.Quantity(np.correlate(signal_before, template_before, mode=mode), units)
    helpers.assert_quantity_almost_equal(result, expected)
    np.testing.assert_array_equal(signal, signal_before)
    np.testing.assert_array_equal(template, template_before)
