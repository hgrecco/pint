from __future__ import annotations

import pint


def _make_quantity():
    ureg = pint.UnitRegistry()
    return ureg.Quantity(1.5, "meter / second")


def test_caret_width_is_center_alignment():
    q = _make_quantity()
    assert f"{q:^30}" == f"{'1.5 meter / second':^30}"


def test_caret_width_with_unit_format_is_center_alignment():
    q = _make_quantity()
    assert f"{q:^30~P}" == f"{'1.5 m/s':^30}"


def test_fill_caret_width_is_center_alignment_with_fill():
    q = _make_quantity()
    assert f"{q:*^30}" == f"{'1.5 meter / second':*^30}"


def test_remove_custom_flags_keeps_alignment_caret():
    from pint.delegates.formatter._spec_helpers import remove_custom_flags

    assert remove_custom_flags("^30") == "^30"
    assert remove_custom_flags("*^30") == "*^30"


def test_extract_custom_flags_does_not_take_alignment_caret():
    from pint.delegates.formatter._spec_helpers import extract_custom_flags

    assert "^" not in extract_custom_flags("^30")
    assert "^" not in extract_custom_flags("*^30")
