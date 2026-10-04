from __future__ import annotations

from pint import UnitRegistry


def test_angular_mil_fits_6400_in_a_turn():
    ureg = UnitRegistry()
    assert (1 * ureg.turn).to("mil").magnitude == 6400
    assert (1 * ureg.degree).to("arcminute").magnitude == 60
