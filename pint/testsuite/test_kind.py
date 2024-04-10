from __future__ import annotations

import pytest

from pint import UnitRegistry


class TestKind:
    def test_torque_energy(self):
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
        moment_arm = Q_(1, "m").to_kind("[moment_arm]")
        force = Q_(1, "lbf").to_kind("[force]")
        # to_kind converts to the preferred_unit of the kind
        assert force.units == ureg.N

        # both force and moment_arm have kind defined.
        # Torque is defined in default_en:
        # [torque] = [force] * [moment_arm]
        # the result is a quantity with kind [torque]
        torque = force * moment_arm
        assert torque.units == ureg.N * ureg.m
        assert torque.kind == "[torque]"

        # Energy is defined in default_en:
        # [energy] = [force] * [length] = J
        distance = Q_(1, "m").to_kind("[length]")
        energy = force * distance
        assert energy.kind == "[energy]"
        assert energy.units == ureg.J

        # Torque is not energy so cannot be added
        with pytest.raises(ValueError):
            energy + torque
